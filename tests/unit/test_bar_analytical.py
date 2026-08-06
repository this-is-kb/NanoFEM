"""Bar element: closed-form stiffness against an independent hand computation."""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.elements.base import ElementDofSignature
from nanofem.elements.structural.bar import Bar
from nanofem.numerics.assembly.contributions import ContributionKind, OperatorRole
from nanofem.utils.exceptions import InputValidationError


def test_local_stiffness_matches_hand_formula() -> None:
    """K = (EA/L) [[1, -1], [-1, 1]] - never derived by calling nanofem code."""
    e, a, length = 210.0e9, 0.005, 2.0
    bar = Bar(0, (0, 1), np.array([[0.0], [length]]), (0, 1), young_modulus=e, area=a)
    k = e * a / length
    expected = k * np.array([[1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(bar.local_stiffness(), expected)
    assert bar.length == pytest.approx(length)


def test_dof_signature() -> None:
    bar = Bar(0, (0, 1), np.array([[0.0], [1.0]]), (0, 1), young_modulus=1.0, area=1.0)
    assert bar.dof_signature() == ElementDofSignature((("u.x",), ("u.x",)))


def test_transformation_matrix_is_identity_and_orthonormal() -> None:
    bar = Bar(0, (0, 1), np.array([[0.0], [1.0]]), (0, 1), young_modulus=1.0, area=1.0)
    t = bar.transformation_matrix()
    np.testing.assert_array_equal(t, np.eye(2))
    np.testing.assert_allclose(t.T @ t, np.eye(2))


def test_contributions_emit_stiffness_only() -> None:
    bar = Bar(0, (0, 1), np.array([[0.0], [1.0]]), (3, 7), young_modulus=1.0, area=1.0)
    (contribution,) = list(bar.contributions(OperatorRole.STIFFNESS))
    assert contribution.kind is ContributionKind.CELL
    np.testing.assert_array_equal(contribution.row_dofs, [3, 7])
    np.testing.assert_array_equal(contribution.col_dofs, [3, 7])
    np.testing.assert_allclose(contribution.block, bar.local_stiffness())
    assert list(bar.contributions(OperatorRole.MASS)) == []
    assert list(bar.contributions(OperatorRole.FORCE)) == []


def test_axial_response_matches_hand_formula() -> None:
    """strain=(u2-u1)/L, stress=E*strain, force=stress*A - constant along the member."""
    e, a, length = 210.0e9, 0.005, 2.0
    bar = Bar(0, (0, 1), np.array([[0.0], [length]]), (0, 1), young_modulus=e, area=a)
    u1, u2 = 0.0002, 0.0009
    response = bar.axial_response(np.array([u1, u2]))
    expected_strain = (u2 - u1) / length
    assert response.strain == pytest.approx(expected_strain, rel=1e-12)
    assert response.stress == pytest.approx(e * expected_strain, rel=1e-12)
    assert response.force == pytest.approx(e * expected_strain * a, rel=1e-12)


def test_axial_response_rejects_wrong_shape() -> None:
    bar = Bar(0, (0, 1), np.array([[0.0], [1.0]]), (0, 1), young_modulus=1.0, area=1.0)
    with pytest.raises(InputValidationError):
        bar.axial_response(np.array([0.0, 1.0, 2.0]))


def test_rejects_non_positive_length_and_bad_shapes() -> None:
    with pytest.raises(InputValidationError, match="positive length"):
        Bar(0, (0, 1), np.array([[1.0], [1.0]]), (0, 1), young_modulus=1.0, area=1.0)
    with pytest.raises(InputValidationError, match="positive length"):
        Bar(0, (0, 1), np.array([[1.0], [0.0]]), (0, 1), young_modulus=1.0, area=1.0)
    with pytest.raises(InputValidationError, match=r"\(2, 1\)"):
        Bar(0, (0, 1), np.array([[0.0, 0.0], [1.0, 0.0]]), (0, 1), young_modulus=1.0, area=1.0)


def test_rejects_non_positive_material_or_section_properties() -> None:
    with pytest.raises(InputValidationError):
        Bar(0, (0, 1), np.array([[0.0], [1.0]]), (0, 1), young_modulus=-1.0, area=1.0)
    with pytest.raises(InputValidationError):
        Bar(0, (0, 1), np.array([[0.0], [1.0]]), (0, 1), young_modulus=1.0, area=0.0)
