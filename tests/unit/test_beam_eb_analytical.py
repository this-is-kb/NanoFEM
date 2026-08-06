"""Euler-Bernoulli beam element: closed-form stiffness against an independent hand computation."""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.elements.base import ElementDofSignature
from nanofem.elements.structural.beam_eb import EulerBernoulliBeam
from nanofem.numerics.assembly.contributions import ContributionKind, OperatorRole
from nanofem.utils.exceptions import InputValidationError


def test_local_stiffness_matches_hand_formula() -> None:
    """K = (EI/L^3)[[12,6L,-12,6L],[6L,4L^2,-6L,2L^2],[-12,-6L,12,-6L],[6L,2L^2,-6L,4L^2]]."""
    e, i, length = 210.0e9, 8.0e-8, 2.0
    beam = EulerBernoulliBeam(
        0, (0, 1), np.array([[0.0], [length]]), (0, 1, 2, 3), young_modulus=e, second_moment=i
    )
    ei = e * i
    expected = (ei / length**3) * np.array(
        [
            [12.0, 6 * length, -12.0, 6 * length],
            [6 * length, 4 * length**2, -6 * length, 2 * length**2],
            [-12.0, -6 * length, 12.0, -6 * length],
            [6 * length, 2 * length**2, -6 * length, 4 * length**2],
        ]
    )
    np.testing.assert_allclose(beam.local_stiffness(), expected)
    assert beam.length == pytest.approx(length)


def test_dof_signature() -> None:
    beam = EulerBernoulliBeam(
        0, (0, 1), np.array([[0.0], [1.0]]), (0, 1, 2, 3), young_modulus=1.0, second_moment=1.0
    )
    assert beam.dof_signature() == ElementDofSignature((("u.y", "r.z"), ("u.y", "r.z")))


def test_transformation_matrix_is_identity_and_orthonormal() -> None:
    beam = EulerBernoulliBeam(
        0, (0, 1), np.array([[0.0], [1.0]]), (0, 1, 2, 3), young_modulus=1.0, second_moment=1.0
    )
    t = beam.transformation_matrix()
    np.testing.assert_array_equal(t, np.eye(4))
    np.testing.assert_allclose(t.T @ t, np.eye(4))


def test_contributions_emit_stiffness_only() -> None:
    beam = EulerBernoulliBeam(
        0,
        (0, 1),
        np.array([[0.0], [1.0]]),
        (3, 7, 11, 13),
        young_modulus=1.0,
        second_moment=1.0,
    )
    (contribution,) = list(beam.contributions(OperatorRole.STIFFNESS))
    assert contribution.kind is ContributionKind.CELL
    np.testing.assert_array_equal(contribution.row_dofs, [3, 7, 11, 13])
    np.testing.assert_array_equal(contribution.col_dofs, [3, 7, 11, 13])
    np.testing.assert_allclose(contribution.block, beam.local_stiffness())
    assert list(beam.contributions(OperatorRole.MASS)) == []
    assert list(beam.contributions(OperatorRole.FORCE)) == []


def test_rejects_non_positive_length_and_bad_shapes() -> None:
    with pytest.raises(InputValidationError, match="positive length"):
        EulerBernoulliBeam(
            0, (0, 1), np.array([[1.0], [1.0]]), (0, 1, 2, 3), young_modulus=1.0, second_moment=1.0
        )
    with pytest.raises(InputValidationError, match="positive length"):
        EulerBernoulliBeam(
            0, (0, 1), np.array([[1.0], [0.0]]), (0, 1, 2, 3), young_modulus=1.0, second_moment=1.0
        )
    with pytest.raises(InputValidationError, match=r"\(2, 1\)"):
        EulerBernoulliBeam(
            0,
            (0, 1),
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            (0, 1, 2, 3),
            young_modulus=1.0,
            second_moment=1.0,
        )


def test_rejects_non_positive_material_or_section_properties() -> None:
    with pytest.raises(InputValidationError):
        EulerBernoulliBeam(
            0, (0, 1), np.array([[0.0], [1.0]]), (0, 1, 2, 3), young_modulus=-1.0, second_moment=1.0
        )
    with pytest.raises(InputValidationError):
        EulerBernoulliBeam(
            0, (0, 1), np.array([[0.0], [1.0]]), (0, 1, 2, 3), young_modulus=1.0, second_moment=0.0
        )


def test_stiffness_matrix_is_symmetric_and_rigid_body_null() -> None:
    """K == K^T; rigid translation and rigid rotation both produce zero net force/moment.

    The rigid-rotation mode of a beam rotating about node 1 is (w1,theta1,w2,theta2)
    = (0, 1, L, 1) - w2 = L * theta, not (0,1,0,1) - since a rigid rotation about
    node 1 carries node 2 through an arc of length L*theta to first order.
    """
    length = 3.0
    beam = EulerBernoulliBeam(
        0,
        (0, 1),
        np.array([[0.0], [length]]),
        (0, 1, 2, 3),
        young_modulus=200e9,
        second_moment=1e-6,
    )
    k = beam.local_stiffness()
    np.testing.assert_allclose(k, k.T)
    np.testing.assert_allclose(k @ np.array([1.0, 0.0, 1.0, 0.0]), 0.0, atol=1e-6)
    np.testing.assert_allclose(k @ np.array([0.0, 1.0, length, 1.0]), 0.0, atol=1e-6)


def test_curvature_response_matches_the_classical_cantilever_moment() -> None:
    """A cantilever under a tip load P: recovered M(fixed end) = P*L, M(tip) = 0.

    Verified independently (both symbolically against the classical closed form and
    numerically against a from-scratch finite-difference curvature of the raw Hermite
    polynomial) before this method was written - see the scratch derivation referenced
    in ``docs/design/ERINGEN_DIFFERENTIAL_BAR.md``'s sibling note in ``docs/dev/notes.md``.
    """
    e, i, length, p = 200e9, 8.0e-6, 2.0, 1000.0
    beam = EulerBernoulliBeam(
        0, (0, 1), np.array([[0.0], [length]]), (0, 1, 2, 3), young_modulus=e, second_moment=i
    )
    k = beam.local_stiffness()
    free = [2, 3]
    k_ff = k[np.ix_(free, free)]
    f_f = np.array([p, 0.0])
    w2, theta2 = np.linalg.solve(k_ff, f_f)
    assert w2 == pytest.approx(p * length**3 / (3.0 * e * i), rel=1e-9)
    assert theta2 == pytest.approx(p * length**2 / (2.0 * e * i), rel=1e-9)

    response = beam.curvature_response(np.array([0.0, 0.0, w2, theta2]))
    assert response.moment[0] == pytest.approx(p * length, rel=1e-9)
    assert response.moment[1] == pytest.approx(0.0, abs=1e-6)


def test_curvature_response_rejects_wrong_shape() -> None:
    beam = EulerBernoulliBeam(
        0, (0, 1), np.array([[0.0], [1.0]]), (0, 1, 2, 3), young_modulus=1.0, second_moment=1.0
    )
    with pytest.raises(InputValidationError):
        beam.curvature_response(np.array([0.0, 0.0, 0.0]))
