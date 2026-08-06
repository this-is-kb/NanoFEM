"""Timoshenko beam element: closed-form stiffness against an independent hand computation."""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.elements.base import ElementDofSignature
from nanofem.elements.structural.beam_timoshenko import TimoshenkoBeam
from nanofem.numerics.assembly.contributions import ContributionKind, OperatorRole
from nanofem.utils.exceptions import InputValidationError


def _beam(
    length: float = 2.0,
    e: float = 210.0e9,
    g: float = 80.0e9,
    i: float = 8.0e-8,
    a_s: float = 0.001,
    global_dofs: tuple[int, int, int, int] = (0, 1, 2, 3),
) -> TimoshenkoBeam:
    return TimoshenkoBeam(
        0,
        (0, 1),
        np.array([[0.0], [length]]),
        global_dofs,
        young_modulus=e,
        shear_modulus=g,
        second_moment=i,
        shear_area=a_s,
    )


def test_local_stiffness_matches_hand_formula() -> None:
    """The selective-reduced-integration closed form, independently hand-built."""
    e, g, i, a_s, length = 210.0e9, 80.0e9, 8.0e-8, 0.001, 2.0
    beam = _beam(length, e, g, i, a_s)
    ei = e * i
    gas = g * a_s
    expected = np.array(
        [
            [gas / length, gas / 2.0, -gas / length, gas / 2.0],
            [
                gas / 2.0,
                ei / length + gas * length / 4.0,
                -gas / 2.0,
                -ei / length + gas * length / 4.0,
            ],
            [-gas / length, -gas / 2.0, gas / length, -gas / 2.0],
            [
                gas / 2.0,
                -ei / length + gas * length / 4.0,
                -gas / 2.0,
                ei / length + gas * length / 4.0,
            ],
        ]
    )
    np.testing.assert_allclose(beam.local_stiffness(), expected)
    assert beam.length == pytest.approx(length)


def test_dof_signature() -> None:
    beam = _beam()
    assert beam.dof_signature() == ElementDofSignature((("u.y", "r.z"), ("u.y", "r.z")))


def test_transformation_matrix_is_identity_and_orthonormal() -> None:
    beam = _beam()
    t = beam.transformation_matrix()
    np.testing.assert_array_equal(t, np.eye(4))
    np.testing.assert_allclose(t.T @ t, np.eye(4))


def test_contributions_emit_stiffness_only() -> None:
    beam = _beam(global_dofs=(3, 7, 11, 13))
    (contribution,) = list(beam.contributions(OperatorRole.STIFFNESS))
    assert contribution.kind is ContributionKind.CELL
    np.testing.assert_array_equal(contribution.row_dofs, [3, 7, 11, 13])
    np.testing.assert_array_equal(contribution.col_dofs, [3, 7, 11, 13])
    np.testing.assert_allclose(contribution.block, beam.local_stiffness())
    assert list(beam.contributions(OperatorRole.MASS)) == []
    assert list(beam.contributions(OperatorRole.FORCE)) == []


def test_rejects_non_positive_length_and_bad_shapes() -> None:
    with pytest.raises(InputValidationError, match="positive length"):
        TimoshenkoBeam(
            0,
            (0, 1),
            np.array([[1.0], [1.0]]),
            (0, 1, 2, 3),
            young_modulus=1.0,
            shear_modulus=1.0,
            second_moment=1.0,
            shear_area=1.0,
        )
    with pytest.raises(InputValidationError, match="positive length"):
        TimoshenkoBeam(
            0,
            (0, 1),
            np.array([[1.0], [0.0]]),
            (0, 1, 2, 3),
            young_modulus=1.0,
            shear_modulus=1.0,
            second_moment=1.0,
            shear_area=1.0,
        )
    with pytest.raises(InputValidationError, match=r"\(2, 1\)"):
        TimoshenkoBeam(
            0,
            (0, 1),
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            (0, 1, 2, 3),
            young_modulus=1.0,
            shear_modulus=1.0,
            second_moment=1.0,
            shear_area=1.0,
        )


def test_rejects_non_positive_material_or_section_properties() -> None:
    for kwargs in (
        {"young_modulus": -1.0},
        {"shear_modulus": -1.0},
        {"second_moment": 0.0},
        {"shear_area": 0.0},
    ):
        base = {
            "young_modulus": 1.0,
            "shear_modulus": 1.0,
            "second_moment": 1.0,
            "shear_area": 1.0,
        }
        base.update(kwargs)
        with pytest.raises(InputValidationError):
            TimoshenkoBeam(0, (0, 1), np.array([[0.0], [1.0]]), (0, 1, 2, 3), **base)


def test_stiffness_matrix_is_symmetric_and_rigid_body_null() -> None:
    """K == K^T; rigid translation and rigid rotation both produce zero net force/moment.

    The rigid-rotation mode about node 1 is (w1,theta1,w2,theta2) = (0, 1, L, 1)
    - w2 = L * theta, not (0,1,0,1) - matching EulerBernoulliBeam's own null vector.
    """
    length = 3.0
    beam = _beam(length=length, e=200e9, g=80e9, i=1e-6, a_s=0.001)
    k = beam.local_stiffness()
    np.testing.assert_allclose(k, k.T)
    np.testing.assert_allclose(k @ np.array([1.0, 0.0, 1.0, 0.0]), 0.0, atol=1e-6)
    np.testing.assert_allclose(k @ np.array([0.0, 1.0, length, 1.0]), 0.0, atol=1e-6)
