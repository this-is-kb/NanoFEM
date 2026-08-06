"""SO(d) rotation, the Bond transformation pair, and tensor invariants/spectra.

Rotation checks use hand-built matrices (a known 2-D angle, a reflection) so
the reference values are independent of the rotation code under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.numerics.tensors import (
    NotRotationError,
    bond_matrix_strain,
    bond_matrix_stress,
    deviatoric_second_invariant,
    first_invariant,
    is_rotation,
    principal_directions,
    rotate_second_order,
    rotate_stiffness_voigt,
    rotate_vector,
    strain_to_voigt,
    stress_to_voigt,
    third_invariant,
    von_mises,
)


def _rotation_2d(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


# ---- rotation validity ----------------------------------------------------------


def test_is_rotation_true_for_identity_and_a_known_angle() -> None:
    assert is_rotation(np.eye(3))
    assert is_rotation(_rotation_2d(np.pi / 3))


def test_is_rotation_false_for_a_reflection() -> None:
    reflection = np.array([[1.0, 0.0], [0.0, -1.0]])  # det = -1
    assert not is_rotation(reflection)


def test_rotate_functions_raise_not_rotation_error_for_a_reflection() -> None:
    reflection = np.array([[1.0, 0.0], [0.0, -1.0]])
    with pytest.raises(NotRotationError):
        rotate_vector(reflection, np.array([1.0, 0.0]))
    with pytest.raises(NotRotationError):
        rotate_second_order(reflection, np.eye(2))
    with pytest.raises(NotRotationError):
        bond_matrix_stress(reflection)


# ---- rotation against hand-built references --------------------------------------


def test_rotate_vector_matches_a_hand_built_2d_rotation() -> None:
    q = _rotation_2d(np.pi / 2)  # 90 degrees counterclockwise
    v = np.array([1.0, 0.0])
    assert np.allclose(rotate_vector(q, v), [0.0, 1.0], atol=1e-12)


def test_rotate_second_order_matches_a_hand_built_transform() -> None:
    q = _rotation_2d(np.pi / 2)
    a = np.array([[1.0, 0.0], [0.0, 0.0]])  # uniaxial along x
    rotated = rotate_second_order(q, a)
    assert np.allclose(rotated, [[0.0, 0.0], [0.0, 1.0]], atol=1e-12)  # uniaxial along y


def test_rotate_second_order_round_trip() -> None:
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(raw)
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    tensor = 0.5 * (raw + raw.T)
    round_tripped = rotate_second_order(q.T, rotate_second_order(q, tensor))
    assert np.allclose(round_tripped, tensor)


# ---- Bond transformation pair -----------------------------------------------------


def test_bond_matrices_satisfy_the_inverse_transpose_identity() -> None:
    q = _rotation_2d(0.4)
    m_sigma = bond_matrix_stress(q)
    m_epsilon = bond_matrix_strain(q)
    assert np.allclose(m_epsilon, np.linalg.inv(m_sigma).T)


def test_bond_rotation_preserves_work() -> None:
    q = _rotation_2d(0.7)
    sigma = np.array([[2.0, 0.3], [0.3, 1.0]])
    eps = np.array([[0.1, 0.02], [0.02, -0.05]])

    work_before = float(np.einsum("ij,ij->", sigma, eps))

    sigma_voigt_rotated = bond_matrix_stress(q) @ stress_to_voigt(sigma)
    eps_voigt_rotated = bond_matrix_strain(q) @ strain_to_voigt(eps)
    work_after = float(sigma_voigt_rotated @ eps_voigt_rotated)

    assert work_after == pytest.approx(work_before)


def test_rotate_stiffness_voigt_matches_bond_matrix_construction() -> None:
    q = _rotation_2d(0.2)
    d_voigt = np.diag([10.0, 12.0, 3.0])
    m_sigma = bond_matrix_stress(q)
    expected = m_sigma @ d_voigt @ m_sigma.T
    assert np.allclose(rotate_stiffness_voigt(d_voigt, q), expected)


# ---- invariants and spectra --------------------------------------------------------


def test_von_mises_uniaxial_closed_form() -> None:
    sigma = np.zeros((1, 3, 3))
    sigma[0, 0, 0] = 7.0
    assert von_mises(sigma)[0] == pytest.approx(7.0)


def test_first_and_third_invariants_agree_with_principal_values() -> None:
    tensor = np.array([[[2.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, -1.0]]])
    values, _ = principal_directions(tensor)
    assert first_invariant(tensor)[0] == pytest.approx(np.sum(values[0]))
    assert third_invariant(tensor)[0] == pytest.approx(np.prod(values[0]))


def test_deviatoric_second_invariant_is_nonnegative() -> None:
    rng = np.random.default_rng(3)
    raw = rng.normal(size=(5, 3, 3))
    tensor = 0.5 * (raw + np.swapaxes(raw, -1, -2))
    assert np.all(deviatoric_second_invariant(tensor) >= 0.0)


def test_principal_directions_reconstruct_the_tensor() -> None:
    tensor = np.array([[[2.0, 1.0], [1.0, 3.0]]])
    values, vectors = principal_directions(tensor)
    reconstructed = np.einsum("...ij,...j,...kj->...ik", vectors, values, vectors)
    assert np.allclose(reconstructed, tensor)
