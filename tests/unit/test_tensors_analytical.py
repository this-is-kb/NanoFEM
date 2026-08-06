"""Second- and fourth-order tensor algebra against hand-worked closed forms.

Every check here uses a matrix small enough to verify by hand; the isotropic
oracle is checked against the independently-stated Lame form, never against
itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.numerics.tensors import (
    TensorError,
    apply,
    determinant,
    deviator,
    deviatoric_projector,
    double_contraction,
    frobenius_norm,
    has_major_symmetry,
    has_minor_symmetry,
    identity_fourth,
    inverse,
    is_symmetric,
    isotropic_stiffness,
    outer,
    single_contraction,
    skew_part,
    symmetric_outer,
    symmetric_part,
    symmetrizer,
    trace,
    volumetric_projector,
)

# ---- second order -----------------------------------------------------------


def test_symmetric_and_skew_parts_sum_to_the_original() -> None:
    a = np.array([[[1.0, 2.0], [4.0, 3.0]]])
    assert np.allclose(symmetric_part(a) + skew_part(a), a)
    assert np.allclose(symmetric_part(a)[0], [[1.0, 3.0], [3.0, 3.0]])
    assert np.allclose(skew_part(a)[0], [[0.0, -1.0], [1.0, 0.0]])


def test_trace_and_determinant_of_a_known_matrix() -> None:
    a = np.array([[[2.0, 0.0], [0.0, 3.0]]])
    assert trace(a)[0] == pytest.approx(5.0)
    assert determinant(a)[0] == pytest.approx(6.0)


def test_inverse_of_a_known_matrix() -> None:
    a = np.array([[[2.0, 0.0], [0.0, 4.0]]])
    assert np.allclose(inverse(a)[0], [[0.5, 0.0], [0.0, 0.25]])


def test_inverse_raises_tensor_error_on_a_singular_matrix() -> None:
    a = np.array([[[1.0, 1.0], [1.0, 1.0]]])
    with pytest.raises(TensorError):
        inverse(a)


def test_deviator_removes_the_trace() -> None:
    a = np.array([[[4.0, 0.0], [0.0, 2.0]]])  # trace 6, dim 2 -> volumetric part 3*I
    assert np.allclose(deviator(a)[0], [[1.0, 0.0], [0.0, -1.0]])
    isotropic = np.array([[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]])
    assert np.allclose(deviator(isotropic), 0.0)


def test_frobenius_norm_of_a_known_matrix() -> None:
    a = np.array([[[3.0, 0.0], [0.0, 4.0]]])
    assert frobenius_norm(a)[0] == pytest.approx(5.0)


def test_outer_and_symmetric_outer() -> None:
    u = np.array([[1.0, 0.0]])
    v = np.array([[0.0, 1.0]])
    assert np.allclose(outer(u, v)[0], [[0.0, 1.0], [0.0, 0.0]])
    assert np.allclose(symmetric_outer(u, v)[0], [[0.0, 0.5], [0.5, 0.0]])


def test_single_and_double_contraction() -> None:
    a = np.array([[[2.0, 0.0], [0.0, 3.0]]])
    v = np.array([[1.0, 1.0]])
    assert np.allclose(single_contraction(a, v)[0], [2.0, 3.0])
    b = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    assert double_contraction(a, b)[0] == pytest.approx(5.0)  # A:I = tr(A)


def test_is_symmetric_true_and_false() -> None:
    symmetric = np.array([[[1.0, 2.0], [2.0, 1.0]]])
    asymmetric = np.array([[[1.0, 2.0], [3.0, 1.0]]])
    assert bool(is_symmetric(symmetric)[0])
    assert not bool(is_symmetric(asymmetric)[0])


# ---- fourth order -------------------------------------------------------------


def test_identity_fourth_acts_as_the_identity_on_symmetric_tensors() -> None:
    eps = np.array([[1.0, 0.3], [0.3, -0.5]])
    assert np.allclose(apply(identity_fourth(2), eps), eps)


def test_symmetrizer_acts_as_the_identity_on_symmetric_tensors() -> None:
    eps = np.array([[1.0, 0.3, 0.0], [0.3, -0.5, 0.1], [0.0, 0.1, 0.2]])
    assert np.allclose(apply(symmetrizer(3), eps), eps)


def test_volumetric_and_deviatoric_projectors_split_the_strain() -> None:
    eps = np.array([[1.0, 0.3], [0.3, -0.5]])
    volumetric_part = apply(volumetric_projector(2), eps)
    deviatoric_part = apply(deviatoric_projector(2), eps)
    expected_volumetric = (np.trace(eps) / 2.0) * np.eye(2)
    assert np.allclose(volumetric_part, expected_volumetric)
    assert np.allclose(deviatoric_part, deviator(eps[np.newaxis])[0])
    assert np.allclose(volumetric_part + deviatoric_part, eps)


def test_isotropic_stiffness_matches_the_lame_closed_form() -> None:
    bulk_modulus, shear_modulus = 5.0, 2.0
    dim = 3
    lame_lambda = bulk_modulus - 2.0 * shear_modulus / dim
    eps = np.array([[0.1, 0.02, 0.0], [0.02, -0.05, 0.01], [0.0, 0.01, 0.03]])
    stiffness = isotropic_stiffness(bulk_modulus, shear_modulus, dim)
    expected = lame_lambda * np.trace(eps) * np.eye(dim) + 2.0 * shear_modulus * eps
    assert np.allclose(apply(stiffness, eps), expected)


def test_has_major_and_minor_symmetry_for_the_isotropic_tensor() -> None:
    stiffness = isotropic_stiffness(5.0, 2.0, 3)
    assert bool(has_major_symmetry(stiffness))
    assert bool(has_minor_symmetry(stiffness))


def test_has_minor_symmetry_catches_a_deliberately_broken_tensor() -> None:
    stiffness = isotropic_stiffness(5.0, 2.0, 3).copy()
    stiffness[0, 1, 0, 0] += 1.0  # breaks C_ijkl == C_jikl
    assert not bool(has_minor_symmetry(stiffness))


def test_has_major_symmetry_catches_a_deliberately_broken_tensor() -> None:
    stiffness = isotropic_stiffness(5.0, 2.0, 3).copy()
    stiffness[0, 0, 1, 1] += 1.0  # breaks C_ijkl == C_klij alone
    assert not bool(has_major_symmetry(stiffness))
