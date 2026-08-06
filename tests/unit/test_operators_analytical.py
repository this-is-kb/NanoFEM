"""Operator recipes against hand-derived closed forms and cross-operator identities.

Every check builds its own small ``physical_gradients``/``normals`` batch by
hand (never through a real ``ShapeFunctionFamily``/``GeometricMapping``, since
that would pull mapping/interpolation into these tests) - the values are
textbook enough to verify by inspection.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.numerics.operators import (
    OperatorShapeError,
    UnsupportedDimensionError,
    curl_matrix,
    deviatoric_projector_voigt,
    divergence_matrix,
    gradient_matrix,
    helmholtz_matrix,
    identity_vector,
    laplacian_matrix,
    mass_term,
    surface_gradient_matrix,
    surface_projector,
    symmetric_gradient_matrix,
    tensor_transformation_matrix_voigt,
    trace_row_voigt,
    vector_transformation_matrix,
)
from nanofem.numerics.tensors import deviator, strain_to_voigt, trace
from nanofem.numerics.tensors.voigt import voigt_to_strain

# N1 = x, N2 = y on a 2-D domain: constant gradients, one quadrature point.
LINEAR_GRADIENTS_2D = np.array([[[1.0, 0.0], [0.0, 1.0]]])


# ---- gradient / divergence / curl -----------------------------------------------


def test_gradient_matrix_transposes_the_last_two_axes() -> None:
    result = gradient_matrix(LINEAR_GRADIENTS_2D)
    assert result.shape == (1, 2, 2)
    assert np.allclose(result[0], [[1.0, 0.0], [0.0, 1.0]])


def test_gradient_matrix_rejects_the_wrong_rank() -> None:
    with pytest.raises(OperatorShapeError):
        gradient_matrix(np.zeros((2, 2)))


def test_divergence_of_a_trace_free_field_is_zero() -> None:
    dofs = np.array([[1.0, 0.0], [0.0, -1.0]])  # u = (x, -y)
    row = divergence_matrix(LINEAR_GRADIENTS_2D)
    divergence = np.einsum("qoai,ai->qo", row, dofs)
    assert divergence[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_divergence_of_a_uniform_dilation() -> None:
    dofs = np.array([[1.0, 0.0], [0.0, 1.0]])  # u = (x, y)
    row = divergence_matrix(LINEAR_GRADIENTS_2D)
    divergence = np.einsum("qoai,ai->qo", row, dofs)
    assert divergence[0, 0] == pytest.approx(2.0)


def test_curl_of_a_rigid_rotation_is_two() -> None:
    dofs = np.array([[0.0, 1.0], [-1.0, 0.0]])  # u = (-y, x)
    row = curl_matrix(LINEAR_GRADIENTS_2D)
    theta = np.einsum("qoai,ai->qo", row, dofs)
    assert theta[0, 0] == pytest.approx(2.0)


def test_curl_of_a_gradient_field_is_zero() -> None:
    """curl(grad(phi)) = 0 for the potential phi = (x^2 + y^2)/2, whose gradient is (x, y)."""
    dofs = np.array([[1.0, 0.0], [0.0, 1.0]])
    row = curl_matrix(LINEAR_GRADIENTS_2D)
    theta = np.einsum("qoai,ai->qo", row, dofs)
    assert theta[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_curl_3d_matches_the_standard_formula() -> None:
    gradients = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    # u = (-y, x, 0): a rigid rotation about the z-axis, curl = (0, 0, 2).
    dofs = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    row = curl_matrix(gradients)
    curl = np.einsum("qkai,ai->qk", row, dofs)
    assert np.allclose(curl[0], [0.0, 0.0, 2.0])


def test_curl_rejects_dimension_one() -> None:
    with pytest.raises(UnsupportedDimensionError):
        curl_matrix(np.zeros((1, 2, 1)))


# ---- symmetric gradient (patch test) --------------------------------------------


def test_symmetric_gradient_recovers_a_prescribed_uniform_strain() -> None:
    eps0 = np.array([[0.2, -0.05], [-0.05, 0.1]])
    dofs = np.stack([eps0[:, 0], eps0[:, 1]], axis=0)
    matrix = symmetric_gradient_matrix(LINEAR_GRADIENTS_2D)
    strain_voigt = np.einsum("qvai,ai->qv", matrix, dofs)
    reconstructed = voigt_to_strain(strain_voigt)[0]
    assert np.allclose(reconstructed, eps0)


def test_symmetric_gradient_of_a_rigid_translation_is_zero() -> None:
    zero_gradients = np.zeros_like(LINEAR_GRADIENTS_2D)
    dofs = np.array([[3.0, -2.0], [3.0, -2.0]])  # same displacement everywhere
    matrix = symmetric_gradient_matrix(zero_gradients)
    strain_voigt = np.einsum("qvai,ai->qv", matrix, dofs)
    assert np.allclose(strain_voigt, 0.0)


# ---- cross-operator consistency --------------------------------------------------


def test_divergence_matches_trace_of_symmetric_gradient() -> None:
    rng = np.random.default_rng(99)
    gradients = rng.normal(size=(5, 3, 2))
    dofs = rng.normal(size=(3, 2))

    divergence = np.einsum("qoai,ai->qo", divergence_matrix(gradients), dofs)[:, 0]
    strain_voigt = np.einsum("qvai,ai->qv", symmetric_gradient_matrix(gradients), dofs)
    trace_of_strain = np.trace(voigt_to_strain(strain_voigt), axis1=-2, axis2=-1)

    assert np.allclose(divergence, trace_of_strain, atol=1e-10)


# ---- Laplacian / Helmholtz --------------------------------------------------------


def test_laplacian_matches_the_constant_strain_triangle_pattern() -> None:
    gradients = np.array([[[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]])
    weights = np.array([1.0])
    volume_scale = np.array([0.5])
    matrix = laplacian_matrix(gradients, weights, volume_scale)
    expected = np.array([[1.0, -0.5, -0.5], [-0.5, 0.5, 0.0], [-0.5, 0.0, 0.5]])
    assert np.allclose(matrix, expected)


def test_laplacian_matrix_is_symmetric_and_positive_semidefinite() -> None:
    rng = np.random.default_rng(5)
    gradients = rng.normal(size=(4, 3, 2))
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    volume_scale = np.ones(4)
    matrix = laplacian_matrix(gradients, weights, volume_scale)
    assert np.allclose(matrix, matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert np.all(eigenvalues >= -1e-10)


def test_helmholtz_collapses_to_the_mass_term_at_zero_length_scale() -> None:
    rng = np.random.default_rng(6)
    values = rng.normal(size=(3, 2))
    gradients = rng.normal(size=(3, 2, 2))
    weights = np.array([0.3, 0.3, 0.4])
    volume_scale = np.ones(3)

    zero_scale = helmholtz_matrix(values, gradients, weights, volume_scale, 0.0)
    mass = mass_term(values, weights, volume_scale)
    assert np.allclose(zero_scale, mass)


def test_helmholtz_remainder_is_length_scale_squared_times_laplacian() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(3, 2))
    gradients = rng.normal(size=(3, 2, 2))
    weights = np.array([0.3, 0.3, 0.4])
    volume_scale = np.ones(3)
    length_scale = 0.5

    full = helmholtz_matrix(values, gradients, weights, volume_scale, length_scale)
    mass = mass_term(values, weights, volume_scale)
    laplacian = laplacian_matrix(gradients, weights, volume_scale)
    assert np.allclose(full - mass, length_scale**2 * laplacian)


# ---- surface gradient -------------------------------------------------------------


def test_surface_projector_is_idempotent_symmetric_and_annihilates_the_normal() -> None:
    normals = np.array([[0.6, 0.8]])  # a unit normal, not axis-aligned
    projector = surface_projector(normals)[0]
    assert np.allclose(projector, projector.T)
    assert np.allclose(projector @ projector, projector)
    assert np.allclose(projector @ normals[0], 0.0, atol=1e-12)


def test_surface_gradient_reduces_to_the_projected_gradient() -> None:
    normals = np.array([[1.0, 0.0]])
    row = surface_gradient_matrix(LINEAR_GRADIENTS_2D, normals)
    projector = surface_projector(normals)[0]
    expected = np.einsum("kj,qja->qka", projector, gradient_matrix(LINEAR_GRADIENTS_2D))
    assert np.allclose(row[:, :, :, 0], expected)
    assert np.allclose(row[:, :, :, 1], expected)  # broadcast identically over components


# ---- Voigt operators ---------------------------------------------------------------


def test_identity_vector_recovers_the_trace() -> None:
    eps = np.array([[0.4, 0.1], [0.1, -0.3]])
    eps_voigt = strain_to_voigt(eps)
    assert identity_vector(2) @ eps_voigt == pytest.approx(float(trace(eps[np.newaxis])[0]))
    assert np.allclose(trace_row_voigt(2)[0], identity_vector(2))


def test_deviatoric_projector_voigt_matches_tensors_deviator() -> None:
    eps = np.array([[0.4, 0.1], [0.1, -0.3]])
    eps_voigt = strain_to_voigt(eps)
    projected = deviatoric_projector_voigt(2) @ eps_voigt
    expected = strain_to_voigt(deviator(eps[np.newaxis])[0])
    assert np.allclose(projected, expected)


# ---- transformation -----------------------------------------------------------------


def test_vector_transformation_matrix_is_the_rotation_itself() -> None:
    angle = np.pi / 4
    q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    assert np.allclose(vector_transformation_matrix(q), q)


def test_tensor_transformation_matrix_voigt_strain_is_stress_inverse_transpose() -> None:
    angle = 0.6
    q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    m_sigma = tensor_transformation_matrix_voigt(q, kind="stress")
    m_epsilon = tensor_transformation_matrix_voigt(q, kind="strain")
    assert np.allclose(m_epsilon, np.linalg.inv(m_sigma).T)
