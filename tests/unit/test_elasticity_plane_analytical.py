"""PlaneStressConstitutive/PlaneStrainConstitutive: declarations + SDS Section 5 checks."""

from __future__ import annotations

import numpy as np

from nanofem.physics.elasticity.plane import PlaneStrainConstitutive, PlaneStressConstitutive
from nanofem.state.layout import StateLayout


def _plane_stress_matrix(e: float, nu: float) -> np.ndarray:
    c = e / (1.0 - nu**2)
    return np.array([[c, c * nu, 0.0], [c * nu, c, 0.0], [0.0, 0.0, c * (1.0 - nu) / 2.0]])


def _plane_strain_matrix(e: float, nu: float) -> np.ndarray:
    c = e / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return np.array(
        [
            [c * (1.0 - nu), c * nu, 0.0],
            [c * nu, c * (1.0 - nu), 0.0],
            [0.0, 0.0, c * (1.0 - 2.0 * nu) / 2.0],
        ]
    )


def test_plane_stress_declarations() -> None:
    model = PlaneStressConstitutive()
    assert model.required_properties() == ("E", "nu")
    assert model.state_layout() == StateLayout(())
    assert model.response_components() == 3


def test_plane_strain_declarations() -> None:
    model = PlaneStrainConstitutive()
    assert model.required_properties() == ("E", "nu")
    assert model.state_layout() == StateLayout(())
    assert model.response_components() == 3


def test_plane_stress_matches_hand_matrix() -> None:
    model = PlaneStressConstitutive()
    for e, nu in [(200e9, 0.3), (71.7e9, 0.25), (10e9, -0.2)]:
        strain = np.array([[[0.001, -0.0007, 0.0004]]])
        properties = {"E": np.array([[e]]), "nu": np.array([[nu]])}
        stress, tangent = model.respond_batch(strain, properties)
        expected_d = _plane_stress_matrix(e, nu)
        np.testing.assert_allclose(tangent[0, 0], expected_d, rtol=1e-12)
        np.testing.assert_allclose(stress[0, 0], expected_d @ strain[0, 0], rtol=1e-9)


def test_plane_strain_matches_hand_matrix() -> None:
    model = PlaneStrainConstitutive()
    for e, nu in [(200e9, 0.3), (71.7e9, 0.25), (10e9, -0.2)]:
        strain = np.array([[[0.001, -0.0007, 0.0004]]])
        properties = {"E": np.array([[e]]), "nu": np.array([[nu]])}
        stress, tangent = model.respond_batch(strain, properties)
        expected_d = _plane_strain_matrix(e, nu)
        np.testing.assert_allclose(tangent[0, 0], expected_d, rtol=1e-12)
        np.testing.assert_allclose(stress[0, 0], expected_d @ strain[0, 0], rtol=1e-9)


def test_plane_stress_and_plane_strain_are_genuinely_different() -> None:
    e, nu = 200e9, 0.3
    stress_d = _plane_stress_matrix(e, nu)
    strain_d = _plane_strain_matrix(e, nu)
    assert not np.allclose(stress_d, strain_d)


def test_tangent_matches_finite_difference_per_component() -> None:
    """SDS Section 5 consistency condition 2, checked for each Voigt component."""
    for model_cls in (PlaneStressConstitutive, PlaneStrainConstitutive):
        model = model_cls()
        properties = {"E": np.array([[200e9]]), "nu": np.array([[0.3]])}
        strain0 = np.array([[[0.0021, -0.0013, 0.0009]]])
        h = 1e-9
        for component in range(3):
            perturbation = np.zeros_like(strain0)
            perturbation[..., component] = h
            stress_plus, _ = model.respond_batch(strain0 + perturbation, properties)
            stress_minus, _ = model.respond_batch(strain0 - perturbation, properties)
            fd_column = (stress_plus - stress_minus) / (2.0 * h)
            _, tangent = model.respond_batch(strain0, properties)
            np.testing.assert_allclose(tangent[0, 0, :, component], fd_column[0, 0], rtol=1e-6)


def test_null_response() -> None:
    """SDS Section 5 consistency condition 3: zero strain yields zero stress."""
    for model_cls in (PlaneStressConstitutive, PlaneStrainConstitutive):
        model = model_cls()
        strains = np.zeros((1, 1, 3))
        properties = {"E": np.array([[200e9]]), "nu": np.array([[0.3]])}
        stress, _ = model.respond_batch(strains, properties)
        np.testing.assert_allclose(stress, 0.0)


def test_tangent_is_symmetric() -> None:
    """Hyperelastic work conjugacy (SDS Section 5 condition 1): D == D^T."""
    for model_cls in (PlaneStressConstitutive, PlaneStrainConstitutive):
        model = model_cls()
        properties = {"E": np.array([[200e9]]), "nu": np.array([[0.3]])}
        strain0 = np.zeros((1, 1, 3))
        _, tangent = model.respond_batch(strain0, properties)
        np.testing.assert_allclose(tangent[0, 0], tangent[0, 0].T)
