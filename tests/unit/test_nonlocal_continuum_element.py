"""NonlocalContinuumElement: the coupled (u, e*) block stiffness, verified before it shipped.

Mirrors T3/Q4's own verification style (constant-strain patch test, symmetry, rigid-body null
space) one level up: this is a *mixed* element, so "patch test" here means the Schur-complement
effective stiffness (eliminating e*) matches the classical T3 stiffness exactly at ``e0a=0``,
and the recovered nonlocal strain matches the classical strain exactly for ANY ``e0a`` on a
constant-strain field (a genuine 2-D analogue of the 1-D nonlocal bar's "uniform load leaves no
nonlocal trace" finding, v0.20.0's Peddieson-paradox test).

A genuine sign bug was caught here, not by any check in this file but by a full
``Model``-pipeline solve (``test_static_nonlocal_plate.py``) - see ``docs/dev/notes.md`` and
``elements/continuum/nonlocal_continuum.py``'s own module docstring for the full account. The
lesson is reflected in this file's own test list: the Schur-complement/constant-strain checks
below are kept (they are still real, valid checks of the u-e* *relationship*), but
``test_static_nonlocal_plate.py``'s full monolithic solve is the one that actually pins the
correct sign, precisely because it is not a substitution-based check.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.elements.continuum.continuum import ContinuumElement
from nanofem.elements.continuum.nonlocal_continuum import NonlocalContinuumElement
from nanofem.materials.material import Material
from nanofem.numerics.reference.enums import CellType
from nanofem.physics.elasticity.eringen_differential import (
    EringenDifferentialMaterial,
    EringenDifferentialTheory,
)
from nanofem.physics.elasticity.isotropic import IsotropicElasticity
from nanofem.physics.elasticity.plane import PlaneStressConstitutive
from nanofem.utils.exceptions import InputValidationError

YOUNG_MODULUS = 200e9
POISSON = 0.3
_SCALENE_TRIANGLE = np.array([[0.3, -0.2], [2.1, 0.4], [0.9, 1.7]])
N_DOF_U = 6  # 3 nodes x 2 components
N_DOF_E = 9  # 3 nodes x 3 Voigt components
N_DOF = N_DOF_U + N_DOF_E


def _nonlocal_element(
    e0a: float, coordinates: np.ndarray = _SCALENE_TRIANGLE
) -> NonlocalContinuumElement:
    theory = EringenDifferentialTheory(dim=2)
    material = Material("steel", E=YOUNG_MODULUS, nu=POISSON, e0a=e0a)
    constitutive = EringenDifferentialMaterial(PlaneStressConstitutive())
    return NonlocalContinuumElement(
        cell_id=0,
        node_ids=(0, 1, 2),
        coordinates=coordinates,
        global_dofs=np.arange(N_DOF, dtype=np.int64),
        cell_type=CellType.TRIANGLE,
        interpolation_order=1,
        theory=theory,
        constitutive=constitutive,
        material=material,
    )


def _classical_element(coordinates: np.ndarray = _SCALENE_TRIANGLE) -> ContinuumElement:
    return ContinuumElement(
        cell_id=0,
        node_ids=(0, 1, 2),
        coordinates=coordinates,
        global_dofs=np.arange(N_DOF_U, dtype=np.int64),
        cell_type=CellType.TRIANGLE,
        interpolation_order=1,
        theory=IsotropicElasticity(dim=2),
        constitutive=PlaneStressConstitutive(),
        material=Material("steel", E=YOUNG_MODULUS, nu=POISSON),
    )


def test_local_stiffness_is_symmetric() -> None:
    element = _nonlocal_element(e0a=0.3)
    k = element.local_stiffness()
    assert k.shape == (N_DOF, N_DOF)
    np.testing.assert_allclose(k, k.T, rtol=1e-12)


def test_schur_complement_matches_classical_stiffness_exactly_at_e0a_zero() -> None:
    """Eliminating e* (K_eff = K_ue @ K_ee^-1 @ K_eu) reproduces the classical T3 stiffness
    exactly - not just in a limit - because a linear T3's constant strain field projects onto
    the e* shape-function space with zero error, for any e0a."""
    element = _nonlocal_element(e0a=0.0)
    k = element.local_stiffness()
    k_ue = k[:N_DOF_U, N_DOF_U:]
    k_ee = -k[N_DOF_U:, N_DOF_U:]  # the assembled block is -K_ee; recover K_ee itself
    k_eff = k_ue @ np.linalg.solve(k_ee, k_ue.T)

    classical = _classical_element().local_stiffness()
    np.testing.assert_allclose(k_eff, classical, rtol=1e-9)


@pytest.mark.parametrize("e0a", [0.0, 0.1, 0.5, 2.0])
def test_constant_strain_field_recovers_exactly_regardless_of_e0a(e0a: float) -> None:
    """The 2-D analogue of the nonlocal bar's Peddieson-paradox null effect: a constant-strain
    displacement field's nonlocal strain e* equals the classical strain exactly, for ANY e0a,
    because the Helmholtz relation's particular solution for a spatially-constant source is
    that same constant (its own Laplacian vanishes)."""
    element = _nonlocal_element(e0a=e0a)
    a, b, g = 0.001, -0.0007, 0.0004
    u_local = np.zeros(N_DOF_U)
    for i, (x, y) in enumerate(_SCALENE_TRIANGLE):
        u_local[2 * i] = a * x + 0.5 * g * y
        u_local[2 * i + 1] = b * y + 0.5 * g * x

    k = element.local_stiffness()
    k_ue = k[:N_DOF_U, N_DOF_U:]
    k_ee = -k[N_DOF_U:, N_DOF_U:]
    e_star_nodal = np.linalg.solve(k_ee, k_ue.T @ u_local)

    full_local = np.concatenate([u_local, e_star_nodal])
    e_star_qp, _, _ = element.quadrature_point_response(full_local)
    expected = np.array([a, b, g])
    np.testing.assert_allclose(e_star_qp, np.broadcast_to(expected, e_star_qp.shape), atol=1e-9)


def test_quadrature_point_response_stress_matches_d_matrix() -> None:
    element = _nonlocal_element(e0a=0.2)
    e_star_dofs = np.arange(N_DOF_E, dtype=np.float64) * 1e-5
    full_local = np.concatenate([np.zeros(N_DOF_U), e_star_dofs])
    strain, stress, point_measure = element.quadrature_point_response(full_local)
    assert strain.shape == stress.shape
    assert point_measure.shape[0] == strain.shape[0]
    # sigma* = D @ e* must hold pointwise, independent of how e* was produced.
    e, nu = YOUNG_MODULUS, POISSON
    c = e / (1.0 - nu**2)
    d = np.array([[c, c * nu, 0.0], [c * nu, c, 0.0], [0.0, 0.0, c * (1.0 - nu) / 2.0]])
    expected_stress = strain @ d.T
    np.testing.assert_allclose(stress, expected_stress, rtol=1e-9)


def test_quadrature_point_response_rejects_wrong_shape() -> None:
    element = _nonlocal_element(e0a=0.1)
    with pytest.raises(InputValidationError):
        element.quadrature_point_response(np.zeros(3))


def test_dof_signature_lists_both_field_blocks() -> None:
    element = _nonlocal_element(e0a=0.1)
    signature = element.dof_signature()
    assert signature.dof_names_per_node == (
        ("u.x", "u.y"),
        ("u.x", "u.y"),
        ("u.x", "u.y"),
        ("e_star.xx", "e_star.yy", "e_star.xy"),
        ("e_star.xx", "e_star.yy", "e_star.xy"),
        ("e_star.xx", "e_star.yy", "e_star.xy"),
    )


def test_measure_matches_triangle_area() -> None:
    element = _nonlocal_element(e0a=0.1)
    x1, y1 = _SCALENE_TRIANGLE[0]
    x2, y2 = _SCALENE_TRIANGLE[1]
    x3, y3 = _SCALENE_TRIANGLE[2]
    expected_area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    assert element.measure() == pytest.approx(expected_area, rel=1e-12)
