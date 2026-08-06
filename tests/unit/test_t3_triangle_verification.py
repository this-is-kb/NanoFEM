"""T3 (3-node triangle) verification: a constant-strain patch test through ContinuumElement.

T3 is not a new element class - a 3-node triangle with linear interpolation
is always geometrically affine (3 vertices exactly determine an affine map,
a fact about triangles, not a simplification), so the existing generic
``ContinuumElement`` (built v0.9.0) already composes it correctly with no
new code: ``ContinuumElement(cell_type=TRIANGLE, interpolation_order=1,
...)`` *is* T3. This file verifies that composition against the classical
constant-strain patch test - the standard first check for any new continuum
element - for both plane stress and plane strain.

Stress/strain recovery does not exist yet (a later phase's job), so the
patch test is checked via strain energy: for a constant-strain
displacement field, ``0.5 u^T K u`` must equal the analytical
``0.5 (eps^T D eps) * Area * thickness`` exactly (to floating-point
precision) - this was verified numerically before any production code was
written (see ``docs/design/PLANE_ELASTICITY.md``).

Rigid-body null-space residuals here are compared against a tolerance
scaled to ``K``'s own magnitude, not a small fixed ``atol`` (unlike the
beam elements' tests): ``E ~ 2e11`` Pa gives ``K`` entries of order
``1e11``, so floating-point noise on a supposedly-zero product is itself of
order ``1e11 * 1e-16 ~ 1e-5`` - a fixed ``atol=1e-6`` would be tighter than
double precision allows for a stiffness matrix at this physical scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.elements.continuum.continuum import ContinuumElement
from nanofem.geometry.plane import PlaneGeometry
from nanofem.materials.material import Material
from nanofem.numerics.reference.enums import CellType
from nanofem.physics.elasticity.isotropic import IsotropicElasticity
from nanofem.physics.elasticity.plane import PlaneStrainConstitutive, PlaneStressConstitutive


def _plane_matrix(constitutive_cls: type, e: float, nu: float) -> np.ndarray:
    if constitutive_cls is PlaneStressConstitutive:
        c = e / (1.0 - nu**2)
        return np.array([[c, c * nu, 0.0], [c * nu, c, 0.0], [0.0, 0.0, c * (1.0 - nu) / 2.0]])
    c = e / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return np.array(
        [
            [c * (1.0 - nu), c * nu, 0.0],
            [c * nu, c * (1.0 - nu), 0.0],
            [0.0, 0.0, c * (1.0 - 2.0 * nu) / 2.0],
        ]
    )


def _triangle_area(coordinates: np.ndarray) -> float:
    (x1, y1), (x2, y2), (x3, y3) = coordinates
    return float(0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)))


def _t3_element(
    coordinates: np.ndarray,
    constitutive_cls: type,
    e: float = 200e9,
    nu: float = 0.3,
    thickness: float = 1.0,
) -> ContinuumElement:
    theory = IsotropicElasticity(dim=2)
    material = Material("m", E=e, nu=nu)
    return ContinuumElement(
        cell_id=0,
        node_ids=(0, 1, 2),
        coordinates=coordinates,
        global_dofs=np.arange(6, dtype=np.int64),
        cell_type=CellType.TRIANGLE,
        interpolation_order=1,
        theory=theory,
        constitutive=constitutive_cls(),
        material=material,
        section_measure=PlaneGeometry(thickness=thickness).thickness,
    )


_UNIT_RIGHT_TRIANGLE = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
_SCALENE_ROTATED_TRIANGLE = np.array([[0.3, -0.2], [2.1, 0.4], [0.9, 1.7]])


@pytest.mark.parametrize(
    "coordinates", [_UNIT_RIGHT_TRIANGLE, _SCALENE_ROTATED_TRIANGLE], ids=["unit_right", "scalene"]
)
@pytest.mark.parametrize(
    "constitutive_cls", [PlaneStressConstitutive, PlaneStrainConstitutive], ids=["stress", "strain"]
)
def test_constant_strain_patch_test(constitutive_cls: type, coordinates: np.ndarray) -> None:
    """A prescribed linear (constant-strain) displacement field: FEM energy matches analytical."""
    e, nu, thickness = 200e9, 0.3, 0.01
    element = _t3_element(coordinates, constitutive_cls, e=e, nu=nu, thickness=thickness)
    stiffness = element.local_stiffness()

    a, b, g = 0.001, -0.0005, 0.0007  # eps_xx, eps_yy, gamma_xy targets
    # u_x = a*x + (g/2)*y; u_y = b*y + (g/2)*x -> eps_xx=a, eps_yy=b, gamma_xy=g
    displacement = np.zeros(6)
    for i, (x, y) in enumerate(coordinates):
        displacement[2 * i] = a * x + 0.5 * g * y
        displacement[2 * i + 1] = b * y + 0.5 * g * x

    fem_energy = 0.5 * displacement @ stiffness @ displacement
    strain = np.array([a, b, g])
    constitutive_matrix = _plane_matrix(constitutive_cls, e, nu)
    expected_energy = (
        0.5 * (strain @ constitutive_matrix @ strain) * _triangle_area(coordinates) * thickness
    )
    assert fem_energy == pytest.approx(expected_energy, rel=1e-9)


@pytest.mark.parametrize(
    "constitutive_cls", [PlaneStressConstitutive, PlaneStrainConstitutive], ids=["stress", "strain"]
)
def test_stiffness_is_symmetric_and_rigid_body_null(constitutive_cls: type) -> None:
    element = _t3_element(_SCALENE_ROTATED_TRIANGLE, constitutive_cls)
    stiffness = element.local_stiffness()
    np.testing.assert_allclose(stiffness, stiffness.T)

    scale = np.abs(stiffness).max()
    rigid_x = np.tile([1.0, 0.0], 3)
    rigid_y = np.tile([0.0, 1.0], 3)
    np.testing.assert_allclose(stiffness @ rigid_x, 0.0, atol=1e-9 * scale)
    np.testing.assert_allclose(stiffness @ rigid_y, 0.0, atol=1e-9 * scale)


def test_dof_signature_matches_two_component_displacement_field() -> None:
    element = _t3_element(_UNIT_RIGHT_TRIANGLE, PlaneStressConstitutive)
    signature = element.dof_signature()
    assert signature.dof_names_per_node == (("u.x", "u.y"),) * 3
