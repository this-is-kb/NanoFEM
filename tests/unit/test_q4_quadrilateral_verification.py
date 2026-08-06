"""Q4 (4-node quadrilateral) verification: the isoparametric-mapping fallback path.

Q4's bilinear map is affine only when the quadrilateral is a parallelogram
(``AffineMapping``'s own fit-residual check proves this - see
``numerics/mapping/affine.py``); a general quadrilateral needs
``IsoparametricMapping``. ``ContinuumElement`` (v0.9.0) now falls back to
``IsoparametricMapping`` - built from the same tabulated shape-function
basis the field already uses - whenever ``AffineMapping`` itself raises
``NonAffineError`` (v0.13.0), so ``ContinuumElement(cell_type=QUADRILATERAL,
interpolation_order=1, ...)`` *is* Q4, with no new element class, mirroring
T3's precedent (``docs/design/PLANE_ELASTICITY.md``).

This file verifies the same two properties T3's file does - the classical
constant-strain patch test and the rigid-body translation null space - for
both a parallelogram (still the ``AffineMapping`` path, a regression check)
and a genuinely non-affine quadrilateral (the new ``IsoparametricMapping``
path), confirming both patch-test energy and rigid-body residuals were
already numerically verified before this file was written (see
``docs/design/Q4_QUADRILATERAL.md``).
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.elements.continuum.continuum import ContinuumElement
from nanofem.geometry.plane import PlaneGeometry
from nanofem.materials.material import Material
from nanofem.numerics.mapping import AffineMapping, NonAffineError
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


def _quad_area(coordinates: np.ndarray) -> float:
    """The shoelace formula, valid for any simple (non-self-intersecting) polygon."""
    x, y = coordinates[:, 0], coordinates[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _q4_element(
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
        node_ids=(0, 1, 2, 3),
        coordinates=coordinates,
        global_dofs=np.arange(8, dtype=np.int64),
        cell_type=CellType.QUADRILATERAL,
        interpolation_order=1,
        theory=theory,
        constitutive=constitutive_cls(),
        material=material,
        section_measure=PlaneGeometry(thickness=thickness).thickness,
    )


#: A unit square - a parallelogram, so still the ``AffineMapping`` path.
_PARALLELOGRAM_QUAD = np.array([[0.0, 0.0], [2.0, 0.0], [2.5, 1.5], [0.5, 1.5]])

#: A genuinely non-parallelogram quadrilateral - forces the ``IsoparametricMapping`` fallback.
_NON_AFFINE_QUAD = np.array([[0.0, 0.0], [4.0, 0.5], [3.2, 3.0], [-0.3, 2.4]])


def test_non_affine_quad_actually_needs_the_fallback() -> None:
    """Pin the premise: ``AffineMapping`` itself must reject this quad's vertices."""
    with pytest.raises(NonAffineError):
        AffineMapping(CellType.QUADRILATERAL, _NON_AFFINE_QUAD)


@pytest.mark.parametrize(
    "coordinates",
    [_PARALLELOGRAM_QUAD, _NON_AFFINE_QUAD],
    ids=["parallelogram_affine_path", "non_affine_isoparametric_path"],
)
@pytest.mark.parametrize(
    "constitutive_cls", [PlaneStressConstitutive, PlaneStrainConstitutive], ids=["stress", "strain"]
)
def test_constant_strain_patch_test(constitutive_cls: type, coordinates: np.ndarray) -> None:
    """A prescribed linear (constant-strain) displacement field: FEM energy matches analytical.

    Passing for the non-affine quad is the isoparametric consistency property
    (partition of unity + linear completeness of the bilinear basis), not a
    coincidence specific to the parallelogram case.
    """
    e, nu, thickness = 200e9, 0.3, 0.01
    element = _q4_element(coordinates, constitutive_cls, e=e, nu=nu, thickness=thickness)
    stiffness = element.local_stiffness()

    a, b, g = 0.001, -0.0005, 0.0007  # eps_xx, eps_yy, gamma_xy targets
    displacement = np.zeros(8)
    for i, (x, y) in enumerate(coordinates):
        displacement[2 * i] = a * x + 0.5 * g * y
        displacement[2 * i + 1] = b * y + 0.5 * g * x

    fem_energy = 0.5 * displacement @ stiffness @ displacement
    strain = np.array([a, b, g])
    constitutive_matrix = _plane_matrix(constitutive_cls, e, nu)
    expected_energy = (
        0.5 * (strain @ constitutive_matrix @ strain) * _quad_area(coordinates) * thickness
    )
    assert fem_energy == pytest.approx(expected_energy, rel=1e-9)


@pytest.mark.parametrize(
    "coordinates",
    [_PARALLELOGRAM_QUAD, _NON_AFFINE_QUAD],
    ids=["parallelogram_affine_path", "non_affine_isoparametric_path"],
)
@pytest.mark.parametrize(
    "constitutive_cls", [PlaneStressConstitutive, PlaneStrainConstitutive], ids=["stress", "strain"]
)
def test_stiffness_is_symmetric_and_rigid_body_null(
    constitutive_cls: type, coordinates: np.ndarray
) -> None:
    element = _q4_element(coordinates, constitutive_cls)
    stiffness = element.local_stiffness()
    np.testing.assert_allclose(stiffness, stiffness.T)

    scale = np.abs(stiffness).max()
    rigid_x = np.tile([1.0, 0.0], 4)
    rigid_y = np.tile([0.0, 1.0], 4)
    np.testing.assert_allclose(stiffness @ rigid_x, 0.0, atol=1e-9 * scale)
    np.testing.assert_allclose(stiffness @ rigid_y, 0.0, atol=1e-9 * scale)


def test_dof_signature_matches_two_component_displacement_field() -> None:
    element = _q4_element(_PARALLELOGRAM_QUAD, PlaneStressConstitutive)
    signature = element.dof_signature()
    assert signature.dof_names_per_node == (("u.x", "u.y"),) * 4
