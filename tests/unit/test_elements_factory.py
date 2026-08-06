"""``elements/factory.py``: unified dispatch across all four Stage-3 element families.

Before this phase, ``build_elements`` dispatched ``Bar`` only (the walking
skeleton, v0.8.0); ``EulerBernoulliBeam``/``TimoshenkoBeam``/``ContinuumElement``
(T3/Q4) were all built by hand in their own verification test files, never
through ``Model``. This file checks each family's dispatch branch builds an
element equal (by local stiffness) to one built directly - a plumbing proof,
not new element-mathematics verification (each family's own physics is
already verified in its own test file).
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.core.model import DomainDefinition, Model
from nanofem.elements.continuum.continuum import ContinuumElement
from nanofem.elements.factory import build_elements
from nanofem.elements.structural.bar import Bar
from nanofem.elements.structural.beam_eb import EulerBernoulliBeam
from nanofem.elements.structural.beam_timoshenko import TimoshenkoBeam
from nanofem.geometry.plane import PlaneGeometry
from nanofem.geometry.standard import CircularSection
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.numerics.reference.enums import CellType
from nanofem.physics.elasticity.euler_bernoulli import EulerBernoulliBendingTheory
from nanofem.physics.elasticity.isotropic import IsotropicElasticity
from nanofem.physics.elasticity.plane import PlaneStressConstitutive
from nanofem.physics.elasticity.timoshenko import TimoshenkoBeamTheory
from nanofem.utils.exceptions import MissingSectionError, ModelError

YOUNG_MODULUS = 200e9
SHEAR_MODULUS = 80e9
POISSON = 0.25
RADIUS = 0.02
LENGTH = 1.5


def _line_mesh() -> Mesh:
    coords = np.array([[0.0], [LENGTH]])
    block = CellBlock("line2", np.array([[0, 1]]), region="member")
    return Mesh(coords, (block,), (Region("all", 0, (0, 1)),))


def test_euler_bernoulli_dispatch_matches_direct_construction() -> None:
    mesh = _line_mesh()
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS))
    model.add_section("circ", CircularSection(radius=RADIUS))
    model.add_theory("bending", EulerBernoulliBendingTheory())
    model.add_domain(DomainDefinition("beam", "member", "bending", "steel", geometry="circ"))
    dof_handler = model.build_dof_handler()

    elements = build_elements(model, dof_handler)
    assert len(elements) == 1
    assert isinstance(elements[0], EulerBernoulliBeam)

    section = CircularSection(radius=RADIUS)
    coords = np.array([[0.0], [LENGTH]])
    global_dofs = (
        dof_handler.global_dof(0, "u", "y"),
        dof_handler.global_dof(0, "r", "z"),
        dof_handler.global_dof(1, "u", "y"),
        dof_handler.global_dof(1, "r", "z"),
    )
    expected = EulerBernoulliBeam(
        0, (0, 1), coords, global_dofs, YOUNG_MODULUS, section.second_moment_z()
    )
    np.testing.assert_allclose(elements[0].local_stiffness(), expected.local_stiffness())


def test_timoshenko_dispatch_matches_direct_construction() -> None:
    mesh = _line_mesh()
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, G=SHEAR_MODULUS, nu=POISSON))
    model.add_section("circ", CircularSection(radius=RADIUS))
    model.add_theory("bending_shear", TimoshenkoBeamTheory())
    model.add_domain(DomainDefinition("beam", "member", "bending_shear", "steel", geometry="circ"))
    dof_handler = model.build_dof_handler()

    elements = build_elements(model, dof_handler)
    assert len(elements) == 1
    assert isinstance(elements[0], TimoshenkoBeam)

    section = CircularSection(radius=RADIUS)
    coords = np.array([[0.0], [LENGTH]])
    global_dofs = (
        dof_handler.global_dof(0, "u", "y"),
        dof_handler.global_dof(0, "r", "z"),
        dof_handler.global_dof(1, "u", "y"),
        dof_handler.global_dof(1, "r", "z"),
    )
    shear_area = section.area() * section.shear_correction(POISSON)
    expected = TimoshenkoBeam(
        0,
        (0, 1),
        coords,
        global_dofs,
        YOUNG_MODULUS,
        SHEAR_MODULUS,
        section.second_moment_z(),
        shear_area,
    )
    np.testing.assert_allclose(elements[0].local_stiffness(), expected.local_stiffness())


def test_bar_dispatch_selects_bar_not_continuum() -> None:
    mesh = _line_mesh()
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS))
    model.add_section("circ", CircularSection(radius=RADIUS))
    model.add_theory("axial", IsotropicElasticity(dim=1))
    model.add_domain(DomainDefinition("bar", "member", "axial", "steel", geometry="circ"))
    dof_handler = model.build_dof_handler()
    elements = build_elements(model, dof_handler)
    assert len(elements) == 1
    assert isinstance(elements[0], Bar)


def _quad_mesh() -> Mesh:
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    block = CellBlock("quad4", np.array([[0, 1, 2, 3]]), region="plate")
    return Mesh(coords, (block,), (Region("all", 0, (0, 1, 2, 3)),))


def test_q4_dispatch_matches_direct_construction() -> None:
    mesh = _quad_mesh()
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON))
    model.add_section("plane", PlaneGeometry(thickness=0.01))
    model.add_theory("plane_stress_kinematics", IsotropicElasticity(dim=2))
    model.add_constitutive("plane_stress_law", PlaneStressConstitutive())
    model.add_domain(
        DomainDefinition(
            "plate",
            "plate",
            "plane_stress_kinematics",
            "steel",
            geometry="plane",
            constitutive="plane_stress_law",
        )
    )
    dof_handler = model.build_dof_handler()
    elements = build_elements(model, dof_handler)
    assert len(elements) == 1
    assert isinstance(elements[0], ContinuumElement)

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    global_dofs = np.array(
        [dof_handler.global_dof(n, "u", c) for n in (0, 1, 2, 3) for c in ("x", "y")],
        dtype=np.int64,
    )
    expected = ContinuumElement(
        cell_id=0,
        node_ids=(0, 1, 2, 3),
        coordinates=coords,
        global_dofs=global_dofs,
        cell_type=CellType.QUADRILATERAL,
        interpolation_order=1,
        theory=IsotropicElasticity(dim=2),
        constitutive=PlaneStressConstitutive(),
        material=Material("steel", E=YOUNG_MODULUS, nu=POISSON),
        section_measure=0.01,
    )
    np.testing.assert_allclose(elements[0].local_stiffness(), expected.local_stiffness())


def test_continuum_domain_without_constitutive_raises() -> None:
    mesh = _quad_mesh()
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON))
    model.add_section("plane", PlaneGeometry(thickness=0.01))
    model.add_theory("plane_stress_kinematics", IsotropicElasticity(dim=2))
    model.add_domain(
        DomainDefinition("plate", "plate", "plane_stress_kinematics", "steel", geometry="plane")
    )
    dof_handler = model.build_dof_handler()
    with pytest.raises(ModelError, match="constitutive law"):
        build_elements(model, dof_handler)


def test_continuum_domain_without_geometry_raises() -> None:
    mesh = _quad_mesh()
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON))
    model.add_theory("plane_stress_kinematics", IsotropicElasticity(dim=2))
    model.add_constitutive("plane_stress_law", PlaneStressConstitutive())
    model.add_domain(
        DomainDefinition(
            "plate",
            "plate",
            "plane_stress_kinematics",
            "steel",
            constitutive="plane_stress_law",
        )
    )
    dof_handler = model.build_dof_handler()
    with pytest.raises(MissingSectionError):
        build_elements(model, dof_handler)


def test_bar_domain_on_non_line2_cell_raises() -> None:
    mesh = _quad_mesh()
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS))
    model.add_section("circ", CircularSection(radius=RADIUS))
    model.add_theory("axial", IsotropicElasticity(dim=1))
    model.add_domain(DomainDefinition("bar", "plate", "axial", "steel", geometry="circ"))
    dof_handler = model.build_dof_handler()
    with pytest.raises(ModelError, match="line2"):
        build_elements(model, dof_handler)
