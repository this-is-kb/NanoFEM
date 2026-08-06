"""LinearStaticAnalysis.run() for a T3 plane-stress domain, end to end.

The Mesh -> Model -> LinearStaticAnalysis chain for a 2-D continuum domain
was never exercised before this phase: T3/Q4 verification (v0.12.0/v0.13.0)
constructed ``ContinuumElement`` directly, bypassing ``Model``/``build_
elements``/``LinearStaticAnalysis`` entirely, because ``Model`` had no way to
register a constitutive law or a ``PlaneGeometry`` and ``elements/factory.py``
dispatched ``Bar`` only. This file is the first proof the full pipeline works
for a 2-D domain, mirroring ``test_static_analytical.py``'s bar test.

A two-triangle rectangular plate (0,0)-(L,H) under a uniaxial tension load
applied as the *consistent* nodal load for a uniform traction (``P/2`` at
each right-edge node - exact for a linear edge) reproduces the classical
1-D bar formula ``u_x(L) = P L / (E H t)`` exactly, since ``H*t`` is exactly
the cross-sectional area a uniaxial bar under the same total force would
have - a global patch test, not just the single-element one already proved.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.analysis.static import LinearStaticAnalysis
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import NodalLoad
from nanofem.core.model import DomainDefinition, Model
from nanofem.elements.continuum.continuum import ContinuumElement
from nanofem.elements.factory import build_elements
from nanofem.geometry.plane import PlaneGeometry
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.numerics.reference.enums import CellType
from nanofem.physics.elasticity.isotropic import IsotropicElasticity
from nanofem.physics.elasticity.plane import PlaneStressConstitutive

YOUNG_MODULUS = 200e9
POISSON = 0.3
LENGTH = 2.0
HEIGHT = 1.0
THICKNESS = 0.01
TOTAL_FORCE = 10_000.0


def _plate_model() -> Model:
    """Two triangles covering a rectangle; left edge fixed in x, corner pinned in y."""
    coords = np.array([[0.0, 0.0], [LENGTH, 0.0], [LENGTH, HEIGHT], [0.0, HEIGHT]])
    block = CellBlock("tri3", np.array([[0, 1, 2], [0, 2, 3]]), region="plate")
    regions = (
        Region("left_edge", 0, (0, 3)),
        Region("origin", 0, (0,)),
        Region("right_edge", 0, (1, 2)),
    )
    mesh = Mesh(coords, (block,), regions)
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON))
    model.add_section("plane", PlaneGeometry(thickness=THICKNESS))
    model.add_theory("plane_stress_kinematics", IsotropicElasticity(dim=2))
    model.add_constitutive("plane_stress_law", PlaneStressConstitutive())
    model.add_domain(
        DomainDefinition(
            "plate_domain",
            "plate",
            "plane_stress_kinematics",
            "steel",
            geometry="plane",
            constitutive="plane_stress_law",
        )
    )
    model.add_dirichlet(DirichletBC("left_edge", "u", ("x",), 0.0))
    model.add_dirichlet(DirichletBC("origin", "u", ("y",), 0.0))
    case = LoadCase("tension")
    half_force = TOTAL_FORCE / 2.0
    case.add(NodalLoad("right_edge", "u", np.array([half_force, 0.0])))
    model.add_load_case(case)
    return model


def test_tip_displacement_matches_1d_bar_formula() -> None:
    """A uniaxial-consistent-loaded T3 mesh reproduces P*L/(E*H*t) exactly."""
    model = _plate_model()
    result = LinearStaticAnalysis(model).run()["tension"]
    dof_handler = result.dof_handler
    expected = TOTAL_FORCE * LENGTH / (YOUNG_MODULUS * HEIGHT * THICKNESS)
    for node in (1, 2):
        u_x = result.displacements[dof_handler.global_dof(node, "u", "x")]
        assert u_x == pytest.approx(expected, rel=1e-9)


def test_left_edge_reactions_balance_the_applied_force() -> None:
    model = _plate_model()
    result = LinearStaticAnalysis(model).run()["tension"]
    # reactions are ordered by constrained-DOF index (sorted, SDS C-5); both left_edge nodes'
    # u_x reactions plus origin's u_y reaction are the three constrained DOFs here.
    assert result.reactions.sum() == pytest.approx(-TOTAL_FORCE, rel=1e-9)


def test_build_elements_matches_a_directly_constructed_continuum_element() -> None:
    """Plumbing check: factory-built T3 elements equal hand-built ``ContinuumElement``s."""
    model = _plate_model()
    dof_handler = model.build_dof_handler()
    elements = build_elements(model, dof_handler)
    assert len(elements) == 2
    assert isinstance(elements[0], ContinuumElement)

    theory = IsotropicElasticity(dim=2)
    material = Material("steel", E=YOUNG_MODULUS, nu=POISSON)
    coords = np.array([[0.0, 0.0], [LENGTH, 0.0], [LENGTH, HEIGHT]])
    global_dofs = np.array(
        [dof_handler.global_dof(n, "u", c) for n in (0, 1, 2) for c in ("x", "y")],
        dtype=np.int64,
    )
    expected = ContinuumElement(
        cell_id=0,
        node_ids=(0, 1, 2),
        coordinates=coords,
        global_dofs=global_dofs,
        cell_type=CellType.TRIANGLE,
        interpolation_order=1,
        theory=theory,
        constitutive=PlaneStressConstitutive(),
        material=material,
        section_measure=THICKNESS,
    )
    np.testing.assert_allclose(elements[0].local_stiffness(), expected.local_stiffness())
