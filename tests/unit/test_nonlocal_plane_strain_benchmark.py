"""Eringen differential nonlocal elasticity: the 2-D plane-strain benchmark (Stage 4 Step 4).

Every 2-D Eringen benchmark shipped so far (``test_nonlocal_plate_with_hole_benchmark.py``,
``test_nonlocal_cantilever_benchmark.py``) uses ``PlaneStressConstitutive``.
``EringenDifferentialMaterial`` already wraps ``PlaneStrainConstitutive`` just as generically
(``test_eringen_differential_theory.py`` parametrizes its unit-level delegation checks over both
laws), but no full ``Model -> LinearStaticAnalysis`` solve had ever exercised plane strain - this
file is the Stage-4 directive's explicit "two-dimensional plane strain" benchmark item, built by
mirroring the existing plane-stress cantilever exactly (same geometry, mesh, loads), so the only
thing that changes is the constitutive law.

Two checks:

1. **A single-element, constant-strain patch test.** Exact match between the mixed system and a
   directly-computed classical plane-strain FEM solution, for *any* e0a - the same 2-D Peddieson-
   paradox null effect already established for plane stress in
   ``examples/ex10_classical_to_eringen_theory_swap.py``, now confirmed independently for the
   plane-strain constitutive law.
2. **A cantilever benchmark with genuine strain variation.** Mesh-convergent recovery of the
   classical plane-strain FEM solution at e0a=0 (mirroring
   ``test_nonlocal_cantilever_benchmark.py``'s already-established mesh-convergent local-limit
   property, Section 7 of ``docs/design/ERINGEN_DIFFERENTIAL_CONTINUUM.md``), monotonic nonlocal
   softening with e0a, and a physical cross-check against the plane-stress cantilever: for the
   same (E, nu) and load, the plane-strain solution must be stiffer (smaller-magnitude
   deflection) than the plane-stress solution - the well-known effect of the plane-strain
   constraint (``D`` scaled by ``1/(1-nu^2)`` rather than plane stress's unconstrained ``D``).
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.analysis.static import LinearStaticAnalysis
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import NodalLoad
from nanofem.core.model import DomainDefinition, Model
from nanofem.geometry.plane import PlaneGeometry
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.physics.elasticity.eringen_differential import (
    EringenDifferentialMaterial,
    EringenDifferentialTheory,
)
from nanofem.physics.elasticity.isotropic import IsotropicElasticity
from nanofem.physics.elasticity.plane import PlaneStrainConstitutive, PlaneStressConstitutive

YOUNG_MODULUS = 200e9
POISSON = 0.3
THICKNESS = 0.05
LENGTH = 2.0
HEIGHT = 0.4
TIP_FORCE = 1_000.0


# ---------------------------------------------------------------------------
# Part 1: single-element constant-strain patch test
# ---------------------------------------------------------------------------


def _single_triangle_mesh() -> Mesh:
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    block = CellBlock("tri3", np.array([[0, 1, 2]]), region="patch")
    regions = (Region("clamp", 0, (0, 2)), Region("tip", 0, (1,)))
    return Mesh(coords, (block,), regions)


def _solve_classical_plane_strain(mesh: Mesh, force: float) -> float:
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON))
    model.add_section("plane", PlaneGeometry(thickness=THICKNESS))
    model.add_theory("elastic", IsotropicElasticity(dim=2))
    model.add_constitutive("law", PlaneStrainConstitutive())
    model.add_domain(
        DomainDefinition("dom", "patch", "elastic", "steel", geometry="plane", constitutive="law")
    )
    model.add_dirichlet(DirichletBC("clamp", "u", ("x", "y"), 0.0))
    case = LoadCase("load")
    case.add(NodalLoad("tip", "u", np.array([0.0, -force])))
    model.add_load_case(case)

    result = LinearStaticAnalysis(model).run()["load"]
    dof_handler = result.dof_handler
    tip_node = model.mesh.nodes_in_region("tip")[0]
    return float(result.displacements[dof_handler.global_dof(tip_node, "u", "y")])


def _solve_eringen_plane_strain(mesh: Mesh, force: float, e0a: float) -> float:
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON, e0a=e0a))
    model.add_section("plane", PlaneGeometry(thickness=THICKNESS))
    model.add_theory("nonlocal", EringenDifferentialTheory(dim=2))
    model.add_constitutive("nonlocal_law", EringenDifferentialMaterial(PlaneStrainConstitutive()))
    model.add_domain(
        DomainDefinition(
            "dom", "patch", "nonlocal", "steel", geometry="plane", constitutive="nonlocal_law"
        )
    )
    model.add_dirichlet(DirichletBC("clamp", "u", ("x", "y"), 0.0))
    case = LoadCase("load")
    case.add(NodalLoad("tip", "u", np.array([0.0, -force])))
    model.add_load_case(case)

    result = LinearStaticAnalysis(model).run()["load"]
    dof_handler = result.dof_handler
    tip_node = model.mesh.nodes_in_region("tip")[0]
    return float(result.displacements[dof_handler.global_dof(tip_node, "u", "y")])


@pytest.mark.parametrize("e0a", [0.0, 0.1, 0.3, 1.0])
def test_plane_strain_constant_strain_patch_matches_classical_for_any_e0a(e0a: float) -> None:
    """The 2-D Peddieson-paradox null effect, independently confirmed for plane strain: a single
    constant-strain element's Helmholtz correction vanishes identically, so the mixed system
    matches the classical plane-strain solution regardless of the characteristic length."""
    mesh = _single_triangle_mesh()
    u_classical = _solve_classical_plane_strain(mesh, TIP_FORCE)
    u_eringen = _solve_eringen_plane_strain(mesh, TIP_FORCE, e0a)
    assert np.isclose(u_eringen, u_classical, rtol=1e-9)


# ---------------------------------------------------------------------------
# Part 2: cantilever benchmark - genuine strain variation
# ---------------------------------------------------------------------------


def _cantilever_mesh(n_x: int, n_y: int) -> tuple[Mesh, tuple[int, ...]]:
    xs = np.linspace(0.0, LENGTH, n_x + 1)
    ys = np.linspace(-HEIGHT / 2.0, HEIGHT / 2.0, n_y + 1)
    coords = np.array([[x, y] for y in ys for x in xs])

    def node_id(i: int, j: int) -> int:
        return j * (n_x + 1) + i

    triangles = []
    for j in range(n_y):
        for i in range(n_x):
            a, b, c, d = node_id(i, j), node_id(i + 1, j), node_id(i + 1, j + 1), node_id(i, j + 1)
            triangles.append([a, b, c])
            triangles.append([a, c, d])
    block = CellBlock("tri3", np.array(triangles), region="plate")

    left_nodes = tuple(node_id(0, j) for j in range(n_y + 1))
    right_nodes = tuple(node_id(n_x, j) for j in range(n_y + 1))
    regions = (Region("left", 0, left_nodes), Region("right", 0, right_nodes))
    mesh = Mesh(coords, (block,), regions)
    return mesh, right_nodes


def _cantilever_tip_deflection_classical(n_x: int, n_y: int, constitutive_cls: type) -> float:
    mesh, right_nodes = _cantilever_mesh(n_x, n_y)
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON))
    model.add_section("plane", PlaneGeometry(thickness=THICKNESS))
    model.add_theory("elastic", IsotropicElasticity(dim=2))
    model.add_constitutive("law", constitutive_cls())
    model.add_domain(
        DomainDefinition("dom", "plate", "elastic", "steel", geometry="plane", constitutive="law")
    )
    model.add_dirichlet(DirichletBC("left", "u", ("x", "y"), 0.0))
    per_node_force = -TIP_FORCE / len(right_nodes)
    case = LoadCase("tip")
    case.add(NodalLoad("right", "u", np.array([0.0, per_node_force])))
    model.add_load_case(case)

    result = LinearStaticAnalysis(model).run()["tip"]
    dof_handler = result.dof_handler
    return float(
        np.mean([result.displacements[dof_handler.global_dof(n, "u", "y")] for n in right_nodes])
    )


def _cantilever_tip_deflection_eringen_plane_strain(n_x: int, n_y: int, e0a: float) -> float:
    mesh, right_nodes = _cantilever_mesh(n_x, n_y)
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON, e0a=e0a))
    model.add_section("plane", PlaneGeometry(thickness=THICKNESS))
    model.add_theory("nonlocal", EringenDifferentialTheory(dim=2))
    model.add_constitutive("nonlocal_law", EringenDifferentialMaterial(PlaneStrainConstitutive()))
    model.add_domain(
        DomainDefinition(
            "dom", "plate", "nonlocal", "steel", geometry="plane", constitutive="nonlocal_law"
        )
    )
    model.add_dirichlet(DirichletBC("left", "u", ("x", "y"), 0.0))
    per_node_force = -TIP_FORCE / len(right_nodes)
    case = LoadCase("tip")
    case.add(NodalLoad("right", "u", np.array([0.0, per_node_force])))
    model.add_load_case(case)

    result = LinearStaticAnalysis(model).run()["tip"]
    dof_handler = result.dof_handler
    return float(
        np.mean([result.displacements[dof_handler.global_dof(n, "u", "y")] for n in right_nodes])
    )


def test_plane_strain_e0a_zero_converges_to_classical_plane_strain_fem_under_refinement() -> None:
    """Mesh-CONVERGENT (not exact-on-a-fixed-mesh) local-limit recovery, exactly the property
    established for plane stress - confirming it is a property of the mixed formulation itself,
    not an artifact specific to one constitutive law."""
    levels = [(6, 3), (10, 5), (16, 8)]
    rel_errors = []
    for n_x, n_y in levels:
        classical_tip = _cantilever_tip_deflection_classical(n_x, n_y, PlaneStrainConstitutive)
        mixed_tip = _cantilever_tip_deflection_eringen_plane_strain(n_x, n_y, e0a=0.0)
        rel_errors.append(abs(mixed_tip - classical_tip) / abs(classical_tip))

    for coarser, finer in zip(rel_errors, rel_errors[1:], strict=False):
        assert finer < coarser, f"e0a=0 discrepancy must shrink under refinement: {rel_errors}"
    assert rel_errors[-1] < 0.30, f"finest-mesh e0a=0 discrepancy {rel_errors[-1]:.2%} too large"


def test_plane_strain_tip_deflection_increases_monotonically_with_e0a() -> None:
    """Nonlocal softening under plane strain, same physical direction as plane stress."""
    n_x, n_y = 10, 5
    e0a_values = [0.0, 0.05, 0.1, 0.2]
    deflections = [
        abs(_cantilever_tip_deflection_eringen_plane_strain(n_x, n_y, e0a)) for e0a in e0a_values
    ]
    for smaller, larger in zip(deflections, deflections[1:], strict=False):
        assert larger > smaller, f"tip deflection must grow with e0a: {deflections}"


def test_plane_strain_cantilever_is_stiffer_than_plane_stress_at_the_local_limit() -> None:
    """A textbook cross-check: for identical (E, nu, geometry, load), the plane-strain
    constraint (out-of-plane strain suppressed) always produces a stiffer response than plane
    stress - so the plane-strain tip deflection magnitude must be strictly smaller."""
    n_x, n_y = 10, 5
    plane_stress_tip = _cantilever_tip_deflection_classical(n_x, n_y, PlaneStressConstitutive)
    plane_strain_tip = _cantilever_tip_deflection_classical(n_x, n_y, PlaneStrainConstitutive)
    assert abs(plane_strain_tip) < abs(plane_stress_tip)
