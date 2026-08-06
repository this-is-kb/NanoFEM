"""Tutorial: swapping continuum theories without touching the solver architecture (Stage 4).

This is the executable version of NanoFEM's central architectural promise, stated verbatim in
the Stage 4 directive's final acceptance criteria: a model can be solved with classical
elasticity, then switched to Eringen Differential Nonlocal Elasticity, "without modification of
Mesh, Reference Elements, Shape Functions, Quadrature, Assembly Architecture, Linear Solver,
Post-processing. Only the continuum theory shall change."

Two parts:

**Part A - a single-element patch test.** For a constant-strain field, the mixed (u, e*)
formulation is *exactly* equal to the classical solution on the same mesh, for *any* value of
``e0a`` - not just ``e0a = 0`` (proven at the element level in
``test_nonlocal_continuum_element.py``). A single T3 has a spatially uniform strain field, so
the Helmholtz correction (which acts through spatial *gradients* of the nonlocal strain)
vanishes identically - a 2-D analogue of the Peddieson paradox already established for the 1-D
nonlocal bar/beam, where concentrated or uniform loading also produces zero nonlocal effect.
This is the cleanest possible demonstration of the theory-swap story: build one triangle, solve
it with ``IsotropicElasticity`` + ``PlaneStressConstitutive``, then solve the *identical*
mesh/BCs/load with only the domain's theory and constitutive law swapped to
``EringenDifferentialTheory``/``EringenDifferentialMaterial`` - the two displacement fields
match to machine precision, regardless of ``e0a``.

**Part B - a general (non-constant-strain) field: the 2-D cantilever.** Here the local limit is
only *mesh-convergent*, not exact-on-a-fixed-mesh - a genuine and important property of this
mixed formulation, not a defect (see ``docs/design/ERINGEN_DIFFERENTIAL_CONTINUUM.md`` Section 7
and dev note N-86). The same swap is performed on a realistic cantilever mesh, the ``e0a = 0``
case is compared against the classical solution to show the expected residual discretization
gap, and then ``e0a`` is increased to demonstrate nonlocal softening (increasing compliance with
the characteristic length) - the same physical direction already established for the 1-D
nonlocal bar and beam.
"""

from __future__ import annotations

import numpy as np

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
from nanofem.physics.elasticity.plane import PlaneStressConstitutive

YOUNG_MODULUS = 200e9
POISSON = 0.3
THICKNESS = 0.05


def _solve_classical(mesh: Mesh, force: float) -> float:
    """Build and solve the *classical* elasticity model - untouched pipeline."""
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON))
    model.add_section("plane", PlaneGeometry(thickness=THICKNESS))
    model.add_theory("elastic", IsotropicElasticity(dim=2))
    model.add_constitutive("law", PlaneStressConstitutive())
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


def _solve_eringen(mesh: Mesh, force: float, e0a: float) -> float:
    """Build and solve the *same* mesh/BCs/load - only the theory and constitutive law changed."""
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON, e0a=e0a))
    model.add_section("plane", PlaneGeometry(thickness=THICKNESS))
    model.add_theory("nonlocal", EringenDifferentialTheory(dim=2))
    model.add_constitutive("nonlocal_law", EringenDifferentialMaterial(PlaneStressConstitutive()))
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


# ---------------------------------------------------------------------------
# Part A: a single triangle - constant strain, exact local-limit recovery
# ---------------------------------------------------------------------------


def _single_triangle_mesh() -> Mesh:
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    block = CellBlock("tri3", np.array([[0, 1, 2]]), region="patch")
    regions = (Region("clamp", 0, (0, 2)), Region("tip", 0, (1,)))
    return Mesh(coords, (block,), regions)


def part_a_single_element_patch_test() -> None:
    print("=== Part A: single triangle, constant strain - exact local-limit recovery ===")
    mesh = _single_triangle_mesh()
    force = 1_000.0

    u_classical = _solve_classical(mesh, force)
    u_eringen_local = _solve_eringen(mesh, force, e0a=0.0)
    print(f"  classical IsotropicElasticity:            u_tip_y = {u_classical:.12e} m")
    print(f"  EringenDifferentialTheory, e0a=0 (local):  u_tip_y = {u_eringen_local:.12e} m")

    relative_difference = abs(u_eringen_local - u_classical) / abs(u_classical)
    print(f"  relative difference: {relative_difference:.3e}")
    assert relative_difference < 1e-9, "constant-strain local limit must match to machine precision"
    print("  -> exact match: only Theory/ConstitutiveModel changed, everything else identical\n")

    u_eringen_nonlocal = _solve_eringen(mesh, force, e0a=0.3)
    print(f"  EringenDifferentialTheory, e0a=0.3:        u_tip_y = {u_eringen_nonlocal:.12e} m")
    assert np.isclose(u_eringen_nonlocal, u_classical, rtol=1e-9)
    print(
        "  -> identical for ANY e0a here, not just e0a=0: a single constant-strain element has\n"
        "     a spatially uniform strain field, so the Helmholtz correction (which acts through\n"
        "     spatial *gradients* of the nonlocal strain) vanishes identically - a 2-D analogue\n"
        "     of the Peddieson paradox already established for the 1-D nonlocal bar/beam, where\n"
        "     concentrated or uniform loading also produces zero nonlocal effect. Nonlocality\n"
        "     only bites once the field varies in space - Part B demonstrates that.\n"
    )


# ---------------------------------------------------------------------------
# Part B: the 2-D cantilever - a general, non-constant-strain field
# ---------------------------------------------------------------------------

CANTILEVER_LENGTH = 2.0
CANTILEVER_HEIGHT = 0.4
TIP_FORCE = 1_000.0


def _cantilever_mesh(n_x: int, n_y: int) -> Mesh:
    xs = np.linspace(0.0, CANTILEVER_LENGTH, n_x + 1)
    ys = np.linspace(-CANTILEVER_HEIGHT / 2.0, CANTILEVER_HEIGHT / 2.0, n_y + 1)
    coords = np.array([[x, y] for y in ys for x in xs])

    def node_id(i: int, j: int) -> int:
        return j * (n_x + 1) + i

    triangles = []
    for j in range(n_y):
        for i in range(n_x):
            a, b, c, d = node_id(i, j), node_id(i + 1, j), node_id(i + 1, j + 1), node_id(i, j + 1)
            triangles.append([a, b, c])
            triangles.append([a, c, d])
    block = CellBlock("tri3", np.array(triangles), region="patch")

    left_nodes = tuple(node_id(0, j) for j in range(n_y + 1))
    right_nodes = tuple(node_id(n_x, j) for j in range(n_y + 1))
    regions = (Region("clamp", 0, left_nodes), Region("tip", 0, right_nodes))
    return Mesh(coords, (block,), regions)


def _cantilever_tip_deflection_classical(mesh: Mesh) -> float:
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON))
    model.add_section("plane", PlaneGeometry(thickness=THICKNESS))
    model.add_theory("elastic", IsotropicElasticity(dim=2))
    model.add_constitutive("law", PlaneStressConstitutive())
    model.add_domain(
        DomainDefinition("dom", "patch", "elastic", "steel", geometry="plane", constitutive="law")
    )
    model.add_dirichlet(DirichletBC("clamp", "u", ("x", "y"), 0.0))
    tip_nodes = model.mesh.nodes_in_region("tip")
    per_node_force = -TIP_FORCE / len(tip_nodes)
    case = LoadCase("load")
    case.add(NodalLoad("tip", "u", np.array([0.0, per_node_force])))
    model.add_load_case(case)

    result = LinearStaticAnalysis(model).run()["load"]
    dof_handler = result.dof_handler
    return float(
        np.mean([result.displacements[dof_handler.global_dof(n, "u", "y")] for n in tip_nodes])
    )


def _cantilever_tip_deflection_eringen(mesh: Mesh, e0a: float) -> float:
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON, e0a=e0a))
    model.add_section("plane", PlaneGeometry(thickness=THICKNESS))
    model.add_theory("nonlocal", EringenDifferentialTheory(dim=2))
    model.add_constitutive("nonlocal_law", EringenDifferentialMaterial(PlaneStressConstitutive()))
    model.add_domain(
        DomainDefinition(
            "dom", "patch", "nonlocal", "steel", geometry="plane", constitutive="nonlocal_law"
        )
    )
    model.add_dirichlet(DirichletBC("clamp", "u", ("x", "y"), 0.0))
    tip_nodes = model.mesh.nodes_in_region("tip")
    per_node_force = -TIP_FORCE / len(tip_nodes)
    case = LoadCase("load")
    case.add(NodalLoad("tip", "u", np.array([0.0, per_node_force])))
    model.add_load_case(case)

    result = LinearStaticAnalysis(model).run()["load"]
    dof_handler = result.dof_handler
    return float(
        np.mean([result.displacements[dof_handler.global_dof(n, "u", "y")] for n in tip_nodes])
    )


def part_b_cantilever_softening_study() -> None:
    print("=== Part B: 2-D cantilever, a general (non-constant-strain) field ===")
    mesh = _cantilever_mesh(10, 5)

    u_classical = _cantilever_tip_deflection_classical(mesh)
    u_eringen_local = _cantilever_tip_deflection_eringen(mesh, e0a=0.0)
    relative_difference = abs(u_eringen_local - u_classical) / abs(u_classical)
    print(f"  classical tip deflection:              {u_classical:.6e} m")
    print(f"  Eringen (e0a=0) tip deflection:         {u_eringen_local:.6e} m")
    print(f"  relative difference: {relative_difference:.2%}")
    print(
        "  -> nonzero on a fixed mesh, by design: e0a=0 recovers the classical solution only\n"
        "     in the mesh-refinement limit for a non-constant-strain field, since the coupled\n"
        "     field e* is C0-continuous while the T3's own classical strain is not (see\n"
        "     docs/design/ERINGEN_DIFFERENTIAL_CONTINUUM.md Section 7, dev note N-86)\n"
    )

    print("  nonlocal softening as e0a increases (same mesh, same load):")
    for e0a in (0.0, 0.05, 0.1, 0.2):
        deflection = _cantilever_tip_deflection_eringen(mesh, e0a)
        print(f"    e0a = {e0a:.2f} m  ->  tip deflection = {deflection:.6e} m")


def main() -> None:
    part_a_single_element_patch_test()
    part_b_cantilever_softening_study()
    print("tutorial complete: the same Mesh/DofHandler/Assembler/LinearStaticAnalysis pipeline")
    print("served both continuum theories - only the Theory and ConstitutiveModel changed.")


if __name__ == "__main__":
    main()
