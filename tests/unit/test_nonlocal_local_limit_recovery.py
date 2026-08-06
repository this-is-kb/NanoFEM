"""Stage 4 Step 5: local-limit (e0a -> 0) recovery, quantified field-by-field.

``test_nonlocal_cantilever_benchmark.py`` already establishes mesh-convergent recovery of the
classical FEM *displacement* at e0a=0 (Section 7 of ``docs/design/ERINGEN_DIFFERENTIAL_
CONTINUUM.md``). The Stage 4 directive asks for the same demonstration at the level of strain,
stress, energy, and stiffness individually, with the numerical error quantified - this file adds
exactly those four checks, on the same cantilever benchmark, so the results are directly
comparable to the already-established displacement numbers rather than a new, unrelated setup.

Every threshold below was set from numbers measured by running the actual solves first (this
project's established discipline - see ``docs/dev/notes.md``), not guessed:

======  ============  ============  ============  ==============
 mesh    strain err     stress err    energy err    stiffness err
======  ============  ============  ============  ==============
 6x3        54.0%         27.1%         91.6%          47.8%
10x5        42.3%         19.4%         36.2%          26.6%
16x8        30.2%         15.7%         14.7%          12.8%
======  ============  ============  ============  ==============

All four quantities shrink monotonically under refinement, exactly like the displacement error
already established - confirming the mesh-convergent local limit is a property of the whole
solution (field variables, derived quantities, and energy alike), not just the nodal
displacement.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from nanofem.analysis.static import LinearStaticAnalysis
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import NodalLoad
from nanofem.core.model import DomainDefinition, Model
from nanofem.elements.continuum.continuum import ContinuumElement
from nanofem.elements.continuum.nonlocal_continuum import NonlocalContinuumElement
from nanofem.elements.factory import build_elements
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
LENGTH = 2.0
HEIGHT = 0.4
TIP_FORCE = 1_000.0

MESH_LEVELS = [(6, 3), (10, 5), (16, 8)]


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
    mesh = Mesh(coords, (block,), (Region("left", 0, left_nodes), Region("right", 0, right_nodes)))
    return mesh, right_nodes


def _corner_element_index(
    mesh: Mesh, elements: tuple[ContinuumElement | NonlocalContinuumElement, ...]
) -> int:
    """The element nearest the clamped root and top fiber - where bending strain/stress peak."""
    best_index, best_score = 0, float("inf")
    for index, element in enumerate(elements):
        cx = float(np.mean([mesh.node(n).coordinates[0] for n in element.node_ids]))
        cy = float(np.mean([mesh.node(n).coordinates[1] for n in element.node_ids]))
        score = cx + abs(cy - HEIGHT / 2.0)
        if score < best_score:
            best_score, best_index = score, index
    return best_index


def _solve_mixed(n_x: int, n_y: int, e0a: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Returns (tip deflection, root-fiber strain, root-fiber stress) for the mixed system."""
    mesh, right_nodes = _cantilever_mesh(n_x, n_y)
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON, e0a=e0a))
    model.add_section("plane", PlaneGeometry(thickness=THICKNESS))
    model.add_theory("nonlocal", EringenDifferentialTheory(dim=2))
    model.add_constitutive("nonlocal_law", EringenDifferentialMaterial(PlaneStressConstitutive()))
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
    tip = float(
        np.mean([result.displacements[dof_handler.global_dof(n, "u", "y")] for n in right_nodes])
    )

    elements = cast("tuple[NonlocalContinuumElement, ...]", build_elements(model, dof_handler))
    target = elements[_corner_element_index(mesh, elements)]
    assert isinstance(target, NonlocalContinuumElement)
    local_disp = result.displacements[target.global_dofs]
    strain, stress, _ = target.quadrature_point_response(local_disp)
    return tip, strain[0], stress[0]


def _solve_classical(n_x: int, n_y: int) -> tuple[float, np.ndarray, np.ndarray]:
    mesh, right_nodes = _cantilever_mesh(n_x, n_y)
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON))
    model.add_section("plane", PlaneGeometry(thickness=THICKNESS))
    model.add_theory("elastic", IsotropicElasticity(dim=2))
    model.add_constitutive("law", PlaneStressConstitutive())
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
    tip = float(
        np.mean([result.displacements[dof_handler.global_dof(n, "u", "y")] for n in right_nodes])
    )

    elements = cast("tuple[ContinuumElement, ...]", build_elements(model, dof_handler))
    target = elements[_corner_element_index(mesh, elements)]
    assert isinstance(target, ContinuumElement)
    local_disp = result.displacements[target.global_dofs]
    strain, stress, _ = target.quadrature_point_response(local_disp)
    return tip, strain[0], stress[0]


def test_strain_recovery_converges_to_classical_under_refinement() -> None:
    """Recovered strain at a fixed physical location (the clamped-root top fiber, where bending
    strain peaks) approaches the classical solution's strain as the mesh refines."""
    errors = []
    for n_x, n_y in MESH_LEVELS:
        _, strain_mixed, _ = _solve_mixed(n_x, n_y, e0a=0.0)
        _, strain_classical, _ = _solve_classical(n_x, n_y)
        errors.append(
            float(
                np.linalg.norm(strain_mixed - strain_classical) / np.linalg.norm(strain_classical)
            )
        )
    for coarser, finer in zip(errors, errors[1:], strict=False):
        assert finer < coarser, f"strain recovery error must shrink under refinement: {errors}"
    assert errors[-1] < 0.35, f"finest-mesh strain recovery error {errors[-1]:.2%} too large"


def test_stress_recovery_converges_to_classical_under_refinement() -> None:
    """Same check at the stress level (sigma* = D:e*, so this is a genuinely independent
    quantity from strain only insofar as it is a linear transform of it - both are checked
    since the directive explicitly asks for both)."""
    errors = []
    for n_x, n_y in MESH_LEVELS:
        _, _, stress_mixed = _solve_mixed(n_x, n_y, e0a=0.0)
        _, _, stress_classical = _solve_classical(n_x, n_y)
        errors.append(
            float(
                np.linalg.norm(stress_mixed - stress_classical) / np.linalg.norm(stress_classical)
            )
        )
    for coarser, finer in zip(errors, errors[1:], strict=False):
        assert finer < coarser, f"stress recovery error must shrink under refinement: {errors}"
    assert errors[-1] < 0.20, f"finest-mesh stress recovery error {errors[-1]:.2%} too large"


def test_strain_energy_converges_to_classical_under_refinement() -> None:
    """Strain energy (Clapeyron's theorem, U = 0.5*F*delta for a single tip load) recovery,
    tracked as a mesh-refinement trend rather than the single-mesh check
    ``test_nonlocal_conditioning_and_energy.py`` already performs."""
    errors = []
    for n_x, n_y in MESH_LEVELS:
        tip_mixed, _, _ = _solve_mixed(n_x, n_y, e0a=0.0)
        tip_classical, _, _ = _solve_classical(n_x, n_y)
        energy_mixed = 0.5 * TIP_FORCE * abs(tip_mixed)
        energy_classical = 0.5 * TIP_FORCE * abs(tip_classical)
        errors.append(abs(energy_mixed - energy_classical) / energy_classical)
    for coarser, finer in zip(errors, errors[1:], strict=False):
        assert finer < coarser, f"energy recovery error must shrink under refinement: {errors}"
    assert errors[-1] < 0.20, f"finest-mesh energy recovery error {errors[-1]:.2%} too large"


def test_effective_stiffness_converges_to_classical_under_refinement() -> None:
    """Effective structural stiffness (k = F / |deflection|, the same information the
    displacement-convergence test already carries, reframed as a stiffness quantity per the
    directive's own explicit wording) converges to the classical value under refinement."""
    errors = []
    for n_x, n_y in MESH_LEVELS:
        tip_mixed, _, _ = _solve_mixed(n_x, n_y, e0a=0.0)
        tip_classical, _, _ = _solve_classical(n_x, n_y)
        k_mixed = TIP_FORCE / abs(tip_mixed)
        k_classical = TIP_FORCE / abs(tip_classical)
        errors.append(abs(k_mixed - k_classical) / k_classical)
    for coarser, finer in zip(errors, errors[1:], strict=False):
        assert finer < coarser, f"stiffness recovery error must shrink under refinement: {errors}"
    assert errors[-1] < 0.15, f"finest-mesh stiffness recovery error {errors[-1]:.2%} too large"
