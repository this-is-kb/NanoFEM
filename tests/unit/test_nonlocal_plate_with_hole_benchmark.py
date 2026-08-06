"""Eringen differential nonlocal elasticity: the plate-with-a-hole (Kirsch) benchmark.

Reuses the exact quarter-plate-with-hole Q4 mesh/traction/BC setup
``test_plate_with_hole_benchmark.py`` built for classical elasticity (see that file's own
extensive docstring for the mesh-grading/finite-width-correction reasoning, unchanged here) -
only the theory/constitutive law and the recovery path differ, demonstrating the project's own
"theory-independent backbone" design goal directly: Mesh, Q4 shape functions, mapping,
quadrature, Dirichlet/traction BCs, and the linear solver are byte-for-byte the same code paths;
only ``EringenDifferentialTheory``/``EringenDifferentialMaterial``/``NonlocalContinuumElement``
replace ``IsotropicElasticity``/``PlaneStressConstitutive``/``ContinuumElement``.

Recovery goes through ``NonlocalContinuumElement.quadrature_point_response`` directly (a
quadrature-weighted element average, ``np.average(stress, axis=0, weights=point_measure)`` -
the same pattern ``postprocess.recovery`` uses internally for ``ContinuumElement``), not through
``postprocess.recovery`` itself: that module is explicitly scoped (and frozen) to
``ContinuumElement`` + the two classical plane laws.

**What is verified, and why not more.** No closed-form Eringen-differential solution exists in
the literature for a finite plate with a hole (published nonlocal Kirsch-type results are
almost all for the *integral* model, and often asymptotic) - so this benchmark verifies what is
actually provable:
1. At ``e0a=0``, the recovered peak stress ratio converges under mesh refinement to the same
   value the classical benchmark converges to (exact algebraic equivalence, already proven at
   the element level - this is the end-to-end, full-pipeline confirmation of it).
2. As ``e0a`` increases from zero, the peak (concentrated) stress at the hole boundary
   *decreases* - nonlocal elasticity's defining qualitative behavior, stress-concentration
   regularization, verified directionally rather than against an unavailable exact number.
3. The solution remains well-behaved (finite, mesh-convergent, not oscillating) for ``e0a > 0``
   on a genuinely non-constant stress field - the first time this codebase's mixed formulation
   is exercised on anything other than a constant-strain patch test.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.analysis.static import LinearStaticAnalysis
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import TractionLoad
from nanofem.core.model import DomainDefinition, Model
from nanofem.elements.continuum.nonlocal_continuum import NonlocalContinuumElement
from nanofem.elements.factory import build_elements
from nanofem.geometry.plane import PlaneGeometry
from nanofem.materials.material import Material
from nanofem.mesh.facet_region import FacetRegion
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.physics.elasticity.eringen_differential import (
    EringenDifferentialMaterial,
    EringenDifferentialTheory,
)
from nanofem.physics.elasticity.plane import PlaneStressConstitutive

HOLE_RADIUS = 1.0
OUTER_HALF_WIDTH = 10.0 * HOLE_RADIUS
THICKNESS = 1.0
YOUNG_MODULUS = 200e9
POISSON = 0.3
REMOTE_STRESS = 100e6
RADIAL_GRADING = 2.0


def _quarter_plate_with_hole_mesh(n_r: int, n_theta: int) -> tuple[Mesh, dict[str, int]]:
    """Identical to test_plate_with_hole_benchmark.py's own mesh builder - see that file."""
    assert n_theta % 2 == 0, "n_theta must be even so the outer boundary splits cleanly at 45deg"

    def node_id(i: int, j: int) -> int:
        return i * (n_theta + 1) + j

    coords = np.empty(((n_r + 1) * (n_theta + 1), 2))
    for i in range(n_r + 1):
        t = (i / n_r) ** RADIAL_GRADING
        for j in range(n_theta + 1):
            theta = (j / n_theta) * (np.pi / 2.0)
            inner = HOLE_RADIUS * np.array([np.cos(theta), np.sin(theta)])
            if theta <= np.pi / 4.0 + 1e-12:
                outer = np.array([OUTER_HALF_WIDTH, OUTER_HALF_WIDTH * np.tan(theta)])
            else:
                outer = np.array([OUTER_HALF_WIDTH / np.tan(theta), OUTER_HALF_WIDTH])
            coords[node_id(i, j)] = inner + t * (outer - inner)

    def cell_id(i: int, j: int) -> int:
        return i * n_theta + j

    quads = np.array(
        [
            [node_id(i, j), node_id(i + 1, j), node_id(i + 1, j + 1), node_id(i, j + 1)]
            for i in range(n_r)
            for j in range(n_theta)
        ]
    )
    block = CellBlock("quad4", quads, region="plate")

    bottom_nodes = tuple(node_id(i, 0) for i in range(n_r + 1))
    left_nodes = tuple(node_id(i, n_theta) for i in range(n_r + 1))
    regions = (Region("bottom", 0, bottom_nodes), Region("left", 0, left_nodes))

    half = n_theta // 2
    right_edge_facets = tuple((cell_id(n_r - 1, j), 1) for j in range(half))
    facet_regions = (FacetRegion("right_edge", right_edge_facets),)

    mesh = Mesh(coords, (block,), regions, facet_regions)
    corners = {"top_of_hole": cell_id(0, n_theta - 1), "side_of_hole": cell_id(0, 0)}
    return mesh, corners


def _solve(n_r: int, n_theta: int, e0a: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (peak stress ratios [top, side], reactions)."""
    mesh, corners = _quarter_plate_with_hole_mesh(n_r, n_theta)
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON, e0a=e0a))
    model.add_section("plane", PlaneGeometry(thickness=THICKNESS))
    model.add_theory("nonlocal", EringenDifferentialTheory(dim=2))
    model.add_constitutive("nonlocal_law", EringenDifferentialMaterial(PlaneStressConstitutive()))
    model.add_domain(
        DomainDefinition(
            "plate_domain",
            "plate",
            "nonlocal",
            "steel",
            geometry="plane",
            constitutive="nonlocal_law",
        )
    )
    model.add_dirichlet(DirichletBC("bottom", "u", ("y",), 0.0))
    model.add_dirichlet(DirichletBC("left", "u", ("x",), 0.0))
    case = LoadCase("remote_tension")
    case.add(TractionLoad("right_edge", "u", np.array([REMOTE_STRESS, 0.0])))
    model.add_load_case(case)

    result = LinearStaticAnalysis(model).run()["remote_tension"]
    elements = build_elements(model, result.dof_handler)
    by_id: dict[int, NonlocalContinuumElement] = {
        e.cell_id: e for e in elements if isinstance(e, NonlocalContinuumElement)
    }

    def _average_stress(cell_id: int) -> np.ndarray:
        element = by_id[cell_id]
        local = result.displacements[element.global_dofs]
        _, stress, point_measure = element.quadrature_point_response(local)
        return np.asarray(np.average(stress, axis=0, weights=point_measure))

    top_stress = _average_stress(corners["top_of_hole"])
    side_stress = _average_stress(corners["side_of_hole"])
    ratios = np.array([top_stress[0] / REMOTE_STRESS, side_stress[1] / REMOTE_STRESS])
    return ratios, result.reactions


def test_e0a_zero_converges_to_the_classical_kirsch_solution() -> None:
    """The full nonlocal pipeline at e0a=0 reproduces the classical benchmark's own
    mesh-convergence result - the end-to-end confirmation of the element-level exact
    Schur-complement equivalence already proven in test_nonlocal_continuum_element.py."""
    levels = [(6, 12), (16, 32), (32, 64)]
    errors = [abs(_solve(n_r, n_theta, e0a=0.0)[0][0] - 3.0) for n_r, n_theta in levels]
    assert errors[-1] < errors[0], "refining the mesh must reduce the error, not increase it"
    assert errors[-1] < 0.10, f"finest-mesh SCF error {errors[-1]:.3f} exceeds the 10% tolerance"


def test_reactions_balance_the_applied_traction() -> None:
    """Global equilibrium holds for the coupled (u, e*) system exactly as for the classical
    one - the reaction recovery machinery needed no changes for the mixed formulation."""
    _, reactions = _solve(6, 12, e0a=0.0)
    expected_total_force = -REMOTE_STRESS * OUTER_HALF_WIDTH * THICKNESS
    assert reactions.sum() == pytest.approx(expected_total_force, rel=1e-9)


def test_peak_stress_is_regularized_as_e0a_increases() -> None:
    """Nonlocal elasticity's defining qualitative behaviour: increasing the characteristic
    length reduces the concentrated peak stress at the hole boundary, relative to the
    classical (e0a=0) value - no exact published number exists to match for this exact
    problem/model combination, so this checks the direction and monotonicity instead."""
    n_r, n_theta = 10, 20
    e0a_values = [0.0, 0.15, 0.35]
    peak_ratios = [_solve(n_r, n_theta, e0a)[0][0] for e0a in e0a_values]

    for coarser, finer in zip(peak_ratios, peak_ratios[1:], strict=False):
        assert (
            finer < coarser
        ), f"peak stress ratio must strictly decrease as e0a grows: {peak_ratios}"
    # The effect must be real (well above solver/discretization noise), not negligible.
    assert peak_ratios[0] - peak_ratios[-1] > 0.05


def test_solution_is_mesh_convergent_for_nonzero_e0a() -> None:
    """The mixed formulation stays well-posed (finite, convergent) on a genuinely
    non-constant stress field, not just the trivial constant-strain case."""
    e0a = 0.2
    levels = [(6, 12), (10, 20)]
    ratios = [_solve(n_r, n_theta, e0a)[0][0] for n_r, n_theta in levels]
    assert all(np.isfinite(r) for r in ratios)
    assert abs(ratios[1] - ratios[0]) < abs(ratios[0]) * 0.5, "should not be wildly oscillating"
