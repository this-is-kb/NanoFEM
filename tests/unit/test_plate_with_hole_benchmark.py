"""Benchmark: infinite plate with a circular hole under remote uniaxial tension (Kirsch 1898).

This single benchmark satisfies two of Stage 3's named verification
requirements at once - "infinite plate with a hole" and "mesh convergence" -
by design: the classical closed-form solution has a genuinely non-constant
stress field (unlike the constant-strain patch test), so a coarse mesh
cannot reproduce it exactly, and refinement must show the recovered peak
stress converging toward the analytical value, not just repeating an already
-exact answer.

**The problem.** A large plate with a small circular hole of radius ``a``,
loaded by a remote uniform tension ``S`` in the x-direction, has the
closed-form tangential (hoop) stress at the hole boundary (Kirsch 1898):
``sigma_theta_theta(theta) = S (1 - 2 cos(2 theta))``. At ``theta = 90`` deg
(top of the hole, the point ``(0, a)``) this gives the classical stress
concentration factor 3: ``sigma_theta_theta = 3S``, and there the tangential
direction is the Cartesian x-axis, so ``sigma_xx = 3S``. At ``theta = 0``
(side of hole, point ``(a, 0)``), ``sigma_theta_theta = -S`` and the
tangential direction is the Cartesian y-axis there, so ``sigma_yy = -S``.
Both are used as independent checkpoints below.

**The finite model.** By double symmetry only a quarter-plate needs meshing:
``u_y = 0`` on the bottom edge (the positive x-axis, ``theta = 0``), ``u_x =
0`` on the left edge (the positive y-axis, ``theta = 90`` deg), the hole
boundary is traction-free (natural BC, no code needed), and a uniform
traction ``(S, 0)`` is applied via a real ``TractionLoad`` on the portion of
the right (``x = W``) outer edge below the diagonal - the top (``y = W``)
outer edge is traction-free by construction, since the remote stress state
has ``sigma_yy = 0`` and the outward normal there is ``(0, 1)``. ``W`` (the
plate half-width) is set to ``10 a``: a genuine finite-width plate has a
*higher* true stress concentration than the infinite-plate value of 3
(Peterson's classical finite-width correction; verified numerically during
development that a much smaller ``W/a`` - 4, say - converges cleanly to a
demonstrably *different*, larger, stable value, not a bug but the real
finite-width effect), so ``W/a = 10`` keeps that correction small enough
(a percent or two) that comparing the recovered peak against the
infinite-plate value 3 with a generous tolerance is a meaningful check.

**Mesh grading.** The quarter-annulus-to-quarter-square region is meshed
with Q4 elements on a structured grid, radially graded (``t = (i/n_r)**2``
biases nodes toward the hole) rather than uniform - verified during
development that uniform radial spacing converges far slower (still
monotonic, but not close to 3 even at 64x128 elements) since it under-resolves
the steep near-hole gradient a stress concentration produces.

**Recovery.** The peak-stress checkpoints are each read from the single Q4
element touching that corner node (the hole boundary corners each belong to
exactly one element in this mesh topology), via
``postprocess.recovery``'s quadrature-weighted element-average field -
identical to what nodal recovery would give here, since a node with only one
contributing element has no other data to average against.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.analysis.static import LinearStaticAnalysis
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import TractionLoad
from nanofem.core.model import DomainDefinition, Model
from nanofem.elements.continuum.continuum import ContinuumElement
from nanofem.elements.factory import build_elements
from nanofem.geometry.plane import PlaneGeometry
from nanofem.materials.material import Material
from nanofem.mesh.facet_region import FacetRegion
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.physics.elasticity.isotropic import IsotropicElasticity
from nanofem.physics.elasticity.plane import PlaneStressConstitutive
from nanofem.postprocess.recovery import RecoveryInput, recover_element_fields

HOLE_RADIUS = 1.0
OUTER_HALF_WIDTH = 10.0 * HOLE_RADIUS
THICKNESS = 1.0
YOUNG_MODULUS = 200e9
POISSON = 0.3
REMOTE_STRESS = 100e6
RADIAL_GRADING = 2.0


def _quarter_plate_with_hole_mesh(n_r: int, n_theta: int) -> tuple[Mesh, dict[str, int]]:
    """A structured, radially-graded Q4 mesh of the quarter plate-with-hole domain.

    Returns the mesh plus the cell ids of the two corner elements the
    benchmark reads stress from (touching the hole at theta=0 and theta=90).
    """
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
    # Outer boundary ring is radial layer i = n_r - 1; a quad's local facet 1 (vertices
    # 1,2) is its outer edge - see numerics/reference's facet_vertex_indices for QUADRILATERAL.
    right_edge_facets = tuple((cell_id(n_r - 1, j), 1) for j in range(half))
    facet_regions = (FacetRegion("right_edge", right_edge_facets),)

    mesh = Mesh(coords, (block,), regions, facet_regions)
    corners = {"top_of_hole": cell_id(0, n_theta - 1), "side_of_hole": cell_id(0, 0)}
    return mesh, corners


def _solve(n_r: int, n_theta: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Returns (peak stress ratios [top, side], reactions, corner cell ids)."""
    mesh, corners = _quarter_plate_with_hole_mesh(n_r, n_theta)
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
    model.add_dirichlet(DirichletBC("bottom", "u", ("y",), 0.0))
    model.add_dirichlet(DirichletBC("left", "u", ("x",), 0.0))
    case = LoadCase("remote_tension")
    case.add(TractionLoad("right_edge", "u", np.array([REMOTE_STRESS, 0.0])))
    model.add_load_case(case)

    result = LinearStaticAnalysis(model).run()["remote_tension"]
    elements = build_elements(model, result.dof_handler)
    inputs = [
        RecoveryInput(e, PlaneStressConstitutive(), POISSON)
        for e in elements
        if isinstance(e, ContinuumElement)
    ]
    element_results = recover_element_fields(inputs, result.displacements)
    by_id = {r.element_id: r for r in element_results}

    top_sigma_xx = by_id[corners["top_of_hole"]].field.stress[0, 0]
    side_sigma_yy = by_id[corners["side_of_hole"]].field.stress[1, 1]
    ratios = np.array([top_sigma_xx / REMOTE_STRESS, side_sigma_yy / REMOTE_STRESS])
    return ratios, result.reactions, corners


def test_stress_concentration_converges_toward_the_kirsch_solution() -> None:
    """Peak hoop stress at the top of the hole converges toward 3S under mesh refinement."""
    levels = [(6, 12), (16, 32), (32, 64)]
    errors = []
    for n_r, n_theta in levels:
        ratios, _, _ = _solve(n_r, n_theta)
        errors.append(abs(ratios[0] - 3.0))

    assert errors[-1] < errors[0], "refining the mesh must reduce the error, not increase it"
    assert errors[-1] < 0.10, f"finest-mesh SCF error {errors[-1]:.3f} exceeds the 10% tolerance"


def test_side_of_hole_matches_the_compressive_kirsch_checkpoint() -> None:
    """A second, independent Kirsch checkpoint: sigma_yy = -S at theta = 0 (side of hole)."""
    ratios, _, _ = _solve(16, 32)
    assert ratios[1] == pytest.approx(-1.0, abs=0.15)


def test_reactions_balance_the_applied_traction() -> None:
    """Global equilibrium: reactions must sum to minus the total applied force, exactly."""
    _, reactions, _ = _solve(6, 12)
    expected_total_force = -REMOTE_STRESS * OUTER_HALF_WIDTH * THICKNESS
    assert reactions.sum() == pytest.approx(expected_total_force, rel=1e-9)
