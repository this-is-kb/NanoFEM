"""Eringen differential nonlocal elasticity: the 2-D clamped cantilever benchmark.

The Stage-4 benchmark suite's explicit "2D: Cantilever" target, complementing the 1-D nonlocal
bar/beam (v0.20.0/v0.23.0) and the plate-with-a-hole (v0.22.0). A rectangular plate, clamped
(both displacement components fixed) at ``x=0``, under a tip shear load at ``x=L``, meshed with
T3 - the same ``NonlocalContinuumElement``/``EringenDifferentialTheory`` machinery, applied to a
genuinely different geometry and loading than the plate-with-hole.

**Building this benchmark discovered an important, previously undocumented property of this
mixed formulation** - see ``docs/design/ERINGEN_DIFFERENTIAL_CONTINUUM.md`` Section 7 for the
full account, verified independently before this file's tolerances were chosen. In short:
because ``e*`` is C0-continuous (shared between elements) while a T3's own classical strain is
naturally *discontinuous* between elements, the ``e0a=0`` local limit exactly matches a
directly-computed classical FEM solution only for a constant-strain field (already proven at
the element level, ``test_nonlocal_continuum_element.py``) - for a genuinely non-constant field
like this cantilever's bending strain, ``e0a=0`` recovery is a **mesh-convergent** property
(the discrepancy shrinks from ~186% at a 4x2 mesh to ~4% at 32x16, verified independently
before this file was written), not an exact-on-any-mesh one. This is the well-documented,
accepted behavior of implicit-gradient-type mixed regularization models generally (the same
mathematical structure as Peerlings-style gradient damage), not a defect specific to NanoFEM.
Every test below checks mesh-*convergent* recovery, never exact-match-on-a-fixed-mesh recovery,
for this reason.
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
from nanofem.physics.elasticity.plane import PlaneStressConstitutive

YOUNG_MODULUS = 200e9
POISSON = 0.3
THICKNESS = 0.05
LENGTH = 2.0
HEIGHT = 0.4
TIP_FORCE = 1_000.0


def _cantilever_mesh(n_x: int, n_y: int) -> tuple[Mesh, tuple[int, ...], tuple[int, ...]]:
    """A structured n_x*n_y T3 grid of a rectangular cantilever, clamped at x=0."""
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
    return mesh, left_nodes, right_nodes


def _nonlocal_tip_deflection(n_x: int, n_y: int, e0a: float) -> float:
    mesh, left_nodes, right_nodes = _cantilever_mesh(n_x, n_y)
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


def _classical_tip_deflection(n_x: int, n_y: int) -> tuple[float, np.ndarray]:
    """Returns (tip deflection, reactions) for the directly-computed classical FEM solution."""
    mesh, left_nodes, right_nodes = _cantilever_mesh(n_x, n_y)
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON))
    model.add_section("plane", PlaneGeometry(thickness=THICKNESS))
    model.add_theory("plane_stress", IsotropicElasticity(dim=2))
    model.add_constitutive("plane_law", PlaneStressConstitutive())
    model.add_domain(
        DomainDefinition(
            "plate_domain",
            "plate",
            "plane_stress",
            "steel",
            geometry="plane",
            constitutive="plane_law",
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
    return tip, result.reactions


def _timoshenko_tip_deflection() -> float:
    """The classical closed-form Timoshenko cantilever tip deflection under a shear tip load -
    the "gold standard" both the classical FEM and the mixed e0a=0 FEM should approach as the
    2-D continuum mesh refines and the clamped-root effect diminishes."""
    second_moment = THICKNESS * HEIGHT**3 / 12.0
    shear_modulus = YOUNG_MODULUS / (2.0 * (1.0 + POISSON))
    shear_correction = 5.0 / 6.0  # rectangular cross-section (Cowper)
    shear_area = shear_correction * THICKNESS * HEIGHT
    bending_term = TIP_FORCE * LENGTH**3 / (3.0 * YOUNG_MODULUS * second_moment)
    shear_term = TIP_FORCE * LENGTH / (shear_modulus * shear_area)
    return -(bending_term + shear_term)


def test_e0a_zero_converges_to_the_classical_fem_solution_under_mesh_refinement() -> None:
    """Mesh-CONVERGENT (not exact-on-a-fixed-mesh) recovery, per Section 7 of the design doc -
    the relative discrepancy between the mixed system at e0a=0 and a directly-computed
    classical FEM solution must shrink monotonically as the mesh refines."""
    levels = [(6, 3), (10, 5), (16, 8)]
    rel_errors = []
    for n_x, n_y in levels:
        classical_tip, _ = _classical_tip_deflection(n_x, n_y)
        mixed_tip = _nonlocal_tip_deflection(n_x, n_y, e0a=0.0)
        rel_errors.append(abs(mixed_tip - classical_tip) / abs(classical_tip))

    for coarser, finer in zip(rel_errors, rel_errors[1:], strict=False):
        assert finer < coarser, f"e0a=0 discrepancy must shrink under refinement: {rel_errors}"
    assert rel_errors[-1] < 0.30, f"finest-mesh e0a=0 discrepancy {rel_errors[-1]:.2%} too large"


def test_classical_and_nonlocal_both_converge_toward_timoshenko_beam_theory() -> None:
    """Independent sanity check: both the classical FEM and the mixed e0a=0 FEM approach the
    same Timoshenko closed form as the mesh refines (the well-known clamped-root discrepancy
    shrinking with refinement, exactly as it does for the purely classical benchmark this
    mirrors), confirming neither pipeline has an unrelated, independent error."""
    beam_theory = _timoshenko_tip_deflection()
    classical_ratios = []
    mixed_ratios = []
    for n_x, n_y in [(6, 3), (10, 5), (16, 8)]:
        classical_tip, _ = _classical_tip_deflection(n_x, n_y)
        mixed_tip = _nonlocal_tip_deflection(n_x, n_y, e0a=0.0)
        classical_ratios.append(classical_tip / beam_theory)
        mixed_ratios.append(mixed_tip / beam_theory)

    for coarser, finer in zip(classical_ratios, classical_ratios[1:], strict=False):
        assert finer > coarser, "classical FEM ratio to beam theory must increase toward 1"
    assert classical_ratios[-1] > 0.85
    # The mixed system at e0a=0 need not converge as smoothly (Section 7's mesh-dependent
    # discretization artifact also affects this ratio), but must land in the same right
    # ballpark at the finest mesh, not diverge to an unrelated value.
    assert 0.7 < mixed_ratios[-1] < 1.3


@pytest.mark.parametrize("n_x,n_y", [(10, 5)])
def test_tip_deflection_increases_monotonically_with_e0a(n_x: int, n_y: int) -> None:
    """Nonlocal softening: increasing the characteristic length increases compliance (larger
    magnitude tip deflection under the same load) - the same direction found for the 1-D
    nonlocal bar/beam (v0.20.0/v0.23.0)."""
    e0a_values = [0.0, 0.05, 0.1, 0.2]
    deflections = [abs(_nonlocal_tip_deflection(n_x, n_y, e0a)) for e0a in e0a_values]
    for smaller, larger in zip(deflections, deflections[1:], strict=False):
        assert larger > smaller, f"tip deflection must grow with e0a: {deflections}"


def test_reactions_balance_the_applied_tip_load() -> None:
    """Global equilibrium: the classical FEM solve's reactions must sum to minus the applied
    tip force exactly - a plain sanity check on the mesh/BC/load setup this benchmark shares
    with the nonlocal solves."""
    _, reactions = _classical_tip_deflection(10, 5)
    assert reactions.sum() == pytest.approx(TIP_FORCE, rel=1e-9)


def test_nonlocal_solution_is_mesh_convergent_for_nonzero_e0a() -> None:
    """The mixed formulation stays well-posed (finite, convergent, not oscillating) on a
    genuinely 2-D, non-constant-stress cantilever problem at nonzero e0a."""
    e0a = 0.1
    deflections = [
        _nonlocal_tip_deflection(n_x, n_y, e0a) for n_x, n_y in [(6, 3), (10, 5), (16, 8)]
    ]
    assert all(np.isfinite(d) for d in deflections)
    assert abs(deflections[-1] - deflections[-2]) < abs(deflections[-2]) * 0.25
