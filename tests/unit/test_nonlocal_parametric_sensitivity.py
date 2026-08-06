"""Stage 4 Step 6: parametric sensitivity - numerical integration order and element type.

Every other Step 6 item (characteristic-length sensitivity, mesh-density sensitivity,
conditioning) is already covered by ``examples/ex09_eringen_differential_parametric_study.py``
and ``test_nonlocal_conditioning_and_energy.py``. Two items were not yet directly exercised:

1. **Numerical integration order.** ``NonlocalContinuumElement`` accepts an explicit
   ``quadrature_order`` (defaulting to ``2 * interpolation_order``) - never varied by any
   existing test. For a T3 (affine mapping, linear shape functions), every integrand in
   ``K_ue``/``K_ee`` is a low-degree polynomial (degree <= 2), so the default rule is already
   exact; increasing the order must change nothing beyond floating-point noise.
2. **Element type.** The plate-with-a-hole benchmark already uses ``quad4`` incidentally, but no
   test states element-type independence as its own point. This file adds the ``quad4``
   counterpart of ``test_nonlocal_cantilever_benchmark.py`` (same geometry, load, and mesh
   density progression) to confirm the mesh-convergent local limit and nonlocal softening hold
   for Q4 exactly as they do for T3 - the mixed formulation's behavior is a property of the
   theory, not of one specific element family.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.analysis.static import LinearStaticAnalysis
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import NodalLoad
from nanofem.core.model import DomainDefinition, Model
from nanofem.elements.continuum.nonlocal_continuum import NonlocalContinuumElement
from nanofem.geometry.plane import PlaneGeometry
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.numerics.reference.enums import CellType
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

# ---------------------------------------------------------------------------
# Numerical integration order sensitivity
# ---------------------------------------------------------------------------

_SCALENE_TRIANGLE = np.array([[0.3, -0.2], [2.1, 0.4], [0.9, 1.7]])
_N_DOF = 15  # 3 nodes x (2 u-components + 3 Voigt e*-components)


def _element_with_order(order: int | None) -> NonlocalContinuumElement:
    theory = EringenDifferentialTheory(dim=2)
    material = Material("steel", E=YOUNG_MODULUS, nu=POISSON, e0a=0.25)
    constitutive = EringenDifferentialMaterial(PlaneStressConstitutive())
    return NonlocalContinuumElement(
        cell_id=0,
        node_ids=(0, 1, 2),
        coordinates=_SCALENE_TRIANGLE,
        global_dofs=np.arange(_N_DOF, dtype=np.int64),
        cell_type=CellType.TRIANGLE,
        interpolation_order=1,
        theory=theory,
        constitutive=constitutive,
        material=material,
        quadrature_order=order,
    )


@pytest.mark.parametrize("order", [3, 4, 5])
def test_quadrature_order_above_default_does_not_change_the_stiffness(order: int) -> None:
    """Default order (``2*interpolation_order = 2`` for a T3) is already exact for these
    integrands (mass term: shape-function products, degree 2; diffusion term: constant gradient
    products, degree 0) - raising the order must reproduce the same matrix to machine precision,
    not merely "close"."""
    k_default = _element_with_order(None).local_stiffness()
    k_higher = _element_with_order(order).local_stiffness()
    np.testing.assert_allclose(k_higher, k_default, atol=1e-9 * np.abs(k_default).max(), rtol=0.0)


# ---------------------------------------------------------------------------
# Element type: the Q4 counterpart of the T3 cantilever benchmark
# ---------------------------------------------------------------------------


def _cantilever_mesh_q4(n_x: int, n_y: int) -> tuple[Mesh, tuple[int, ...]]:
    xs = np.linspace(0.0, LENGTH, n_x + 1)
    ys = np.linspace(-HEIGHT / 2.0, HEIGHT / 2.0, n_y + 1)
    coords = np.array([[x, y] for y in ys for x in xs])

    def node_id(i: int, j: int) -> int:
        return j * (n_x + 1) + i

    quads = [
        [node_id(i, j), node_id(i + 1, j), node_id(i + 1, j + 1), node_id(i, j + 1)]
        for j in range(n_y)
        for i in range(n_x)
    ]
    block = CellBlock("quad4", np.array(quads), region="plate")
    left_nodes = tuple(node_id(0, j) for j in range(n_y + 1))
    right_nodes = tuple(node_id(n_x, j) for j in range(n_y + 1))
    mesh = Mesh(coords, (block,), (Region("left", 0, left_nodes), Region("right", 0, right_nodes)))
    return mesh, right_nodes


def _q4_tip_deflection_classical(n_x: int, n_y: int) -> float:
    mesh, right_nodes = _cantilever_mesh_q4(n_x, n_y)
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
    return float(
        np.mean([result.displacements[dof_handler.global_dof(n, "u", "y")] for n in right_nodes])
    )


def _q4_tip_deflection_eringen(n_x: int, n_y: int, e0a: float) -> float:
    mesh, right_nodes = _cantilever_mesh_q4(n_x, n_y)
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
    return float(
        np.mean([result.displacements[dof_handler.global_dof(n, "u", "y")] for n in right_nodes])
    )


def test_q4_e0a_zero_converges_to_classical_q4_fem_under_refinement() -> None:
    """The same mesh-convergent (not exact-on-a-fixed-mesh) local limit already established for
    T3 (``test_nonlocal_cantilever_benchmark.py``), independently confirmed for Q4."""
    levels = [(6, 3), (10, 5), (16, 8)]
    rel_errors = []
    for n_x, n_y in levels:
        classical_tip = _q4_tip_deflection_classical(n_x, n_y)
        mixed_tip = _q4_tip_deflection_eringen(n_x, n_y, e0a=0.0)
        rel_errors.append(abs(mixed_tip - classical_tip) / abs(classical_tip))

    for coarser, finer in zip(rel_errors, rel_errors[1:], strict=False):
        assert finer < coarser, f"Q4 e0a=0 discrepancy must shrink under refinement: {rel_errors}"
    assert rel_errors[-1] < 0.10, f"finest-mesh Q4 e0a=0 discrepancy {rel_errors[-1]:.2%} too large"


def test_q4_tip_deflection_increases_monotonically_with_e0a() -> None:
    """Nonlocal softening, confirmed for Q4 exactly as for T3."""
    n_x, n_y = 10, 5
    e0a_values = [0.0, 0.05, 0.1, 0.2]
    deflections = [abs(_q4_tip_deflection_eringen(n_x, n_y, e0a)) for e0a in e0a_values]
    for smaller, larger in zip(deflections, deflections[1:], strict=False):
        assert larger > smaller, f"Q4 tip deflection must grow with e0a: {deflections}"
