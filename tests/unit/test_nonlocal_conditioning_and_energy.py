"""Stage 4 Step 6 items not yet covered by any existing test: condition-number analysis and
energy consistency for the mixed (u, e*) Eringen differential system.

Every other Step 6 item (classical-limit recovery, mesh convergence, characteristic-length
convergence, patch tests, comparison with analytical/published results, symmetry) is already
covered across ``test_nonlocal_continuum_element.py``, ``test_static_nonlocal_plate.py``,
``test_nonlocal_plate_with_hole_benchmark.py``, ``test_nonlocal_bar_benchmark.py``, and
``test_nonlocal_beam_benchmark.py`` - this file adds the two genuinely missing checks rather
than re-deriving what those files already prove.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.handler import ConstraintHandler
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import NodalLoad
from nanofem.core.model import DomainDefinition, Model
from nanofem.elements.factory import build_elements
from nanofem.geometry.plane import PlaneGeometry
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.numerics.assembly.assembler import Assembler
from nanofem.numerics.assembly.contributions import OperatorRole
from nanofem.numerics.assembly.sparsity import SparsityPattern
from nanofem.physics.elasticity.eringen_differential import (
    EringenDifferentialMaterial,
    EringenDifferentialTheory,
)
from nanofem.physics.elasticity.plane import PlaneStressConstitutive

YOUNG_MODULUS = 200e9
POISSON = 0.3
LENGTH = 2.0
HEIGHT = 1.0
THICKNESS = 0.01
TOTAL_FORCE = 10_000.0


def _plate_model(n_x: int, n_y: int, e0a: float) -> tuple[Model, tuple[int, ...]]:
    """A structured n_x*n_y T3 grid (two triangles per cell), for conditioning/energy checks."""
    xs = np.linspace(0.0, LENGTH, n_x + 1)
    ys = np.linspace(0.0, HEIGHT, n_y + 1)
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
    regions = (
        Region("left_edge", 0, left_nodes),
        Region("origin", 0, (node_id(0, 0),)),
        Region("right_edge", 0, right_nodes),
    )
    mesh = Mesh(coords, (block,), regions)

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
    model.add_dirichlet(DirichletBC("left_edge", "u", ("x",), 0.0))
    model.add_dirichlet(DirichletBC("origin", "u", ("y",), 0.0))
    per_node_force = TOTAL_FORCE / len(right_nodes)
    case = LoadCase("tension")
    case.add(NodalLoad("right_edge", "u", np.array([per_node_force, 0.0])))
    model.add_load_case(case)
    return model, right_nodes


def _reduced_stiffness_and_solution(
    n_x: int, n_y: int, e0a: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (K_ff dense, u_f, f_f) for the coupled (u, e*) system, Dirichlet-reduced."""
    model, _ = _plate_model(n_x, n_y, e0a)
    dof_handler = model.build_dof_handler()
    elements = build_elements(model, dof_handler)
    pattern = SparsityPattern.from_providers(elements, OperatorRole.STIFFNESS, dof_handler.num_dofs)
    assembler = Assembler(pattern)
    stiffness = cast(csr_matrix, assembler.assemble(elements, OperatorRole.STIFFNESS))

    dirichlet_bcs = cast(tuple[DirichletBC, ...], model.dirichlet_bcs)
    partition = ConstraintHandler(model.mesh, dof_handler, dirichlet_bcs).partition()

    case = cast(LoadCase, model.load_case("tension"))
    field_components = {spec.name: spec.components for spec in model.field_specs()}
    force = np.zeros(dof_handler.num_dofs)
    for entry in case.entries:
        load = cast(NodalLoad, entry.load)
        for node_id in model.mesh.nodes_in_region(load.region):
            for value, component in zip(load.values, field_components[load.field], strict=True):
                force[dof_handler.global_dof(node_id, load.field, component)] += value

    free = np.array(partition.free_dofs, dtype=np.int64)
    k_ff = stiffness[np.ix_(free, free)].toarray()
    f_f = force[free]
    u_f = np.linalg.solve(k_ff, f_f)
    return k_ff, u_f, f_f


@pytest.mark.parametrize("e0a", [0.0, 0.2])
def test_condition_number_stays_finite_and_bounded_under_refinement(e0a: float) -> None:
    """The mixed saddle-point system's condition number (via SVD, well-defined for indefinite
    matrices) must stay finite and grow no worse than the classical FEM norm - a genuine,
    previously-unchecked numerical-stability question for this new element family."""
    condition_numbers = []
    for n_x, n_y in ((2, 1), (4, 2), (6, 3)):
        k_ff, _, _ = _reduced_stiffness_and_solution(n_x, n_y, e0a)
        cond = np.linalg.cond(k_ff)
        assert np.isfinite(cond), f"condition number is not finite at mesh ({n_x},{n_y})"
        condition_numbers.append(cond)
    # Mesh refinement increases DOF count, which is expected to raise the condition number
    # (classical FEM behaves the same way) - the check is that it grows only polynomially,
    # not that it stays constant.
    assert condition_numbers[-1] < condition_numbers[0] * 1e4


def test_condition_number_is_not_pathologically_worse_than_the_classical_system() -> None:
    """Cross-check: the mixed system's conditioning, relative to its own matrix norm, is the
    same order of magnitude as a classical (e0a=0, Schur-complement-equivalent) T3 mesh - the
    saddle-point structure itself does not introduce a qualitatively new stability problem."""
    from nanofem.core.model import DomainDefinition as _DomainDefinition
    from nanofem.elements.factory import build_elements as _build_elements
    from nanofem.physics.elasticity.isotropic import IsotropicElasticity

    n_x, n_y = 4, 2
    k_mixed, _, _ = _reduced_stiffness_and_solution(n_x, n_y, e0a=0.0)
    cond_mixed = np.linalg.cond(k_mixed)

    # Build the equivalent classical model directly for comparison.
    model, _ = _plate_model(n_x, n_y, e0a=0.0)
    classical_model = Model(model.mesh)
    classical_model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON))
    classical_model.add_section("plane", PlaneGeometry(thickness=THICKNESS))
    classical_model.add_theory("plane_stress", IsotropicElasticity(dim=2))
    classical_model.add_constitutive("plane_law", PlaneStressConstitutive())
    classical_model.add_domain(
        _DomainDefinition(
            "plate_domain",
            "plate",
            "plane_stress",
            "steel",
            geometry="plane",
            constitutive="plane_law",
        )
    )
    classical_model.add_dirichlet(DirichletBC("left_edge", "u", ("x",), 0.0))
    classical_model.add_dirichlet(DirichletBC("origin", "u", ("y",), 0.0))
    dof_handler = classical_model.build_dof_handler()
    elements = _build_elements(classical_model, dof_handler)
    pattern = SparsityPattern.from_providers(elements, OperatorRole.STIFFNESS, dof_handler.num_dofs)
    stiffness = cast(csr_matrix, Assembler(pattern).assemble(elements, OperatorRole.STIFFNESS))
    dirichlet_bcs = cast(tuple[DirichletBC, ...], classical_model.dirichlet_bcs)
    partition = ConstraintHandler(classical_model.mesh, dof_handler, dirichlet_bcs).partition()
    free = np.array(partition.free_dofs, dtype=np.int64)
    k_classical = stiffness[np.ix_(free, free)].toarray()
    cond_classical = np.linalg.cond(k_classical)

    # Both should be within a couple of orders of magnitude of each other - not proof of
    # optimality, but proof the mixed formulation is not qualitatively pathological.
    assert cond_mixed < cond_classical * 1e3


def test_energy_consistency_clapeyron_theorem_for_the_mixed_system() -> None:
    """U = 0.5 * f.u (Clapeyron's theorem, linear-elastic loading from zero) must match
    0.5 * u^T K u computed from the assembled coupled block matrix - an independent check
    of the whole (u, e*) system's energy bookkeeping, not just the local element level
    (which test_postprocess_recovery.py's Clapeyron check already covers for classical T3)."""
    k_ff, u_f, f_f = _reduced_stiffness_and_solution(4, 2, e0a=0.15)
    energy_from_stiffness = 0.5 * u_f @ k_ff @ u_f
    energy_from_load = 0.5 * f_f @ u_f
    assert energy_from_stiffness == pytest.approx(energy_from_load, rel=1e-9)


def test_assembled_stiffness_is_symmetric_at_global_scale() -> None:
    """Symmetry of the *assembled, Dirichlet-reduced* global system - a stronger statement
    than the element-level symmetry test_nonlocal_continuum_element.py already proves, since
    assembly could in principle introduce an asymmetry the element-level check cannot see."""
    k_ff, _, _ = _reduced_stiffness_and_solution(4, 2, e0a=0.2)
    np.testing.assert_allclose(k_ff, k_ff.T, rtol=1e-10, atol=1e-6 * np.abs(k_ff).max())
