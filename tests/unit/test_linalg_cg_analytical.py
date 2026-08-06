"""``ConjugateGradientSolver``: preconditioned CG against ``SparseDirectSolver`` (SDS 2.15).

Verified two ways: a synthetic random SPD system (where the exact solution
is known by construction) and the real T3 plate stiffness matrix this
session's other tests already build (a genuine FEM system, not a toy) -
both must match ``SparseDirectSolver``'s LU solution to the requested
``rtol``, since the two solvers are alternate routes to the same unique
answer for a nonsingular system.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import scipy.sparse as sp

from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.handler import ConstraintHandler
from nanofem.core.model import DomainDefinition, Model
from nanofem.elements.factory import build_elements
from nanofem.geometry.plane import PlaneGeometry
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.numerics.assembly.assembler import Assembler
from nanofem.numerics.assembly.contributions import OperatorRole
from nanofem.numerics.assembly.sparsity import SparsityPattern
from nanofem.numerics.assembly.system import GlobalSystem, ReducedSystem
from nanofem.numerics.linalg.linear import ConjugateGradientSolver, SparseDirectSolver
from nanofem.physics.elasticity.isotropic import IsotropicElasticity
from nanofem.physics.elasticity.plane import PlaneStressConstitutive
from nanofem.utils.exceptions import InputValidationError, SingularMatrixError


def _random_spd_system(n: int, seed: int) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """A well-conditioned SPD system with a known exact solution."""
    rng = np.random.default_rng(seed)
    a_dense = rng.standard_normal((n, n))
    a_spd = a_dense @ a_dense.T + n * np.eye(n)
    x_true = rng.standard_normal(n)
    matrix = sp.csr_matrix(a_spd)
    rhs = matrix @ x_true
    return matrix, rhs, x_true


def test_cg_matches_lu_and_the_known_solution_on_a_random_spd_system() -> None:
    matrix, rhs, x_true = _random_spd_system(n=40, seed=0)
    x_lu = SparseDirectSolver().solve(matrix, rhs)
    solver = ConjugateGradientSolver(rtol=1e-10)
    x_cg = solver.solve(matrix, rhs)
    np.testing.assert_allclose(x_cg, x_true, rtol=1e-6)
    np.testing.assert_allclose(x_cg, x_lu, rtol=1e-6)


def test_cg_records_convergence_statistics() -> None:
    matrix, rhs, _ = _random_spd_system(n=40, seed=1)
    solver = ConjugateGradientSolver(rtol=1e-10)
    solver.solve(matrix, rhs)
    assert solver.iterations > 0
    assert len(solver.residual_history) == solver.iterations
    final_residual = solver.residual_history[-1]
    assert final_residual <= 1e-10 * np.linalg.norm(rhs)
    # Residual is not required to decrease monotonically every step (CG is not a descent
    # method on the residual itself), but the last value must be far below the first.
    assert solver.residual_history[-1] < solver.residual_history[0]


def _t3_plate_reduced_system() -> ReducedSystem:
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    block = CellBlock("tri3", np.array([[0, 1, 2], [0, 2, 3]]), region="plate")
    regions = (
        Region("left_edge", 0, (0, 3)),
        Region("origin", 0, (0,)),
        Region("right_edge", 0, (1, 2)),
    )
    mesh = Mesh(coords, (block,), regions)
    model = Model(mesh)
    model.add_material(Material("steel", E=200e9, nu=0.3))
    model.add_section("plane", PlaneGeometry(thickness=0.01))
    model.add_theory("t", IsotropicElasticity(dim=2))
    model.add_constitutive("c", PlaneStressConstitutive())
    model.add_domain(
        DomainDefinition("d", "plate", "t", "steel", geometry="plane", constitutive="c")
    )
    model.add_dirichlet(DirichletBC("left_edge", "u", ("x",), 0.0))
    model.add_dirichlet(DirichletBC("origin", "u", ("y",), 0.0))
    dof_handler = model.build_dof_handler()
    elements = build_elements(model, dof_handler)
    pattern = SparsityPattern.from_providers(elements, OperatorRole.STIFFNESS, dof_handler.num_dofs)
    assembler = Assembler(pattern)
    stiffness = assembler.assemble(elements, OperatorRole.STIFFNESS)
    dirichlet_bcs = cast(tuple[DirichletBC, ...], model.dirichlet_bcs)
    partition = ConstraintHandler(mesh, dof_handler, dirichlet_bcs).partition()
    force = np.zeros(dof_handler.num_dofs)
    for node in (1, 2):
        force[dof_handler.global_dof(node, "u", "x")] = 5000.0
    system = GlobalSystem(
        dof_handler.num_dofs, {OperatorRole.STIFFNESS: stiffness}, {OperatorRole.FORCE: force}
    )
    return ReducedSystem.from_global(
        system, partition.free_dofs, partition.constrained_dofs, partition.prescribed_values
    )


def test_cg_matches_lu_on_a_real_fem_stiffness_system() -> None:
    reduced = _t3_plate_reduced_system()
    u_lu = SparseDirectSolver().solve(reduced.k_ff, reduced.f_f)
    u_cg = ConjugateGradientSolver(rtol=1e-10).solve(reduced.k_ff, reduced.f_f)
    np.testing.assert_allclose(u_cg, u_lu, rtol=1e-8, atol=1e-9 * np.abs(u_lu).max())


def test_cg_rejects_a_non_positive_diagonal() -> None:
    matrix = sp.csr_matrix(np.array([[1.0, 2.0], [2.0, -1.0]]))
    with pytest.raises(SingularMatrixError, match="positive diagonal"):
        ConjugateGradientSolver().solve(matrix, np.array([1.0, 1.0]))


def test_cg_raises_when_it_does_not_converge_in_time() -> None:
    matrix, rhs, _ = _random_spd_system(n=40, seed=2)
    solver = ConjugateGradientSolver(rtol=1e-14, max_iterations=1)
    with pytest.raises(SingularMatrixError, match="did not converge"):
        solver.solve(matrix, rhs)


def test_cg_constructor_validates_parameters() -> None:
    with pytest.raises(InputValidationError):
        ConjugateGradientSolver(rtol=0.0)
    with pytest.raises(SingularMatrixError):
        ConjugateGradientSolver(max_iterations=0)
