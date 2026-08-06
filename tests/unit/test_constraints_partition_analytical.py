"""ConstraintHandler.partition() and NodalLoadProvider: a 2-node 1-D bar mesh (SDS 2.14)."""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.handler import ConstraintHandler
from nanofem.constraints.loads import NodalLoad, NodalLoadProvider
from nanofem.core.dof_handler import DofHandler
from nanofem.core.fields import displacement
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.numerics.assembly.contributions import OperatorRole
from nanofem.utils.exceptions import ConstraintConflictError, MeshError


def _bar_mesh() -> Mesh:
    coordinates = np.array([[0.0], [1.0]])
    block = CellBlock("line2", np.array([[0, 1]]), region="bar")
    regions = (Region("fixed", 0, (0,)), Region("tip", 0, (1,)))
    return Mesh(coordinates, (block,), regions)


def _dof_handler(mesh: Mesh) -> DofHandler:
    return DofHandler.generate(mesh, (displacement(1),))


def test_partition_splits_free_and_constrained() -> None:
    mesh = _bar_mesh()
    dof_handler = _dof_handler(mesh)
    bcs = (DirichletBC("fixed", "u", ("x",), 0.0),)
    partition = ConstraintHandler(mesh, dof_handler, bcs).partition()
    np.testing.assert_array_equal(partition.constrained_dofs, [0])
    np.testing.assert_array_equal(partition.free_dofs, [1])
    np.testing.assert_array_equal(partition.prescribed_values, [0.0])
    assert partition.num_dofs == 2


def test_constrained_dofs_are_sorted_regardless_of_bc_order() -> None:
    mesh = _bar_mesh()
    dof_handler = _dof_handler(mesh)
    bcs = (
        DirichletBC("tip", "u", ("x",), 2.0),
        DirichletBC("fixed", "u", ("x",), 0.0),
    )
    partition = ConstraintHandler(mesh, dof_handler, bcs).partition()
    np.testing.assert_array_equal(partition.constrained_dofs, [0, 1])
    np.testing.assert_array_equal(partition.prescribed_values, [0.0, 2.0])
    assert partition.free_dofs.size == 0


def test_conflicting_dirichlet_values_raise() -> None:
    mesh = _bar_mesh()
    dof_handler = _dof_handler(mesh)
    bcs = (
        DirichletBC("fixed", "u", ("x",), 0.0),
        DirichletBC("fixed", "u", ("x",), 1.0),
    )
    with pytest.raises(ConstraintConflictError):
        ConstraintHandler(mesh, dof_handler, bcs).partition()


def test_unknown_region_raises_constraint_conflict_error() -> None:
    mesh = _bar_mesh()
    dof_handler = _dof_handler(mesh)
    bcs = (DirichletBC("nowhere", "u", ("x",), 0.0),)
    with pytest.raises(ConstraintConflictError):
        ConstraintHandler(mesh, dof_handler, bcs).partition()
    with pytest.raises(MeshError):
        mesh.nodes_in_region("nowhere")


def test_nodal_load_provider_yields_force_contribution_at_tip() -> None:
    mesh = _bar_mesh()
    dof_handler = _dof_handler(mesh)
    load = NodalLoad("tip", "u", np.array([5.0]))
    provider = NodalLoadProvider(load, mesh, dof_handler, field_components=("x",))
    contributions = list(provider.contributions(OperatorRole.FORCE))
    assert len(contributions) == 1
    (contribution,) = contributions
    assert contribution.col_dofs is None
    np.testing.assert_array_equal(contribution.row_dofs, [1])
    np.testing.assert_allclose(contribution.block, [5.0])


def test_nodal_load_provider_ignores_other_roles() -> None:
    mesh = _bar_mesh()
    dof_handler = _dof_handler(mesh)
    load = NodalLoad("tip", "u", np.array([5.0]))
    provider = NodalLoadProvider(load, mesh, dof_handler, field_components=("x",))
    assert list(provider.contributions(OperatorRole.STIFFNESS)) == []


def test_nodal_load_provider_applies_factor() -> None:
    mesh = _bar_mesh()
    dof_handler = _dof_handler(mesh)
    load = NodalLoad("tip", "u", np.array([5.0]))
    provider = NodalLoadProvider(load, mesh, dof_handler, field_components=("x",), factor=-2.0)
    (contribution,) = list(provider.contributions(OperatorRole.FORCE))
    np.testing.assert_allclose(contribution.block, [-10.0])
