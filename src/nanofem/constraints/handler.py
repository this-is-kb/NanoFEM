"""DOF partitioning and elimination reduction maps (SDS 2.14, ADR-003).

Produces the free/constrained partition and the MPC transformation T with
u = T u_tilde + g0; reduced operators are T^T A T. Elimination keeps
symmetric definiteness so CG and shift-invert stay available.

MPC transformation is out of scope this phase (no model here declares one);
only Dirichlet partitioning is real.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.constraints.dirichlet import DirichletBC
from nanofem.core.dof_handler import DofHandler
from nanofem.mesh.mesh import Mesh
from nanofem.utils.exceptions import ConstraintConflictError, MeshError


@dataclass(frozen=True)
class DofPartition:
    """Free/constrained DOF split with prescribed values, aligned index-for-index."""

    free_dofs: NDArray[np.int64]
    constrained_dofs: NDArray[np.int64]
    prescribed_values: NDArray[np.float64]
    num_dofs: int


class ConstraintHandler:
    """Partition DOFs; build reduction and recovery maps consumed by assembly/solvers."""

    def __init__(
        self,
        mesh: Mesh,
        dof_handler: DofHandler,
        dirichlet_bcs: tuple[DirichletBC, ...],
    ) -> None:
        self._mesh = mesh
        self._dof_handler = dof_handler
        self._dirichlet_bcs = dirichlet_bcs

    def partition(self) -> DofPartition:
        """Return the free/constrained partition (deterministic, SDS C-5).

        Constrained DOFs are sorted ascending so partition order is a pure
        function of the model, independent of ``dirichlet_bcs`` iteration order.
        """
        prescribed: dict[int, float] = {}
        for bc in self._dirichlet_bcs:
            try:
                node_ids = self._mesh.nodes_in_region(bc.region)
            except MeshError as exc:
                raise ConstraintConflictError(str(exc)) from exc
            for node_id in node_ids:
                for component in bc.components:
                    dof = self._dof_handler.global_dof(node_id, bc.field, component)
                    if dof in prescribed and prescribed[dof] != bc.value:
                        raise ConstraintConflictError(
                            f"DOF {dof} (node {node_id}, field '{bc.field}', component "
                            f"'{component}') has conflicting Dirichlet values "
                            f"{prescribed[dof]} and {bc.value}"
                        )
                    prescribed[dof] = bc.value

        num_dofs = self._dof_handler.num_dofs
        constrained_dofs = np.array(sorted(prescribed), dtype=np.int64)
        prescribed_values = np.array(
            [prescribed[d] for d in constrained_dofs.tolist()], dtype=np.float64
        )
        free_mask = np.ones(num_dofs, dtype=bool)
        free_mask[constrained_dofs] = False
        free_dofs = np.flatnonzero(free_mask).astype(np.int64)
        return DofPartition(
            free_dofs=free_dofs,
            constrained_dofs=constrained_dofs,
            prescribed_values=prescribed_values,
            num_dofs=num_dofs,
        )
