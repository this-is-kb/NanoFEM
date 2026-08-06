"""Global and reduced system containers (SDS 2.13, 2.14).

Reduction is elimination-based (ADR-003): K_ff u_f = f_f - K_fc u_c keeps
symmetric definiteness so CG and shift-invert stay available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from nanofem.numerics.assembly.contributions import OperatorRole


class GlobalSystem:
    """Assembled operators and vectors for one model state, keyed by role.

    Takes ``num_dofs`` rather than a ``DofHandler``: ``numerics`` is a leaf
    package (rule R1) and must not import ``core``, and a bare count is all
    this container needs.
    """

    def __init__(
        self,
        num_dofs: int,
        operators: dict[OperatorRole, csr_matrix],
        vectors: dict[OperatorRole, NDArray[np.float64]],
    ) -> None:
        self.num_dofs = num_dofs
        self._operators = dict(operators)
        self._vectors = dict(vectors)

    def operator(self, role: OperatorRole) -> csr_matrix:
        """Return the assembled operator for ``role``."""
        return self._operators[role]

    def vector(self, role: OperatorRole) -> NDArray[np.float64]:
        """Return the assembled vector for ``role``."""
        return self._vectors[role]


@dataclass(frozen=True)
class ReducedSystem:
    """Constraint-partitioned system with the recovery map back to full vectors."""

    free_dofs: NDArray[np.int64]
    constrained_dofs: NDArray[np.int64]
    prescribed_values: NDArray[np.float64]
    k_ff: csr_matrix
    k_fc: csr_matrix
    k_cf: csr_matrix
    k_cc: csr_matrix
    f_f: NDArray[np.float64]
    f_c: NDArray[np.float64]
    num_dofs: int

    @classmethod
    def from_global(
        cls,
        system: GlobalSystem,
        free_dofs: NDArray[np.int64],
        constrained_dofs: NDArray[np.int64],
        prescribed_values: NDArray[np.float64],
        *,
        operator_role: OperatorRole = OperatorRole.STIFFNESS,
        vector_role: OperatorRole = OperatorRole.FORCE,
    ) -> ReducedSystem:
        """Partition one operator/vector pair by role into free/constrained blocks (SDS 2.14).

        ``f_f`` is stored already reduced (``f_f - K_fc u_c``), ready for the
        linear solver; ``f_c`` is stored raw, as :meth:`reactions` needs it.
        """
        k = system.operator(operator_role)
        f = system.vector(vector_role)
        k_ff = k[np.ix_(free_dofs, free_dofs)].tocsr()
        k_fc = k[np.ix_(free_dofs, constrained_dofs)].tocsr()
        k_cf = k[np.ix_(constrained_dofs, free_dofs)].tocsr()
        k_cc = k[np.ix_(constrained_dofs, constrained_dofs)].tocsr()
        f_c = f[constrained_dofs]
        f_f = f[free_dofs] - k_fc @ prescribed_values
        return cls(
            free_dofs=free_dofs,
            constrained_dofs=constrained_dofs,
            prescribed_values=prescribed_values,
            k_ff=k_ff,
            k_fc=k_fc,
            k_cf=k_cf,
            k_cc=k_cc,
            f_f=f_f,
            f_c=f_c,
            num_dofs=system.num_dofs,
        )

    def recover(self, u_f: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reinsert constrained values into the full solution vector."""
        full = np.zeros(self.num_dofs, dtype=np.float64)
        full[self.free_dofs] = u_f
        full[self.constrained_dofs] = self.prescribed_values
        return full

    def reactions(self, u_f: NDArray[np.float64]) -> NDArray[np.float64]:
        """``R = K_cf u_f + K_cc u_c - f_c`` (SDS 2.14)."""
        result = self.k_cf @ u_f + self.k_cc @ self.prescribed_values - self.f_c
        return np.asarray(result, dtype=np.float64)
