"""Multipoint constraints: u_slave = sum c_i u_master_i + offset (SDS 2.14).

The primitive periodic BCs and hanging-node adaptivity specialize; cycles
and slave-also-Dirichlet conflicts are ConstraintHandler business (phase 2).
"""

from __future__ import annotations

from dataclasses import dataclass

from nanofem.utils.exceptions import ConstraintConflictError
from nanofem.utils.validation import require_finite


@dataclass(frozen=True)
class MultiPointConstraint:
    """Linear relation among DOFs; each DOF is (node_id, field, component)."""

    slave: tuple[int, str, str]
    masters: tuple[tuple[int, str, str], ...]
    coefficients: tuple[float, ...]
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not self.masters:
            raise ConstraintConflictError("MPC must reference at least one master DOF")
        if len(self.masters) != len(self.coefficients):
            raise ConstraintConflictError(
                f"MPC has {len(self.masters)} masters but {len(self.coefficients)} coefficients"
            )
        if self.slave in self.masters:
            raise ConstraintConflictError(f"MPC slave {self.slave} appears among its masters")
        for c in self.coefficients:
            require_finite(c, "MPC coefficient")
        require_finite(self.offset, "MPC offset")
