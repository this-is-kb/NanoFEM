"""Transient analysis metadata: time grid + integrator choice (SDS 2.17, 2.18)."""

from __future__ import annotations

from dataclasses import dataclass

from nanofem.analysis.base import AnalysisBase
from nanofem.core.model import Model
from nanofem.numerics.assembly.contributions import OperatorRole
from nanofem.utils.exceptions import InputValidationError
from nanofem.utils.validation import require_identifier, require_positive


@dataclass(frozen=True)
class TransientOptions:
    """Time grid, integrator key, and load cases to drive."""

    t_end: float
    dt: float
    integrator: str = "newmark_beta"
    load_cases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_positive(self.t_end, "t_end")
        require_positive(self.dt, "dt")
        require_identifier(self.integrator, "integrator")
        if self.dt > self.t_end:
            raise InputValidationError(f"dt {self.dt} exceeds t_end {self.t_end}")


class TransientAnalysis(AnalysisBase):
    """Assemble K, M (C optional); step the integrator; stream states (phase 3)."""

    def __init__(self, model: Model, options: TransientOptions) -> None:
        super().__init__(model)
        self.options = options

    def required_roles(self) -> tuple[OperatorRole, ...]:
        """Transient recipe roles (DAMPING is optional, added by Rayleigh settings)."""
        return (OperatorRole.STIFFNESS, OperatorRole.MASS, OperatorRole.FORCE)
