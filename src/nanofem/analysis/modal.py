"""Modal analysis metadata: K phi = omega^2 M phi (SDS 2.16, 2.18)."""

from __future__ import annotations

from dataclasses import dataclass

from nanofem.analysis.base import AnalysisBase
from nanofem.core.model import Model
from nanofem.numerics.assembly.contributions import OperatorRole
from nanofem.utils.validation import require_finite, require_positive_int


@dataclass(frozen=True)
class ModalOptions:
    """Mode count and shift for the shift-invert eigensolver."""

    num_modes: int = 6
    shift: float = 0.0

    def __post_init__(self) -> None:
        require_positive_int(self.num_modes, "num_modes")
        require_finite(self.shift, "shift")


class ModalAnalysis(AnalysisBase):
    """Assemble STIFFNESS + MASS, reduce, eigensolve, mass-normalize (phase 3)."""

    def __init__(self, model: Model, options: ModalOptions | None = None) -> None:
        super().__init__(model)
        self.options = options if options is not None else ModalOptions()

    def required_roles(self) -> tuple[OperatorRole, ...]:
        """Modal recipe roles."""
        return (OperatorRole.STIFFNESS, OperatorRole.MASS)
