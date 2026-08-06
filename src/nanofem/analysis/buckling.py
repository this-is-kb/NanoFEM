"""Linear buckling metadata: the chained flow (SDS 2.18)."""

from __future__ import annotations

from dataclasses import dataclass

from nanofem.analysis.base import AnalysisBase
from nanofem.core.model import Model
from nanofem.numerics.assembly.contributions import OperatorRole
from nanofem.utils.exceptions import ModelError
from nanofem.utils.validation import require_identifier, require_positive_int


@dataclass(frozen=True)
class BucklingOptions:
    """Pre-load case for the static pre-stress solve, and mode count."""

    preload_case: str
    num_modes: int = 4

    def __post_init__(self) -> None:
        require_identifier(self.preload_case, "preload_case")
        require_positive_int(self.num_modes, "num_modes")


class LinearBucklingAnalysis(AnalysisBase):
    """Static pre-stress -> K_g(sigma_0) -> eigenpair (K, -K_g) (phase 3)."""

    def __init__(self, model: Model, options: BucklingOptions) -> None:
        super().__init__(model)
        self.options = options

    def required_roles(self) -> tuple[OperatorRole, ...]:
        """Buckling recipe roles."""
        return (
            OperatorRole.STIFFNESS,
            OperatorRole.GEOMETRIC_STIFFNESS,
            OperatorRole.FORCE,
        )

    def validate(self) -> None:
        """Base checks plus pre-load case resolution."""
        super().validate()
        if self.options.preload_case not in self.model.load_case_names:
            raise ModelError(
                f"buckling pre-load case '{self.options.preload_case}' is not registered "
                f"(registered: {list(self.model.load_case_names)})"
            )
