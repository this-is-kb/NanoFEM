"""FGM grading laws: validated parameter storage; evaluation is phase-2 (SDS Section 6)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.utils.validation import require_finite, require_positive


class GradingLaw(ABC):
    """Position -> property value; composes with every constitutive model."""

    @abstractmethod
    def evaluate(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return graded values at points, shape (n_qp,)."""


@dataclass(frozen=True)
class PowerLawGrading(GradingLaw):
    """P(z) = (P_top - P_bottom)(z/h + 1/2)^n + P_bottom; parameters stored, validated."""

    p_top: float
    p_bottom: float
    thickness: float
    exponent: float

    def __post_init__(self) -> None:
        require_finite(self.p_top, "p_top")
        require_finite(self.p_bottom, "p_bottom")
        require_positive(self.thickness, "thickness")
        require_finite(self.exponent, "exponent")

    def evaluate(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """TODO(phase-2): grading evaluation."""
        raise NotImplementedError("TODO(phase-2): power-law grading evaluation")


@dataclass(frozen=True)
class ExponentialGrading(GradingLaw):
    """Exponential through-thickness grading; parameters stored, validated."""

    p_reference: float
    rate: float

    def __post_init__(self) -> None:
        require_finite(self.p_reference, "p_reference")
        require_finite(self.rate, "rate")

    def evaluate(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """TODO(phase-2): grading evaluation."""
        raise NotImplementedError("TODO(phase-2): exponential grading evaluation")
