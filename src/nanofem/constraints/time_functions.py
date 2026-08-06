"""Amplitude functions f(t) for load scaling (SDS 2.17): parameter storage.

ConstantTF returns its stored value (storage retrieval); ramp and harmonic
*evaluation* is arithmetic reserved for phase 3 - parameters are stored and
validated now so load cases can reference them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from nanofem.utils.validation import require_finite, require_positive


class TimeFunction(ABC):
    """Scalar amplitude of time."""

    @abstractmethod
    def __call__(self, t: float) -> float:
        """Return the amplitude at time t."""


@dataclass(frozen=True)
class ConstantTF(TimeFunction):
    """f(t) = value."""

    value: float = 1.0

    def __post_init__(self) -> None:
        require_finite(self.value, "ConstantTF value")

    def __call__(self, t: float) -> float:
        """Stored constant amplitude."""
        return self.value


@dataclass(frozen=True)
class RampTF(TimeFunction):
    """Linear ramp to a plateau over rise_time; evaluation lands in phase 3."""

    rise_time: float
    plateau: float = 1.0

    def __post_init__(self) -> None:
        require_positive(self.rise_time, "RampTF rise_time")
        require_finite(self.plateau, "RampTF plateau")

    def __call__(self, t: float) -> float:
        """TODO(phase-3): ramp evaluation."""
        raise NotImplementedError("TODO(phase-3): ramp evaluation")


@dataclass(frozen=True)
class HarmonicTF(TimeFunction):
    """f(t) = amplitude * sin(omega t + phase); evaluation lands in phase 3."""

    amplitude: float
    omega: float
    phase: float = 0.0

    def __post_init__(self) -> None:
        require_finite(self.amplitude, "HarmonicTF amplitude")
        require_positive(self.omega, "HarmonicTF omega")
        require_finite(self.phase, "HarmonicTF phase")

    def __call__(self, t: float) -> float:
        """TODO(phase-3): harmonic evaluation."""
        raise NotImplementedError("TODO(phase-3): harmonic evaluation")
