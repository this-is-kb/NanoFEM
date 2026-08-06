"""Nondimensionalization scales (SDS C-6): property storage only in phase 1.

Scale application/inversion maps are phase-2 work; storing validated
characteristic scales is the object-model half.
"""

from __future__ import annotations

from nanofem.utils.validation import require_positive


class Nondimensionalizer:
    """Characteristic length/force scale manager (stress scale = force/length^2)."""

    def __init__(self, length: float, force: float) -> None:
        require_positive(length, "length scale")
        require_positive(force, "force scale")
        self._length = length
        self._force = force

    @property
    def length(self) -> float:
        """Characteristic length."""
        return self._length

    @property
    def force(self) -> float:
        """Characteristic force."""
        return self._force

    def to_model(self, value: float, unit: str) -> float:
        """Scale a physical value into model units. TODO(phase-2)."""
        raise NotImplementedError("TODO(phase-2): scale application")

    def from_model(self, value: float, unit: str) -> float:
        """Invert scaling at results. TODO(phase-2)."""
        raise NotImplementedError("TODO(phase-2): scale inversion")
