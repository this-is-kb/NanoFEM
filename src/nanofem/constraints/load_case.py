"""Named load collections with scale factors and time functions (SDS Section 11)."""

from __future__ import annotations

from dataclasses import dataclass

from nanofem.constraints.time_functions import TimeFunction
from nanofem.utils.validation import require_finite, require_identifier


@dataclass(frozen=True)
class LoadEntry:
    """One load with its scale factor and optional time function."""

    load: object
    factor: float = 1.0
    time_function: TimeFunction | None = None

    def __post_init__(self) -> None:
        require_finite(self.factor, "load factor")


class LoadCase:
    """f(t) = sum_i factor_i * lambda_i(t) * f_i - assembled in phase 2."""

    def __init__(self, name: str) -> None:
        require_identifier(name, "load case name")
        self._name = name
        self._entries: list[LoadEntry] = []

    @property
    def name(self) -> str:
        """Registry key inside a Model."""
        return self._name

    def add(
        self, load: object, factor: float = 1.0, time_function: TimeFunction | None = None
    ) -> None:
        """Append a load entry."""
        self._entries.append(LoadEntry(load, factor, time_function))

    @property
    def entries(self) -> tuple[LoadEntry, ...]:
        """Entries in insertion order."""
        return tuple(self._entries)
