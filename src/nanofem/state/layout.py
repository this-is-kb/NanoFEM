"""State layout declarations: which variables a constitutive model needs (SDS Section 7)."""

from __future__ import annotations

from dataclasses import dataclass

from nanofem.utils.exceptions import StateError


@dataclass(frozen=True)
class StateLayout:
    """Ordered (name, n_components) variable declarations; empty for elastic laws."""

    variables: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        names = [n for n, _ in self.variables]
        if len(set(names)) != len(names):
            raise StateError(f"duplicate state variable names: {names}")
        for name, ncomp in self.variables:
            if not name or ncomp < 1:
                raise StateError(f"invalid state variable ({name!r}, {ncomp})")

    @property
    def names(self) -> tuple[str, ...]:
        """Declared variable names in order."""
        return tuple(n for n, _ in self.variables)

    @property
    def total_components(self) -> int:
        """Sum of components across variables."""
        return sum(c for _, c in self.variables)

    def is_empty(self) -> bool:
        """True when no state is declared (elastic laws pay zero memory)."""
        return not self.variables
