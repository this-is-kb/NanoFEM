"""ModelState: structure-of-arrays trial/committed banks (SDS Section 7).

Constitutive code reads committed and writes trial, never the reverse;
commit and revert are whole-bank copies (data movement, not mathematics).
Allocation is layout-driven: empty layouts allocate nothing, by contract.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nanofem.state.layout import StateLayout
from nanofem.utils.exceptions import StateError
from nanofem.utils.validation import require_positive_int


class ModelState:
    """Global state banks with the commit/revert lifecycle."""

    def __init__(self) -> None:
        self._layout: StateLayout | None = None
        self._trial: dict[str, NDArray[np.float64]] = {}
        self._committed: dict[str, NDArray[np.float64]] = {}

    def allocate(self, layout: StateLayout, n_cells: int, n_qp: int) -> None:
        """Allocate (n_cells, n_qp, n_comp) float64 arrays per declared variable."""
        if self._layout is not None:
            raise StateError("state already allocated; build a new ModelState instead")
        require_positive_int(n_cells, "n_cells")
        require_positive_int(n_qp, "n_qp")
        self._layout = layout
        for name, ncomp in layout.variables:
            self._trial[name] = np.zeros((n_cells, n_qp, ncomp), dtype=np.float64)
            self._committed[name] = np.zeros((n_cells, n_qp, ncomp), dtype=np.float64)

    @property
    def is_allocated(self) -> bool:
        """Whether allocate() has run."""
        return self._layout is not None

    @property
    def variables(self) -> tuple[str, ...]:
        """Allocated variable names."""
        return tuple(self._trial)

    def _bank(self, bank: dict[str, NDArray[np.float64]], name: str) -> NDArray[np.float64]:
        try:
            return bank[name]
        except KeyError:
            raise StateError(
                f"unknown or unallocated state variable '{name}'; allocated: {list(self._trial)}"
            ) from None

    def view(self, name: str) -> NDArray[np.float64]:
        """Zero-copy view of one variable's *trial* bank (constitutive writes here)."""
        return self._bank(self._trial, name)

    def committed_view(self, name: str) -> NDArray[np.float64]:
        """Zero-copy view of one variable's *committed* bank (reads come from here)."""
        return self._bank(self._committed, name)

    def commit(self) -> None:
        """Trial becomes committed (converged solve / accepted step)."""
        for name, trial in self._trial.items():
            np.copyto(self._committed[name], trial)

    def revert(self) -> None:
        """Discard trial (diverged iteration / rejected step)."""
        for name, committed in self._committed.items():
            np.copyto(self._trial[name], committed)
