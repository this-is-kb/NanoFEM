"""Bounded committed-bank snapshots; checkpoint/restart placeholders (SDS Section 7, C-5)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nanofem.state.model_state import ModelState
from nanofem.utils.exceptions import StateError
from nanofem.utils.validation import require_positive_int


class StateHistory:
    """Ring of past committed snapshots for multi-step schemes."""

    def __init__(self, state: ModelState, max_depth: int = 3) -> None:
        require_positive_int(max_depth, "max_depth")
        self._state = state
        self._max_depth = max_depth
        self._snapshots: list[dict[str, NDArray[np.float64]]] = []

    @property
    def depth(self) -> int:
        """Snapshots currently held."""
        return len(self._snapshots)

    def snapshot(self) -> None:
        """Record a copy of the committed bank (drops the oldest beyond max_depth)."""
        snap = {n: self._state.committed_view(n).copy() for n in self._state.variables}
        self._snapshots.append(snap)
        if len(self._snapshots) > self._max_depth:
            self._snapshots.pop(0)

    def restore(self) -> None:
        """Restore the most recent snapshot into both banks."""
        if not self._snapshots:
            raise StateError("no snapshots to restore")
        snap = self._snapshots.pop()
        for name, values in snap.items():
            np.copyto(self._state.committed_view(name), values)
            np.copyto(self._state.view(name), values)

    def checkpoint(self, path: str) -> None:
        """TODO(future): fingerprinted checkpoint via io/ (SDS Section 7)."""
        raise NotImplementedError("TODO(future): checkpointing via io/")

    def restore_checkpoint(self, path: str) -> None:
        """TODO(future): fingerprint-verified restart."""
        raise NotImplementedError("TODO(future): restart path")
