"""Per-quadrature-point view over ModelState banks (SDS Section 7)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nanofem.state.model_state import ModelState


class QuadraturePointState:
    """Convenience accessor for one (cell, qp): indexing, not computation."""

    def __init__(self, state: ModelState, cell: int, qp: int) -> None:
        self._state = state
        self._cell = cell
        self._qp = qp

    def trial(self, name: str) -> NDArray[np.float64]:
        """Trial values at this point, shape (n_comp,)."""
        values: NDArray[np.float64] = self._state.view(name)[self._cell, self._qp]
        return values

    def committed(self, name: str) -> NDArray[np.float64]:
        """Committed values at this point, shape (n_comp,)."""
        values: NDArray[np.float64] = self._state.committed_view(name)[self._cell, self._qp]
        return values
