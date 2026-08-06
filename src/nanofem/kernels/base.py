"""Attenuation kernel contract: alpha(r; length scale) (ADR-009, SDS Section 8).

Kernels are pure functions of distance and parameters - no mesh, no elements,
no materials - which is why user kernels are safe to accept from third
parties. Kernels are dimension-aware (the Helmholtz Green's function is the
bi-exponential in 1-D but Bessel-K0 in 2-D).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Kernel(ABC):
    """Radial attenuation function with declared support and normalization."""

    @abstractmethod
    def evaluate(self, distances: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return alpha(r) on a batch of distances, same shape."""

    @abstractmethod
    def support_radius(self) -> float:
        """Return the horizon that numerics.search queries (SDS Section 10, PAIR)."""
