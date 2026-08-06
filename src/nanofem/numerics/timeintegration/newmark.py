"""Time integrators (SDS 2.17).

Newmark: unconditional stability iff 2 beta >= gamma >= 1/2; default average
acceleration (beta = 1/4, gamma = 1/2) conserves energy for undamped linear
systems - a property the validation suite exploits. Conditionally stable
parameter choices proceed with a stability-estimate warning, never silently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TimeIntegrator(ABC):
    """Advance (u, v, a) one step for M a + C v + K u = f(t)."""

    @abstractmethod
    def advance(self, state: object, force: object, dt: float) -> object:
        """Return the advanced (u, v, a) plus step diagnostics."""


class NewmarkBeta(TimeIntegrator):
    """Implicit Newmark family; effective operator reuses the cached factorization."""
