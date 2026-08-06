"""Generalized symmetric eigensolvers (SDS 2.16).

Modal: K phi = omega^2 M phi, mass-normalized. Buckling: the pair (K, -K_g),
smallest positive load factors reported, negative ones flagged (load-reversal
buckling), never discarded silently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EigenSolver(ABC):
    """Generalized symmetric eigenproblem interface."""

    @abstractmethod
    def solve_pair(self, a: object, b: object, count: int) -> object:
        """Return (values, vectors) with declared normalization."""


class ShiftInvertEigensolver(EigenSolver):
    """Smallest eigenpairs via shift-invert; handles semidefinite M by construction."""


class BucklingEigensolver(EigenSolver):
    """Critical load factors from (K, -K_g); requires a pre-stress state."""
