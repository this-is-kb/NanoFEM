"""Discrete operator vocabulary and the Continuity derivation rule (SDS Section 8, ADR-015).

A discrete operator is a *stateless recipe*: tabulated interpolation data at
quadrature points -> the matrix mapping element DOFs to the operator's
pointwise value. Each operator declares the highest interpolation derivative
it consumes; a theory's continuity requirement is DERIVED as the maximum over
its operators - which is how continuity errors become structurally impossible.

Continuity lives here (not in physics, not in interpolation) because it is
the shared vocabulary of both sides: operators declare what they need,
interpolation families declare what they provide, and rule R2 stays intact
(physics may import numerics.operators, never numerics.interpolation).

Phase 1 ships the vocabulary (the catalog as registry-style name strings,
matching Theory.operators_used) and the derivation table as data; the
recipes themselves are phase-2 numerics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, unique


@unique
class Continuity(Enum):
    """Inter-element continuity classes of interpolation families (SDS Section 4)."""

    C0 = 0
    C1 = 1


#: Highest interpolation derivative each SDS Section 8 operator consumes.
OPERATOR_DERIVATIVE_ORDER: dict[str, int] = {
    "gradient": 1,
    "symmetric_gradient": 1,
    "divergence": 1,
    "curl": 1,
    "laplacian": 1,
    "helmholtz": 1,
    "nonlocal_integral": 1,
    "surface_gradient": 1,
    "second_gradient": 2,
    "voigt_map": 0,
    "transformation": 0,
}

#: The SDS Section 8 catalog as a closed set of registry-style names.
OPERATOR_CATALOG: frozenset[str] = frozenset(OPERATOR_DERIVATIVE_ORDER)


def derived_continuity(operators: tuple[str, ...]) -> Continuity:
    """Derive the continuity requirement from composed operators (SDS Section 8 rule).

    A theory composing any operator that consumes second derivatives is C1
    whether its author remembered or not. Unknown names are the caller's
    error and raise ``KeyError`` deliberately loudly.
    """
    highest = max((OPERATOR_DERIVATIVE_ORDER[op] for op in operators), default=0)
    return Continuity.C1 if highest >= 2 else Continuity.C0


class DiscreteOperator(ABC):
    """Stateless recipe from tabulated data to operator matrices (pure, mechanics-free)."""

    @abstractmethod
    def name(self) -> str:
        """Return this recipe's OPERATOR_CATALOG entry."""

    @abstractmethod
    def required_derivative_order(self) -> int:
        """Return the highest interpolation derivative consumed (Hessian -> 2 -> C1)."""

    @abstractmethod
    def verify(self) -> None:
        """Self-check this recipe against fixed synthetic data; raise on the first failure."""

    @abstractmethod
    def is_valid(self) -> bool:
        """Return ``True`` if :meth:`verify` passes, ``False`` otherwise."""
