"""Declared placeholder for the nonlocal integral operator (SDS Section 8).

Carries ``PROVISIONAL_METADATA`` and a ``BLOCKED_BY`` string naming what it
structurally needs, matching the pattern established in
``numerics/quadrature/future.py`` and ``numerics/mapping/future.py``. Not
blocked by scheduling: ``kernels/`` is itself a phase-0 skeleton (no concrete
``Kernel`` yet) and ``numerics/search/`` is a one-method stub, and both sit
*above* ``numerics`` in the import-linter layer contract for ``kernels`` -
this operator cannot be built today from anything that exists yet, which is
the same bar every other real recipe in this package had to clear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from nanofem.numerics.operators.base import DiscreteOperator


@dataclass(frozen=True, eq=False, repr=False, init=False)
class _FutureOperator(DiscreteOperator):
    """Shared placeholder base: satisfies the interface but refuses construction."""

    PROVISIONAL_METADATA: ClassVar[dict[str, Any]] = {}
    BLOCKED_BY: ClassVar[str] = ""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Refuse construction, naming what the operator still needs."""
        raise NotImplementedError(
            f"{type(self).__name__} is a declared placeholder: {self.BLOCKED_BY}"
        )

    def name(self) -> str:  # pragma: no cover - construction raises first
        """Not available for a placeholder."""
        raise NotImplementedError

    def required_derivative_order(self) -> int:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    def verify(self) -> None:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    def is_valid(self) -> bool:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError


@dataclass(frozen=True, eq=False, repr=False, init=False)
class NonlocalIntegralOperator(_FutureOperator):
    """Placeholder for the two-phase Eringen integral pairwise operator (SDS Section 4.3, 8).

    Intended: the pair-block recipe
    ``K_NL^(e,e') = integral_e integral_e' B^T(x) alpha(|x-x'|) C B(x') dOmega' dOmega``,
    including self-pairs. This needs a kernel ``alpha`` (evaluation, support
    radius, normalization policy - ``kernels.Kernel``, phase-0 skeleton only)
    and horizon pairs from ``numerics.search.NeighborSearch`` (a one-method
    stub). Both are structural prerequisites, not scheduling ones: this
    operator's own contract (SDS Section 8) names them as its inputs.
    """

    PROVISIONAL_METADATA: ClassVar[dict[str, Any]] = {
        "name": "nonlocal_integral",
        "locality": "PAIRWISE",
        "derivative_order": 1,
        "needs": [
            "kernels.Kernel: evaluate(r), support_radius(), normalization policy",
            "numerics.search.NeighborSearch: horizon pairs (e, e')",
        ],
    }
    BLOCKED_BY = (
        "it needs an attenuation kernel from kernels/ (still a phase-0 skeleton with no "
        "concrete Kernel) and neighbor pairs from numerics/search/ (a one-method stub); "
        "both structurally precede this operator rather than merely following it on a "
        "schedule, and kernels/ sits above numerics in the import-linter layer contract"
    )
