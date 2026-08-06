"""Polynomial spaces as sets of monomial exponent tuples.

A polynomial space is described here *structurally*: by the multi-indices of
the monomials that span it. Nothing evaluates a polynomial at an arbitrary
point, and nothing forms a nodal (shape function) basis - the space is the
span, the shape functions are the basis dual to a set of degrees of freedom,
and constructing that dual basis is a later phase.

Spaces
------
- ``P_k`` (total degree): all exponent tuples with ``sum(alpha) <= k``.
  Dimension ``C(k + d, d)``. The natural space of simplices.
- ``Q_k`` (tensor product): all exponent tuples with ``max(alpha) <= k``.
  Dimension ``(k + 1)^d``. The natural space of tensor-product cells.

Three degree notions are kept distinct because they differ for ``Q_k`` and
are routinely conflated:

- **order** ``k``: the family's nominal order,
- **completeness degree**: the largest ``p`` with ``P_p`` contained in the
  space - what governs the approximation rate,
- **maximum total degree**: the highest total degree present (``k`` for
  ``P_k``, ``d * k`` for ``Q_k``).

Monomial ordering is graded lexicographic (by total degree, then
lexicographic by exponent tuple), which is deterministic and stable.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

from nanofem.numerics.interpolation.enums import PolynomialSpaceType
from nanofem.utils.exceptions import InputValidationError

#: Reference-coordinate symbol names by spatial dimension (SDS Section 8 notation).
_SYMBOLS: dict[int, tuple[str, ...]] = {
    1: ("xi",),
    2: ("xi", "eta"),
    3: ("xi", "eta", "zeta"),
}


def _graded_lex_key(exponents: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    """Sort key: total degree first, then lexicographic on the exponent tuple."""
    return (sum(exponents), exponents)


def _monomials_up_to_total_degree(order: int, n_variables: int) -> tuple[tuple[int, ...], ...]:
    """All exponent tuples with total degree <= ``order`` (the ``P_k`` set)."""
    found = [
        alpha
        for alpha in itertools.product(range(order + 1), repeat=n_variables)
        if sum(alpha) <= order
    ]
    return tuple(sorted(found, key=_graded_lex_key))


def _monomials_tensor_product(order: int, n_variables: int) -> tuple[tuple[int, ...], ...]:
    """All exponent tuples with per-variable degree <= ``order`` (the ``Q_k`` set)."""
    found = list(itertools.product(range(order + 1), repeat=n_variables))
    return tuple(sorted(found, key=_graded_lex_key))


def monomial_label(exponents: tuple[int, ...]) -> str:
    """Human-readable label of a monomial, e.g. ``(2, 1) -> "xi^2*eta"``."""
    symbols = _SYMBOLS[len(exponents)]
    parts = [
        symbol if power == 1 else f"{symbol}^{power}"
        for symbol, power in zip(symbols, exponents, strict=True)
        if power > 0
    ]
    return "*".join(parts) if parts else "1"


@dataclass(frozen=True)
class PolynomialSpace:
    """The span of a set of monomials, described by their exponent tuples.

    Immutable and hashable, so it can key the tabulation caches a later phase
    will need.
    """

    space_type: PolynomialSpaceType
    order: int
    n_variables: int
    exponents: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """Validate the space is well formed and its exponents are canonical."""
        if self.order < 0:
            raise InputValidationError(f"polynomial order must be >= 0, got {self.order}")
        if self.n_variables not in _SYMBOLS:
            raise InputValidationError(f"n_variables must be 1, 2, or 3, got {self.n_variables}")
        if not self.exponents:
            raise InputValidationError("a polynomial space needs at least one monomial")
        if len(set(self.exponents)) != len(self.exponents):
            raise InputValidationError("duplicate monomial in polynomial space")
        for alpha in self.exponents:
            if len(alpha) != self.n_variables:
                raise InputValidationError(
                    f"monomial {alpha} has {len(alpha)} exponents, expected {self.n_variables}"
                )
            if any(power < 0 for power in alpha):
                raise InputValidationError(f"monomial {alpha} has a negative exponent")

    # ---- constructors -------------------------------------------------------

    @classmethod
    def total_degree(cls, order: int, n_variables: int) -> PolynomialSpace:
        """The complete space ``P_k`` of total degree ``k`` in ``n_variables``."""
        if order < 0:
            raise InputValidationError(f"polynomial order must be >= 0, got {order}")
        if n_variables not in _SYMBOLS:
            raise InputValidationError(f"n_variables must be 1, 2, or 3, got {n_variables}")
        return cls(
            PolynomialSpaceType.TOTAL_DEGREE,
            order,
            n_variables,
            _monomials_up_to_total_degree(order, n_variables),
        )

    @classmethod
    def tensor_product(cls, order: int, n_variables: int) -> PolynomialSpace:
        """The tensor-product space ``Q_k`` of per-variable degree ``k``."""
        if order < 0:
            raise InputValidationError(f"polynomial order must be >= 0, got {order}")
        if n_variables not in _SYMBOLS:
            raise InputValidationError(f"n_variables must be 1, 2, or 3, got {n_variables}")
        return cls(
            PolynomialSpaceType.TENSOR_PRODUCT,
            order,
            n_variables,
            _monomials_tensor_product(order, n_variables),
        )

    # ---- structural queries -------------------------------------------------

    @property
    def dimension(self) -> int:
        """Number of monomials spanning the space (its dimension as a vector space)."""
        return len(self.exponents)

    @property
    def max_total_degree(self) -> int:
        """Highest total degree present (``k`` for ``P_k``, ``d * k`` for ``Q_k``)."""
        return max(sum(alpha) for alpha in self.exponents)

    @property
    def completeness_degree(self) -> int:
        """Largest ``p`` such that every monomial of total degree <= ``p`` is present.

        This is the degree that governs the approximation rate, and it is not
        the same as :pyattr:`max_total_degree`: ``Q_2`` in 2-D reaches total
        degree 4 (``xi^2*eta^2``) but is complete only to degree 2.
        """
        present = set(self.exponents)
        degree = -1
        for candidate in range(self.max_total_degree + 1):
            needed = set(_monomials_up_to_total_degree(candidate, self.n_variables))
            if not needed <= present:
                break
            degree = candidate
        return degree

    @property
    def contains_constant(self) -> bool:
        """Whether the constant monomial is in the space (needed to reproduce constants)."""
        return (0,) * self.n_variables in set(self.exponents)

    def contains_total_degree(self, degree: int) -> bool:
        """Whether every monomial of total degree exactly ``degree`` is present."""
        present = set(self.exponents)
        needed = {
            alpha
            for alpha in _monomials_up_to_total_degree(degree, self.n_variables)
            if sum(alpha) == degree
        }
        return needed <= present

    @property
    def monomial_labels(self) -> tuple[str, ...]:
        """Readable label of each monomial, index-aligned with :pyattr:`exponents`."""
        return tuple(monomial_label(alpha) for alpha in self.exponents)

    @property
    def name(self) -> str:
        """Conventional name of the space, e.g. ``"P2"`` or ``"Q3"``."""
        return f"{self.space_type.symbol}{self.order}"

    @property
    def expected_dimension(self) -> int:
        """Dimension predicted by the construction rule, for cross-checking.

        ``C(k + d, d)`` for ``P_k`` and ``(k + 1)^d`` for ``Q_k``.
        """
        if self.space_type is PolynomialSpaceType.TOTAL_DEGREE:
            return math.comb(self.order + self.n_variables, self.n_variables)
        if self.space_type is PolynomialSpaceType.TENSOR_PRODUCT:
            return int((self.order + 1) ** self.n_variables)
        raise NotImplementedError(
            f"dimension formula for {self.space_type.value} spaces arrives with that family"
        )

    # ---- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible record of the space."""
        return {
            "space_type": self.space_type.value,
            "name": self.name,
            "order": self.order,
            "n_variables": self.n_variables,
            "dimension": self.dimension,
            "completeness_degree": self.completeness_degree,
            "max_total_degree": self.max_total_degree,
            "exponents": [list(alpha) for alpha in self.exponents],
            "monomial_labels": list(self.monomial_labels),
        }

    def __repr__(self) -> str:
        return (
            f"PolynomialSpace({self.name}, n_variables={self.n_variables}, "
            f"dimension={self.dimension})"
        )
