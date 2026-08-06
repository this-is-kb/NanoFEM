"""Declared placeholders for the remaining quadrature families.

Each carries ``PROVISIONAL_METADATA`` and a ``BLOCKED_BY`` string naming what it
structurally needs. As in the other layers, none is blocked by scheduling - and
what each needs says something about where it belongs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.quadrature.enums import QuadratureFamily
from nanofem.numerics.quadrature.rules import QuadratureRule
from nanofem.numerics.reference.element import ReferenceElement


@dataclass(frozen=True, eq=False, repr=False, init=False)
class _FutureQuadrature(QuadratureRule):
    """Shared placeholder base: satisfies the interface but refuses construction."""

    PROVISIONAL_METADATA: ClassVar[dict[str, Any]] = {}
    BLOCKED_BY: ClassVar[str] = ""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Refuse construction, naming what the family still needs."""
        raise NotImplementedError(
            f"{type(self).__name__} is a declared placeholder: {self.BLOCKED_BY}"
        )

    @property
    def family(self) -> QuadratureFamily:  # pragma: no cover - construction raises first
        """Not available for a placeholder."""
        raise NotImplementedError

    @property
    def reference_element(self) -> ReferenceElement:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    @property
    def order(self) -> int:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    @property
    def exactness_degree(self) -> int:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    @property
    def points(self) -> NDArray[np.float64]:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    @property
    def weights(self) -> NDArray[np.float64]:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError


@dataclass(frozen=True, eq=False, repr=False, init=False)
class GaussJacobiQuadrature(_FutureQuadrature):
    """Placeholder for Gauss-Jacobi quadrature, exact against a weight function.

    Intended: points and weights that make ``integral w(x) f(x)`` exact for
    polynomial ``f``, where ``w(x) = (1-x)^a (1+x)^b`` is absorbed into the rule
    rather than sampled. Two uses, both on this library's roadmap.

    The collapsed-coordinate (Duffy) map from a square onto a triangle carries a
    Jacobian ``(1-eta)/2``, which is a Jacobi weight; integrating it with
    Gauss-Legendre wastes points on a function the rule could have absorbed
    exactly. That is the standard route to high-order simplex rules, and it is
    what would carry the triangle past the degree Dunavant's tables cover.

    More importantly for the physics: Eringen's integral formulation weights the
    strain at ``x'`` by an attenuation kernel ``alpha(|x - x'|)``. That kernel is
    not polynomial, so a Gauss-Legendre rule integrates it only approximately -
    and this is a family designed for exactly that situation.
    """

    PROVISIONAL_METADATA: ClassVar[dict[str, Any]] = {
        "family": QuadratureFamily.GAUSS_JACOBI.value,
        "weight_function": "(1-x)^alpha (1+x)^beta",
        "exactness_degree": "2n - 1, against the weight",
        "has_positive_weights": True,
        "needs": ["Jacobi polynomial roots", "alpha/beta parameters on the rule"],
    }
    BLOCKED_BY = (
        "it needs Jacobi polynomial roots and a place on the interface for the weight "
        "exponents; the rule is otherwise the same shape as Gauss-Legendre, and numpy "
        "provides no Jacobi routine, so the recurrence would be written here"
    )


@dataclass(frozen=True, eq=False, repr=False, init=False)
class AdaptiveQuadrature(_FutureQuadrature):
    """Placeholder for error-driven adaptive quadrature.

    Intended: subdivide the reference domain until an estimated error falls
    below a tolerance, refining only where the integrand needs it.

    This one does not fit the interface, and that is the interesting part. Every
    rule here has a fixed point set decided before it sees an integrand, which
    is what lets a tabulation be computed once and shared across every element
    of a block (SDS C-8). An adaptive rule's points depend on the *function*, so
    it cannot be a ``QuadratureRule`` in this sense at all - it is an
    integration *algorithm* that calls rules. It would sit beside this package
    rather than inside it.
    """

    PROVISIONAL_METADATA: ClassVar[dict[str, Any]] = {
        "family": QuadratureFamily.ADAPTIVE.value,
        "points_depend_on_integrand": True,
        "needs": ["an error estimator", "domain subdivision", "a different interface"],
    }
    BLOCKED_BY = (
        "its points depend on the integrand, so it is an integration algorithm rather than "
        "a rule; a fixed point set is what lets tabulations be shared across a block "
        "(SDS C-8), and adaptivity breaks that contract rather than extending it"
    )


@dataclass(frozen=True, eq=False, repr=False, init=False)
class SparseGridQuadrature(_FutureQuadrature):
    """Placeholder for Smolyak sparse-grid quadrature in high dimensions.

    Intended: a Smolyak combination of one-dimensional rules whose point count
    grows polynomially in the dimension rather than as ``n^d``.

    Not useful for the element integrals this library performs - ``d`` is at
    most 3, where a full tensor product is already cheap. It earns its place
    against the roadmap's *stochastic* directions: uncertainty quantification
    over many material parameters, where the integration dimension is the number
    of uncertain inputs rather than the number of spatial ones. Its weights are
    signed by construction, which is why :pyattr:`has_positive_weights` had to
    be data from the start.
    """

    PROVISIONAL_METADATA: ClassVar[dict[str, Any]] = {
        "family": QuadratureFamily.SPARSE_GRID.value,
        "has_positive_weights": False,
        "useful_dimension": "> 4, i.e. parameter space rather than physical space",
        "needs": ["nested 1-D rule hierarchies", "the Smolyak combination formula"],
    }
    BLOCKED_BY = (
        "it pays off only above three or four dimensions, which element integration never "
        "reaches; it would arrive with parametric or stochastic analysis, and needs nested "
        "rule hierarchies the tensor product does not require"
    )
