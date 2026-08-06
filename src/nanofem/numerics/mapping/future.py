"""Declared placeholders for the remaining mapping kinds.

Each is named so registries and interfaces anticipate it, carries a
``PROVISIONAL_METADATA`` mapping recording its intended shape, and raises on
construction with a ``BLOCKED_BY`` string naming what it structurally needs.

None of the three is blocked by scheduling. What is worth noticing is how little
each needs, because the base class was written through the pseudo-inverse and
the mapping Hessian rather than around them: every one of these is a new
``map`` / ``jacobian`` / ``mapping_hessian`` triple, and inherits the metric,
the bases, the derivative transformations, the validation, and the verification
suite unchanged. That is the measurable claim of this layer's design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.interpolation.tabulation import PointsLike
from nanofem.numerics.mapping.base import GeometricMapping
from nanofem.numerics.mapping.enums import MappingType
from nanofem.numerics.reference.element import ReferenceElement


@dataclass(frozen=True, eq=False, repr=False, init=False)
class _FutureMapping(GeometricMapping):
    """Shared placeholder base: satisfies the interface but refuses construction."""

    #: Intended metadata, for documentation and as the future build target.
    PROVISIONAL_METADATA: ClassVar[dict[str, Any]] = {}

    #: Why this mapping cannot be built yet, in terms of what it needs.
    BLOCKED_BY: ClassVar[str] = ""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Refuse construction, naming what the mapping still needs."""
        raise NotImplementedError(
            f"{type(self).__name__} is a declared placeholder: {self.BLOCKED_BY}"
        )

    @property
    def mapping_type(self) -> MappingType:  # pragma: no cover - construction raises first
        """Not available for a placeholder."""
        raise NotImplementedError

    @property
    def reference_element(self) -> ReferenceElement:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    @property
    def physical_nodes(self) -> NDArray[np.float64]:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    @property
    def is_affine(self) -> bool:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    def map(self, points: PointsLike) -> NDArray[np.float64]:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    def inverse_map(self, points: PointsLike) -> NDArray[np.float64]:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    def jacobian(self, points: PointsLike) -> NDArray[np.float64]:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    def mapping_hessian(self, points: PointsLike) -> NDArray[np.float64]:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError


@dataclass(frozen=True, eq=False, repr=False, init=False)
class CurvilinearMapping(_FutureMapping):
    """Placeholder for a mapping onto an analytically described curved boundary.

    Intended: the geometry follows an exact surface - a circle, a cylinder, a CAD
    patch - rather than a polynomial interpolant, so a boundary element's nodes
    are projected onto the true surface and the map is the composition. This is
    what keeps a curved boundary from being approximated to the element order,
    and it is the natural home for the manifold description a shell needs.
    """

    PROVISIONAL_METADATA: ClassVar[dict[str, Any]] = {
        "mapping_type": MappingType.CURVILINEAR.value,
        "is_affine": False,
        "geometry_source": "analytic surface description",
        "needs": ["surface parameterization", "point projection onto the surface"],
    }
    BLOCKED_BY = (
        "it needs an analytic surface description to map onto - a geometry/CAD concern that "
        "has no representation in the library yet; the map itself would be a map / jacobian "
        "/ mapping_hessian triple like any other"
    )


@dataclass(frozen=True, eq=False, repr=False, init=False)
class NURBSMapping(_FutureMapping):
    """Placeholder for the isogeometric mapping, with NURBS as the geometry basis.

    Intended: ``x(xi) = sum_a R_a(xi) P_a`` over rational B-spline basis
    functions and control points. Structurally an isoparametric map with a
    different basis - which is exactly why it is blocked here and not in this
    layer: a NURBS basis is not a ``ShapeFunctionFamily``. Its functions are
    rational rather than polynomial, so they are not spanned by a monomial
    space, its support is a patch rather than a cell, and its control points are
    not interpolation nodes. That is an interpolation-layer question first.
    """

    PROVISIONAL_METADATA: ClassVar[dict[str, Any]] = {
        "mapping_type": MappingType.NURBS.value,
        "is_affine": False,
        "geometry_source": "rational B-spline basis over control points",
        "needs": ["knot vectors", "control points and weights", "a rational basis family"],
    }
    BLOCKED_BY = (
        "a NURBS basis is rational, patch-supported, and non-interpolatory, so it is not a "
        "polynomial ShapeFunctionFamily; the interpolation layer would need a rational "
        "family before this map could consume one"
    )


@dataclass(frozen=True, eq=False, repr=False, init=False)
class HighOrderMapping(_FutureMapping):
    """Placeholder for a mapping optimized for high-order polynomial geometry.

    Intended: the same mathematics ``IsoparametricMapping`` already implements -
    which is the point. ``IsoparametricMapping`` is order-agnostic today and
    accepts any nodal basis the interpolation layer offers, so a cubic geometry
    already works. What this class would add is the machinery high order *needs*
    to be trustworthy: nodes at Gauss-Lobatto rather than equispaced positions
    (which is the spectral family, still waiting on quadrature), validity
    checking away from the sample points, since a high-order map can invert in a
    cell's interior while every vertex looks fine, and the blending that keeps
    interior nodes sensible when only the boundary is curved.
    """

    PROVISIONAL_METADATA: ClassVar[dict[str, Any]] = {
        "mapping_type": MappingType.HIGH_ORDER.value,
        "is_affine": False,
        "geometry_source": "high-order nodal basis",
        "needs": [
            "well-conditioned node sets (the spectral family, blocked on quadrature)",
            "interior validity checks, not just at sample points",
            "transfinite blending for partially curved elements",
        ],
    }
    BLOCKED_BY = (
        "IsoparametricMapping already handles high-order geometry mathematically; what is "
        "missing is what makes it trustworthy - Gauss-Lobatto node sets (blocked on "
        "quadrature), interior validity checking, and transfinite blending"
    )
