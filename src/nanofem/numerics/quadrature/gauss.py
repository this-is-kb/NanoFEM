"""Gauss-Legendre and Gauss-Lobatto quadrature on the reference line ``[-1, 1]``.

Both are one-dimensional. The tensor-product construction lifts them to the
square and the cube; the triangle needs a simplex family instead, because a
tensor product of interval rules does not fit a simplex.

Why Gauss-Legendre is optimal
-----------------------------
An ``n``-point rule has ``2n`` free parameters - ``n`` points and ``n`` weights -
so it can in principle satisfy ``2n`` moment conditions and integrate every
polynomial of degree ``2n - 1``. Gauss-Legendre achieves exactly that, and no
``n``-point rule can do better: given any rule, the polynomial
``prod_q (x - x_q)^2`` has degree ``2n``, is strictly positive except at the
points, and integrates to zero under the rule while its true integral is
positive. So ``2n - 1`` is a ceiling, and Gauss attains it.

The points that achieve it are the roots of the Legendre polynomial ``P_n``,
which is what makes the family "Legendre": the Legendre polynomials are
orthogonal on ``[-1, 1]`` under the constant weight, and a rule placed at the
roots of the degree-``n`` orthogonal polynomial integrates the extra ``n``
degrees for free, because the remainder of any degree ``2n-1`` polynomial
divided by ``P_n`` is orthogonal to ``P_n``.

Gauss-Lobatto trades exactness for endpoints
--------------------------------------------
Fixing two of the points at ``-1`` and ``+1`` spends two degrees of freedom, so
an ``n``-point Lobatto rule reaches degree ``2n - 3`` rather than ``2n - 1``.
What it buys is the endpoints, which is precisely what lets its points coincide
with the nodes of a nodal basis. That collocation is the whole of the spectral
element method: when quadrature points and interpolation nodes coincide, a mass
matrix comes out diagonal. Its interior points are the roots of ``P'_{n-1}``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.polynomial import legendre
from numpy.typing import NDArray

from nanofem.numerics.quadrature.enums import QuadratureFamily
from nanofem.numerics.quadrature.rules import QuadratureRule
from nanofem.numerics.reference.element import ReferenceElement
from nanofem.numerics.reference.enums import CellType
from nanofem.numerics.reference.registry import reference_element
from nanofem.utils.exceptions import InputValidationError


@dataclass(frozen=True, eq=False, repr=False)
class GaussLegendreQuadrature(QuadratureRule):
    """Gauss-Legendre quadrature on ``[-1, 1]``: ``n`` points, exact to degree ``2n - 1``.

    Parameters
    ----------
    order
        The polynomial degree to integrate exactly. The point count follows,
        ``n = ceil((order + 1) / 2)``, so the delivered exactness may exceed the
        request by one - point counts are integers and the SDS forbids
        delivering less than asked.
    """

    order_: int

    def __init__(self, order: int) -> None:
        """Validate the requested degree."""
        if not isinstance(order, int) or isinstance(order, bool) or order < 0:
            raise InputValidationError(f"order must be an integer >= 0, got {order!r}")
        object.__setattr__(self, "order_", order)

    @property
    def family(self) -> QuadratureFamily:
        """The Gauss-Legendre family."""
        return QuadratureFamily.GAUSS_LEGENDRE

    @property
    def reference_element(self) -> ReferenceElement:
        """The reference line."""
        return reference_element(CellType.LINE)

    @property
    def order(self) -> int:
        """The requested exactness degree."""
        return self.order_

    @property
    def num_points(self) -> int:
        """``ceil((order + 1) / 2)``: the fewest points reaching the requested degree."""
        return max(1, math.ceil((self.order_ + 1) / 2))

    @property
    def exactness_degree(self) -> int:
        """``2n - 1``, the theoretical maximum for ``n`` points."""
        return 2 * self.num_points - 1

    @cached_property
    def _rule(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Points and weights from numpy's Legendre-Gauss routine.

        ``numpy.polynomial.legendre.leggauss`` returns the roots of ``P_n`` and
        the companion weights on ``[-1, 1]`` - the reference line's own domain,
        so no rescaling is needed and none is done.
        """
        nodes, weights = legendre.leggauss(self.num_points)
        points = np.ascontiguousarray(nodes.reshape(-1, 1), dtype=np.float64)
        values = np.ascontiguousarray(weights, dtype=np.float64)
        points.setflags(write=False)
        values.setflags(write=False)
        return points, values

    @property
    def points(self) -> NDArray[np.float64]:
        """The roots of ``P_n``, shape ``(n, 1)``, read-only."""
        return self._rule[0]

    @property
    def weights(self) -> NDArray[np.float64]:
        """The Gauss weights, all strictly positive, read-only."""
        return self._rule[1]


@dataclass(frozen=True, eq=False, repr=False)
class GaussLobattoQuadrature(QuadratureRule):
    """Gauss-Lobatto quadrature on ``[-1, 1]``: ``n`` points including both ends.

    Exact to degree ``2n - 3``. Needs at least two points, which is the
    trapezoid rule; three points is Simpson's rule.

    Parameters
    ----------
    order
        The polynomial degree to integrate exactly. The point count follows,
        ``n = max(2, ceil((order + 3) / 2))``.
    """

    order_: int

    def __init__(self, order: int) -> None:
        """Validate the requested degree."""
        if not isinstance(order, int) or isinstance(order, bool) or order < 0:
            raise InputValidationError(f"order must be an integer >= 0, got {order!r}")
        object.__setattr__(self, "order_", order)

    @property
    def family(self) -> QuadratureFamily:
        """The Gauss-Lobatto family."""
        return QuadratureFamily.GAUSS_LOBATTO

    @property
    def reference_element(self) -> ReferenceElement:
        """The reference line."""
        return reference_element(CellType.LINE)

    @property
    def order(self) -> int:
        """The requested exactness degree."""
        return self.order_

    @property
    def num_points(self) -> int:
        """``max(2, ceil((order + 3) / 2))``: two endpoints plus enough interior points."""
        return max(2, math.ceil((self.order_ + 3) / 2))

    @property
    def exactness_degree(self) -> int:
        """``2n - 3``: two degrees short of Gauss, the price of fixing the endpoints."""
        return 2 * self.num_points - 3

    @cached_property
    def _rule(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Interior points are the roots of ``P'_{n-1}``; weights close the moments.

        ``w_i = 2 / (n (n-1) P_{n-1}(x_i)^2)`` at every point, endpoints
        included. numpy has no Lobatto routine, but it has Legendre
        differentiation and root finding, which is all the construction needs.
        """
        count = self.num_points
        coefficients = np.zeros(count)
        coefficients[count - 1] = 1.0  # the polynomial P_{n-1}
        interior = legendre.legroots(legendre.legder(coefficients))
        nodes = np.concatenate([[-1.0], np.sort(interior), [1.0]])
        values = legendre.legval(nodes, coefficients)
        weights = 2.0 / (count * (count - 1) * values**2)
        points = np.ascontiguousarray(nodes.reshape(-1, 1), dtype=np.float64)
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        points.setflags(write=False)
        weights.setflags(write=False)
        return points, weights

    @property
    def points(self) -> NDArray[np.float64]:
        """The endpoints plus the roots of ``P'_{n-1}``, shape ``(n, 1)``, read-only."""
        return self._rule[0]

    @property
    def weights(self) -> NDArray[np.float64]:
        """The Lobatto weights, all strictly positive, read-only."""
        return self._rule[1]
