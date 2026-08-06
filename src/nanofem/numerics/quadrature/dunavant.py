"""Dunavant quadrature on the reference triangle (Dunavant, 1985).

A tensor product does not fit a simplex - the integration limits are coupled -
so the triangle needs rules constructed for it directly. Dunavant's are the
standard choice: fully symmetric under the triangle's six vertex permutations,
with the fewest points known for each degree.

Symmetric orbits
----------------
The rules are published in *barycentric* coordinates and organized into orbits
of the permutation group, which is what makes their symmetry exact rather than
approximate:

- **centroid**: one point at ``(1/3, 1/3, 1/3)``;
- **three-fold**: three points, the permutations of ``(a, b, b)`` with
  ``b = (1 - a) / 2``, so one parameter names the whole orbit;
- **six-fold**: six points from ``(a, b, c)``, needed from degree 6 upward.

Storing the generator ``a`` and deriving ``b = (1-a)/2`` removes a transcription
error: a table listing ``a`` and ``b`` independently to fifteen digits leaves
``a + 2b`` off by around 1e-15, which drags the points microscopically off the
triangle. Deriving it leaves at most one rounding of the division. Only centroid
and three-fold orbits appear up to degree 5.

The degree-3 rule has a negative weight
---------------------------------------
Its centroid weight is ``-27/48``. This is not a transcription error: it is the
well known price of reaching degree 3 with four points, and it is why
:pyattr:`has_positive_weights` is data rather than an assumption. Negative
weights cost accuracy through cancellation and can make an integrated mass
matrix indefinite, so a caller who needs positivity should ask for degree 4 -
which costs six points but is positive - rather than degree 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.quadrature.enums import QuadratureFamily
from nanofem.numerics.quadrature.rules import QuadratureRule
from nanofem.numerics.reference.element import ReferenceElement
from nanofem.numerics.reference.enums import CellType
from nanofem.numerics.reference.registry import reference_element
from nanofem.utils.exceptions import InputValidationError

#: Orbits per degree as (generator, normalized weight). ``None`` names the centroid;
#: a float ``a`` names the three-fold orbit through ``(a, (1-a)/2, (1-a)/2)``.
#: Weights are normalized to sum to 1 and are scaled by the triangle's area at use.
#: Values from Dunavant (1985), Int. J. Numer. Methods Eng. 21, Table 2.
_ORBITS: dict[int, tuple[tuple[float | None, float], ...]] = {
    1: ((None, 1.000000000000000),),
    2: ((0.666666666666667, 0.333333333333333),),
    3: (
        (None, -0.562500000000000),
        (0.600000000000000, 0.520833333333333),
    ),
    4: (
        (0.108103018168070, 0.223381589678011),
        (0.816847572980459, 0.109951743655322),
    ),
    5: (
        (None, 0.225000000000000),
        (0.059715871789770, 0.132394152788506),
        (0.797426985353087, 0.125939180544827),
    ),
}

#: The degrees this table covers. Dunavant's paper tabulates up to 20; higher
#: degrees land by extending _ORBITS, and need the six-fold orbit as well.
SUPPORTED_DEGREES: tuple[int, ...] = tuple(sorted(_ORBITS))


@dataclass(frozen=True, eq=False, repr=False)
class DunavantQuadrature(QuadratureRule):
    """A symmetric triangle rule of the requested degree (Dunavant, 1985).

    Parameters
    ----------
    order
        The polynomial degree to integrate exactly, 1 to 5. Dunavant's tables
        are published per degree, so the rule delivers exactly what is asked -
        unlike a Gauss rule, whose integer point count can overshoot.
    """

    order_: int

    def __init__(self, order: int) -> None:
        """Validate the requested degree against the tabulated rules."""
        if not isinstance(order, int) or isinstance(order, bool):
            raise InputValidationError(f"order must be an integer, got {order!r}")
        if order not in _ORBITS:
            available = ", ".join(str(degree) for degree in SUPPORTED_DEGREES)
            raise InputValidationError(
                f"no Dunavant rule of degree {order} is tabulated; available degrees: "
                f"{available}. Dunavant (1985) publishes rules to degree 20, which land by "
                f"extending the table (degree 6 and up also need the six-fold orbit)"
            )
        object.__setattr__(self, "order_", order)

    @property
    def family(self) -> QuadratureFamily:
        """The Dunavant family."""
        return QuadratureFamily.DUNAVANT

    @property
    def reference_element(self) -> ReferenceElement:
        """The reference triangle."""
        return reference_element(CellType.TRIANGLE)

    @property
    def order(self) -> int:
        """The requested exactness degree."""
        return self.order_

    @property
    def exactness_degree(self) -> int:
        """Equal to the order: the tables are published per degree."""
        return self.order_

    @property
    def orbits(self) -> tuple[tuple[float | None, float], ...]:
        """The (generator, normalized weight) pairs this rule is built from."""
        return _ORBITS[self.order_]

    @cached_property
    def _rule(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Expand the symmetric orbits into cartesian points and weights.

        The reference triangle has vertices ``(0,0)``, ``(1,0)``, ``(0,1)``, so
        barycentric ``(l0, l1, l2)`` sits at cartesian ``(l1, l2)``. Weights are
        the normalized table values times the triangle's area.
        """
        area = self.reference_element.reference_measure
        coordinates: list[tuple[float, float]] = []
        weights: list[float] = []
        for generator, weight in self.orbits:
            for barycentric in _expand_orbit(generator):
                coordinates.append((barycentric[1], barycentric[2]))
                weights.append(weight * area)
        points = np.ascontiguousarray(coordinates, dtype=np.float64)
        values = np.ascontiguousarray(weights, dtype=np.float64)
        points.setflags(write=False)
        values.setflags(write=False)
        return points, values

    @property
    def points(self) -> NDArray[np.float64]:
        """The orbit points in cartesian reference coordinates, read-only."""
        return self._rule[0]

    @property
    def weights(self) -> NDArray[np.float64]:
        """The orbit weights, scaled by the triangle's area, read-only."""
        return self._rule[1]

    def verify_barycentric_consistency(self) -> None:
        """Check every point's barycentric coordinates sum to one and are in range.

        Exact by construction, since ``b`` is derived from ``a`` rather than
        transcribed - which is the point of storing the generator. Asserted
        anyway, because a table entry outside ``[0, 1]`` would put a point
        outside the triangle and this is the check that names why.
        """
        for index, point in enumerate(self.points):
            barycentric = np.array([1.0 - point[0] - point[1], point[0], point[1]])
            if not np.isclose(barycentric.sum(), 1.0, atol=1e-15):
                raise InputValidationError(
                    f"{self._label()}: point {index} has barycentric coordinates summing "
                    f"to {barycentric.sum():.17g}"
                )
            if barycentric.min() < -1e-15:
                raise InputValidationError(
                    f"{self._label()}: point {index} has a negative barycentric coordinate, "
                    f"so it lies outside the triangle"
                )

    def verify(self) -> None:
        """Run the generic identities plus the barycentric check."""
        super().verify()
        self.verify_barycentric_consistency()


def _expand_orbit(generator: float | None) -> tuple[tuple[float, float, float], ...]:
    """Expand one symmetric orbit into its barycentric points.

    ``None`` is the centroid. A float ``a`` gives the three cyclic permutations
    of ``(a, b, b)`` with ``b = (1 - a) / 2``, so the barycentric sum is exact.
    """
    if generator is None:
        third = 1.0 / 3.0
        return ((third, third, third),)
    other = (1.0 - generator) / 2.0
    return (
        (generator, other, other),
        (other, generator, other),
        (other, other, generator),
    )
