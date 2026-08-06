"""The ``QuadratureRule`` abstract base class (SDS 2.5).

A quadrature rule is a finite set of points and weights on a reference domain
such that ``sum_q w_q f(x_q)`` approximates ``integral f``, exactly for every
polynomial up to a declared degree.

That is the whole of it. A rule knows nothing about shape functions, finite
elements, constitutive models, or assembly; it integrates scalar functions on a
reference domain. It does not multiply weights by a Jacobian and it does not
integrate over a physical element - those compose a rule with a mapping, which
is the element layer's job, and doing it here would weld two independent ideas
together.

Reconciliation with the phase-0 seam
------------------------------------
The phase-0 skeleton declared ``QuadratureRule`` as a concrete value object with
a ``cell_name: str``. That draft predates the reference-element layer, which now
owns the reference domain, so the class becomes an abstract base and carries a
``ReferenceElement``; ``cell_name`` remains available for the mesh-facing
lookup. Same reconciliation as the ``ShapeFunctions`` seam in phase 4: the
declaration is filled, not bypassed.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.quadrature.enums import QuadratureFamily
from nanofem.numerics.quadrature.errors import ExactnessError, SymmetryError, WeightError
from nanofem.numerics.quadrature.moments import (
    exact_monomial_integral,
    monomial_exponents,
    monomial_exponents_up_to,
    monomial_values,
)
from nanofem.numerics.quadrature.symmetry import AffineSymmetry, symmetry_group
from nanofem.numerics.reference.element import ReferenceElement
from nanofem.numerics.reference.enums import CellType
from nanofem.utils.exceptions import InputValidationError

#: Tolerance for the exactness and moment identities on the unit reference scale.
_EXACTNESS_TOL: float = 1.0e-12

#: How far above the tolerance a rule must land to count as *not* exact.
_INEXACTNESS_TOL: float = 1.0e-9

#: Cells whose reference domain is a product of intervals, and cells that are simplices.
_TENSOR_PRODUCT_CELLS: frozenset[CellType] = frozenset(
    {CellType.LINE, CellType.QUADRILATERAL, CellType.HEXAHEDRON}
)
_SIMPLEX_CELLS: frozenset[CellType] = frozenset(
    {CellType.LINE, CellType.TRIANGLE, CellType.TETRAHEDRON}
)


class QuadratureRule(ABC):
    """Points and weights on a reference domain, with a declared exactness degree.

    Concrete rules declare :pyattr:`family`, :pyattr:`reference_element`,
    :pyattr:`order`, :pyattr:`exactness_degree`, :pyattr:`points`, and
    :pyattr:`weights`. Integration, moments, the verification suite,
    serialization, and value semantics are derived here and shared.
    """

    # ---- abstract declarations ---------------------------------------------

    @property
    @abstractmethod
    def family(self) -> QuadratureFamily:
        """The construction this rule belongs to."""

    @property
    @abstractmethod
    def reference_element(self) -> ReferenceElement:
        """The reference domain integrated over (phase 2)."""

    @property
    @abstractmethod
    def order(self) -> int:
        """The polynomial degree the caller asked to be integrated exactly.

        Distinct from :pyattr:`exactness_degree`, which is what the rule
        actually delivers. A Gauss rule asked for degree 4 uses three points and
        achieves degree 5, because point counts are integers. The invariant is
        ``exactness_degree >= order``: the SDS 2.5 policy is full integration by
        default, and implicit reduction is prohibited, so a rule never returns
        less than it was asked for.
        """

    @property
    @abstractmethod
    def exactness_degree(self) -> int:
        """The largest ``p`` for which every polynomial of total degree ``p`` is exact."""

    @property
    @abstractmethod
    def points(self) -> NDArray[np.float64]:
        """Integration points, shape ``(n_points, dim)``, read-only."""

    @property
    @abstractmethod
    def weights(self) -> NDArray[np.float64]:
        """Integration weights, shape ``(n_points,)``, read-only."""

    # ---- derived metadata ---------------------------------------------------

    @property
    def num_points(self) -> int:
        """How many points the rule uses."""
        return int(self.points.shape[0])

    @property
    def topological_dimension(self) -> int:
        """Dimension of the reference domain."""
        return self.reference_element.topological_dimension

    @property
    def cell_type(self) -> CellType:
        """The shape of the reference domain."""
        return self.reference_element.cell_type

    @property
    def is_tensor_product(self) -> bool:
        """Whether the reference domain is a product of intervals.

        A property of the *domain*, not of how this particular rule was built:
        a Gauss-Legendre rule on the line and a Dunavant rule on the triangle
        answer this the same way any other rule on those domains would. Whether
        the rule's *points* form a grid is :pyattr:`family`'s business.
        """
        return self.cell_type in _TENSOR_PRODUCT_CELLS

    @property
    def is_simplex(self) -> bool:
        """Whether the reference domain is a simplex.

        The line is both a 1-simplex and a 1-cube, so both flags are ``True``
        for it - the same honest edge case the interpolation layer records.
        """
        return self.cell_type in _SIMPLEX_CELLS

    @property
    def has_positive_weights(self) -> bool:
        """Whether every weight is strictly positive.

        Not a given. Negative weights are legal and some classical rules have
        them - the Dunavant degree-3 triangle rule being the famous case - but
        they cost accuracy in floating point (cancellation) and can make an
        integrated mass matrix indefinite, so a caller integrating something
        that must stay positive wants to know.
        """
        return bool(np.all(self.weights > 0.0))

    @property
    def total_weight(self) -> float:
        """Sum of the weights, which must equal the reference measure."""
        return float(self.weights.sum())

    @cached_property
    def normalized_weights(self) -> NDArray[np.float64]:
        """Weights scaled to sum to one, read-only.

        The form classical tables are published in, since it separates the rule
        from the domain's measure.
        """
        scaled: NDArray[np.float64] = self.weights / self.total_weight
        scaled.setflags(write=False)
        return scaled

    @property
    def declares_symmetry(self) -> bool:
        """Whether this rule claims invariance under the domain's full symmetry group.

        A claim, not a fact - :pyattr:`is_symmetric` is the fact, and
        :meth:`verify_symmetry` checks the two agree. Most classical rules are
        symmetric by construction and this default says so, but an anisotropic
        tensor product genuinely is not, and imposing symmetry on it would be
        rejecting a legitimate rule rather than testing one.
        """
        return True

    @cached_property
    def invariant_symmetries(self) -> tuple[AffineSymmetry, ...]:
        """The subgroup of the domain's symmetries the rule is actually invariant under.

        Computed, not declared. An isotropic rule on the square is invariant
        under all eight dihedral maps; an anisotropic one keeps only the four
        that fix the axes, because swapping the axes swaps factors that differ.
        """
        return tuple(
            symmetry
            for symmetry in symmetry_group(self.reference_element)
            if self._is_invariant_under(symmetry)
        )

    @property
    def is_symmetric(self) -> bool:
        """Whether the rule is invariant under *every* symmetry of its domain."""
        return len(self.invariant_symmetries) == len(symmetry_group(self.reference_element))

    def _is_invariant_under(self, symmetry: AffineSymmetry) -> bool:
        """Whether every point maps onto a point of the rule carrying the same weight."""
        images = symmetry.apply(self.points)
        for index, image in enumerate(images):
            matches = np.nonzero(np.all(np.isclose(self.points, image, atol=1.0e-10), axis=1))[0]
            if matches.size == 0:
                return False
            if not np.isclose(self.weights[matches[0]], self.weights[index], atol=_EXACTNESS_TOL):
                return False
        return True

    @cached_property
    def symmetry_orbits(self) -> tuple[tuple[int, ...], ...]:
        """Point indices grouped into orbits of the symmetries the rule respects.

        Every point in an orbit carries the same weight. This is how classical
        tables are published - "one centroid point plus a three-fold orbit" -
        and it is recovered here from the points rather than declared. Built
        from :pyattr:`invariant_symmetries` rather than the domain's full group,
        so an anisotropic rule's orbits are the true ones.
        """
        remaining = set(range(self.num_points))
        orbits: list[tuple[int, ...]] = []
        while remaining:
            seed = min(remaining)
            orbit = {seed}
            for symmetry in self.invariant_symmetries:
                image = symmetry.apply(self.points[seed : seed + 1])[0]
                matches = np.nonzero(np.all(np.isclose(self.points, image, atol=1e-10), axis=1))[0]
                orbit.update(int(index) for index in matches)
            orbits.append(tuple(sorted(orbit)))
            remaining -= orbit
        return tuple(orbits)

    @cached_property
    def _moment_cache(self) -> dict[tuple[int, ...], float]:
        """Per-rule memo of computed moments."""
        return {}

    # ---- integration --------------------------------------------------------

    def integrate(
        self, function: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    ) -> NDArray[np.float64] | float:
        """Integrate a scalar function over the reference domain.

        ``function`` receives the points as an ``(n_points, dim)`` array and
        returns ``(n_points,)`` for a scalar field or ``(n_points, ...)`` for a
        vector or tensor one, which is integrated componentwise. The result is a
        float in the scalar case.

        This is integration over the *reference* domain. Integrating over a
        physical element means composing this with a mapping's measure scaling,
        which is the element layer's business - a rule that did it itself would
        have to know what a Jacobian is.
        """
        values = np.asarray(function(self.points), dtype=np.float64)
        return self.integrate_values(values)

    def integrate_values(self, values: NDArray[np.float64]) -> NDArray[np.float64] | float:
        """Integrate a field already evaluated at :pyattr:`points`.

        The form a caller with a cached tabulation wants: no callable, no
        re-evaluation.
        """
        array = np.asarray(values, dtype=np.float64)
        if array.shape[0] != self.num_points:
            raise InputValidationError(
                f"values has {array.shape[0]} rows but the rule has {self.num_points} " f"points"
            )
        result = np.tensordot(self.weights, array, axes=(0, 0))
        if result.ndim == 0:
            return float(result)
        return np.asarray(result, dtype=np.float64)

    def moment(self, exponents: tuple[int, ...]) -> float:
        """The numerical moment ``sum_q w_q x_q^exponents``, cached per rule."""
        key = tuple(exponents)
        cached = self._moment_cache.get(key)
        if cached is not None:
            return cached
        value = float(self.weights @ monomial_values(key, self.points))
        self._moment_cache[key] = value
        return value

    def exact_moment(self, exponents: tuple[int, ...]) -> float:
        """The closed-form moment, computed analytically rather than by this rule."""
        return exact_monomial_integral(self.reference_element, tuple(exponents))

    def integrate_monomial(self, exponents: tuple[int, ...]) -> float:
        """The numerical integral of one monomial; an alias of :meth:`moment`."""
        return self.moment(exponents)

    def measure(self) -> float:
        """The domain's measure by quadrature: ``integral 1``, i.e. the weight sum."""
        return self.moment((0,) * self.topological_dimension)

    def centroid(self) -> NDArray[np.float64]:
        """The domain's centroid by quadrature: ``integral x / integral 1``.

        Recovers ``(1/3, 1/3)`` for the triangle and the origin for the line and
        square. Unlike the reference element's vertex-average centroid, this one
        is the genuine first moment - the two agree on these domains, which is a
        useful cross-check between the layers.
        """
        dimension = self.topological_dimension
        first = np.array(
            [
                self.moment(tuple(1 if d == axis else 0 for d in range(dimension)))
                for axis in range(dimension)
            ]
        )
        return np.asarray(first / self.measure(), dtype=np.float64)

    # ---- verification -------------------------------------------------------

    def is_valid(self) -> bool:
        """Return ``True`` if :meth:`verify` passes, ``False`` otherwise."""
        try:
            self.verify()
        except (ExactnessError, WeightError, SymmetryError):
            return False
        return True

    def verify(self) -> None:
        """Run every identity the rule must satisfy; raise on the first failure."""
        self.verify_weight_normalization()
        self.verify_weight_positivity()
        self.verify_points_in_domain()
        self.verify_exactness()
        self.verify_exactness_is_maximal()
        self.verify_moment_identities()
        self.verify_symmetry()

    def verify_weight_normalization(self) -> None:
        """Check the weights sum to the reference measure.

        Equivalently, that the rule integrates the constant 1 exactly - the
        first and weakest exactness condition, worth stating separately because
        it is the one a mis-scaled table fails.
        """
        expected = self.reference_element.reference_measure
        if not np.isclose(self.total_weight, expected, atol=_EXACTNESS_TOL):
            raise WeightError(
                f"{self._label()}: the weights sum to {self.total_weight:.16g} but the "
                f"reference {self.cell_type.value} has measure {expected:.16g}"
            )
        if not np.isclose(float(self.normalized_weights.sum()), 1.0, atol=_EXACTNESS_TOL):
            raise WeightError(f"{self._label()}: the normalized weights do not sum to 1")

    def verify_weight_positivity(self) -> None:
        """Check the weights are positive, when the rule claims they are.

        Deliberately checks the *claim* rather than imposing it: a negative
        weight is a legitimate property of some classical rules, so failing them
        would be wrong. What must never happen is a rule reporting positive
        weights while carrying a negative one.
        """
        actual = bool(np.all(self.weights > 0.0))
        if self.has_positive_weights and not actual:
            worst = float(self.weights.min())
            raise WeightError(
                f"{self._label()}: the rule reports positive weights but the smallest is "
                f"{worst:.6g}"
            )
        if np.any(np.isclose(self.weights, 0.0, atol=_EXACTNESS_TOL)):
            raise WeightError(f"{self._label()}: a weight is zero, so a point does nothing")

    def verify_points_in_domain(self) -> None:
        """Check every point lies in the reference domain.

        Uses the phase-2 containment test, so a rule tabulated for a different
        domain - or with a transcription error in the table - is caught before
        it can integrate anything.
        """
        for index, point in enumerate(self.points):
            if not self.reference_element.contains(point, tol=1.0e-10):
                raise WeightError(
                    f"{self._label()}: point {index} at {point.tolist()} lies outside the "
                    f"reference {self.cell_type.value}"
                )

    def verify_exactness(self) -> None:
        """Check every polynomial within the declared degree integrates exactly.

        Against the closed forms in ``moments``, which are analytic - a rule
        checked against itself, or against another rule, would prove nothing.
        """
        dimension = self.topological_dimension
        for exponents in monomial_exponents_up_to(self.exactness_degree, dimension):
            numerical = self.moment(exponents)
            exact = self.exact_moment(exponents)
            if not np.isclose(numerical, exact, atol=_EXACTNESS_TOL, rtol=1.0e-12):
                raise ExactnessError(
                    f"{self._label()}: the monomial with exponents {exponents} integrates "
                    f"to {numerical:.16g} but its exact value is {exact:.16g}"
                )
        if self.exactness_degree < self.order:
            raise ExactnessError(
                f"{self._label()}: the rule was asked for degree {self.order} but only "
                f"delivers {self.exactness_degree}; implicit reduction is prohibited"
            )

    def verify_exactness_is_maximal(self) -> None:
        """Check the rule is *not* exact one degree higher.

        Without this, a rule could understate itself and every exactness test
        would still pass - and a caller would pay for points it did not need. A
        rule fails this only if some monomial of the next degree is integrated
        wrongly, so symmetry matters: an odd monomial that vanishes by symmetry
        proves nothing, which is why every monomial of the next degree is tried
        and only one needs to miss.
        """
        dimension = self.topological_dimension
        next_degree = self.exactness_degree + 1
        for exponents in monomial_exponents(next_degree, dimension):
            numerical = self.moment(exponents)
            exact = self.exact_moment(exponents)
            if not np.isclose(numerical, exact, atol=_INEXACTNESS_TOL, rtol=1.0e-9):
                return  # found the monomial that fails: the degree is maximal
        raise ExactnessError(
            f"{self._label()}: every monomial of degree {next_degree} also integrates "
            f"exactly, so the rule is stronger than the degree {self.exactness_degree} it "
            f"reports"
        )

    def verify_moment_identities(self) -> None:
        """Check the measure and centroid the rule computes.

        A restatement of low-order exactness in the form a reader recognizes:
        the measure must be the domain's, and the first moment must land on the
        centroid the reference element reports from its vertices. The two
        centroids agree on every domain here, so this also cross-checks phase 2.
        """
        measure = self.measure()
        expected = self.reference_element.reference_measure
        if not np.isclose(measure, expected, atol=_EXACTNESS_TOL):
            raise ExactnessError(
                f"{self._label()}: integrating 1 gives {measure:.16g}, expected {expected:.16g}"
            )
        if self.exactness_degree >= 1:
            centroid = self.centroid()
            if not np.allclose(centroid, self.reference_element.centroid, atol=1.0e-12):
                raise ExactnessError(
                    f"{self._label()}: the integrated centroid {centroid.tolist()} differs "
                    f"from the reference centroid "
                    f"{self.reference_element.centroid.tolist()}"
                )

    def verify_symmetry(self) -> None:
        """Check the rule's symmetry claim against its points.

        Symmetry matters because it makes a rule's behaviour independent of how
        a mesh generator happened to number the element's vertices: a
        non-symmetric rule integrates the same polynomial differently depending
        on which corner is called vertex zero. So a rule that claims symmetry
        must have it. A rule that does not claim it - an anisotropic tensor
        product - is not failed for lacking it; its true invariant subgroup is
        reported by :pyattr:`invariant_symmetries` instead.
        """
        if not self.declares_symmetry:
            return
        full = symmetry_group(self.reference_element)
        if len(self.invariant_symmetries) == len(full):
            return
        missing = next(s for s in full if not self._is_invariant_under(s))
        raise SymmetryError(
            f"{self._label()}: the rule claims symmetry but is not invariant under the "
            f"vertex permutation {missing.permutation}"
        )

    # ---- serialization and utilities ---------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible record of the rule."""
        return {
            "schema": "nanofem-quadrature-rule/1",
            "family": self.family.value,
            "cell_type": self.cell_type.value,
            "order": self.order,
            "exactness_degree": self.exactness_degree,
            "num_points": self.num_points,
            "dimension": self.topological_dimension,
            "is_simplex": self.is_simplex,
            "is_tensor_product": self.is_tensor_product,
            "has_positive_weights": self.has_positive_weights,
            "total_weight": self.total_weight,
            "points": self.points.tolist(),
            "weights": self.weights.tolist(),
            "is_symmetric": self.is_symmetric,
            "symmetry_orbits": [list(orbit) for orbit in self.symmetry_orbits],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize :meth:`to_dict` to JSON (keys sorted for determinism)."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    def pretty(self) -> str:
        """A compact, human-readable summary of the rule."""
        return "\n".join(
            [
                f"{type(self).__name__} [{self._label()}]",
                f"  points / weights : {self.num_points} points, "
                f"{'all positive' if self.has_positive_weights else 'MIXED SIGN'}",
                f"  exactness        : degree {self.exactness_degree} " f"(asked for {self.order})",
                f"  weight sum       : {self.total_weight:.12g} "
                f"(reference measure {self.reference_element.reference_measure:g})",
                f"  flags            : simplex={self.is_simplex} "
                f"tensor_product={self.is_tensor_product} symmetric={self.is_symmetric}",
                f"  symmetry orbits  : {[len(o) for o in self.symmetry_orbits]}",
            ]
        )

    def debug_summary(self) -> str:
        """A verbose dump: the summary plus every point and weight."""
        parts = [self.pretty(), "  quadrature points:"]
        for index, (point, weight) in enumerate(zip(self.points, self.weights, strict=True)):
            coordinates = ", ".join(f"{x:+.12f}" for x in point)
            parts.append(f"    [{index}] ({coordinates})  w = {weight:+.12f}")
        return "\n".join(parts)

    def _label(self) -> str:
        """Short identifier used in error messages."""
        return f"{self.family.value} order {self.order} on the reference " f"{self.cell_type.value}"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(cell_type={self.cell_type.value!r}, "
            f"order={self.order}, points={self.num_points}, "
            f"exactness={self.exactness_degree})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuadratureRule):
            return NotImplemented
        return (
            self.family == other.family
            and self.cell_type == other.cell_type
            and self.order == other.order
            and np.array_equal(self.points, other.points)
            and np.array_equal(self.weights, other.weights)
        )

    def __hash__(self) -> int:
        return hash(
            (
                QuadratureRule,
                self.family,
                self.cell_type,
                self.order,
                self.points.tobytes(),
                self.weights.tobytes(),
            )
        )
