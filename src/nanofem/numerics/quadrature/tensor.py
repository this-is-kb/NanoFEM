"""Tensor-product quadrature: lifting one-dimensional rules onto product domains.

On a domain that is a product of intervals, the integral separates::

    integral_[-1,1]^d f = integral ... integral f dxi_1 ... dxi_d

so a rule follows from ``d`` one-dimensional rules: the points are the cartesian
product of the one-dimensional points, and each weight is the product of the
corresponding one-dimensional weights.

Exactness of the construction
-----------------------------
If each factor is exact to degree ``p_k`` in its own variable, the product is
exact for every monomial with ``a_k <= p_k`` for all ``k`` - that is, for the
whole tensor-product space ``Q_p`` with ``p = min_k p_k``. Since ``P_p`` is
contained in ``Q_p``, the rule's *total-degree* exactness is ``p``, and it is
not more: ``xi^(p+1)`` is a total-degree ``p+1`` monomial the first factor gets
wrong.

So a tensor-product rule reports ``exactness_degree = p`` while quietly being
exact on a space strictly larger than ``P_p``. That extra strength is real and
is what makes these rules the right choice on quadrilaterals; it is recorded by
:pyattr:`per_variable_exactness` rather than smuggled into the total-degree
number, which would break the meaning the rest of the library relies on.

Why this does not work on a triangle
------------------------------------
The construction needs the integration limits to be independent, which they are
not on a simplex: the inner limit depends on the outer variable. Collapsing a
square onto a triangle (the Duffy transformation) restores separability but
introduces a Jacobian that clusters points at the collapsed edge and costs
exactness, which is why simplices get their own families instead.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.quadrature.enums import QuadratureFamily
from nanofem.numerics.quadrature.gauss import GaussLegendreQuadrature, GaussLobattoQuadrature
from nanofem.numerics.quadrature.rules import QuadratureRule
from nanofem.numerics.reference.element import ReferenceElement
from nanofem.numerics.reference.enums import CellType
from nanofem.numerics.reference.registry import reference_element
from nanofem.utils.exceptions import InputValidationError

#: The one-dimensional families a tensor product can be built from.
_LINE_FAMILIES: dict[QuadratureFamily, type[QuadratureRule]] = {
    QuadratureFamily.GAUSS_LEGENDRE: GaussLegendreQuadrature,
    QuadratureFamily.GAUSS_LOBATTO: GaussLobattoQuadrature,
}

#: Reference domains that are products of intervals.
_PRODUCT_CELLS: frozenset[CellType] = frozenset(
    {CellType.LINE, CellType.QUADRILATERAL, CellType.HEXAHEDRON}
)


@dataclass(frozen=True, eq=False, repr=False)
class TensorProductQuadrature(QuadratureRule):
    """A rule on a product domain, built from one one-dimensional rule per axis.

    Parameters
    ----------
    reference
        The product domain, or its cell type / name.
    order
        The polynomial degree to integrate exactly, applied to every axis.
    base_family
        Which one-dimensional family to build from.
    """

    reference_: ReferenceElement
    factors_: tuple[QuadratureRule, ...]
    order_: int

    def __init__(
        self,
        reference: ReferenceElement | CellType | str,
        order: int,
        base_family: QuadratureFamily = QuadratureFamily.GAUSS_LEGENDRE,
    ) -> None:
        """Build an isotropic product rule of the requested degree on every axis."""
        element = _resolve(reference)
        factory = _LINE_FAMILIES.get(base_family)
        if factory is None:
            available = ", ".join(sorted(f.value for f in _LINE_FAMILIES))
            raise InputValidationError(
                f"a tensor product needs a one-dimensional family to build from; "
                f"{base_family.value!r} is not one. Available: {available}"
            )
        factors = tuple(
            factory(order) for _ in range(element.topological_dimension)  # type: ignore[call-arg]
        )
        _bind(self, element, factors, order)

    @classmethod
    def from_rules(
        cls,
        reference: ReferenceElement | CellType | str,
        rules: Sequence[QuadratureRule],
    ) -> TensorProductQuadrature:
        """Compose one rule per axis, which may differ - an anisotropic product.

        The general form of the construction: nothing requires the axes to carry
        the same family or the same order. A field that is cubic in one
        direction and linear in the other can be integrated exactly without
        paying for the cross terms, which is what rule composition is for.
        """
        element = _resolve(reference)
        factors = tuple(rules)
        if len(factors) != element.topological_dimension:
            raise InputValidationError(
                f"a {element.cell_type.value} needs {element.topological_dimension} "
                f"one-dimensional rules, got {len(factors)}"
            )
        for index, rule in enumerate(factors):
            if rule.topological_dimension != 1:
                raise InputValidationError(
                    f"factor {index} is a {rule.topological_dimension}-D rule; a tensor "
                    f"product composes one-dimensional rules"
                )
        instance = cls.__new__(cls)
        _bind(instance, element, factors, min(rule.order for rule in factors))
        return instance

    # ---- declarations -------------------------------------------------------

    @property
    def family(self) -> QuadratureFamily:
        """The tensor-product construction."""
        return QuadratureFamily.TENSOR_PRODUCT

    @property
    def reference_element(self) -> ReferenceElement:
        """The product domain."""
        return self.reference_

    @property
    def order(self) -> int:
        """The requested exactness degree."""
        return self.order_

    @property
    def factors(self) -> tuple[QuadratureRule, ...]:
        """The one-dimensional rules this product was composed from, axis by axis."""
        return self.factors_

    @property
    def base_families(self) -> tuple[QuadratureFamily, ...]:
        """The family used on each axis."""
        return tuple(rule.family for rule in self.factors_)

    @property
    def declares_symmetry(self) -> bool:
        """Only an isotropic product claims the domain's full symmetry.

        Swapping two axes of the square maps the rule onto a product with those
        factors exchanged, which is the same rule only when the factors are
        equal. An anisotropic product keeps the axis reflections and loses the
        diagonals - a true statement about a legitimate rule, not a defect.
        """
        return all(rule == self.factors_[0] for rule in self.factors_)

    @property
    def per_variable_exactness(self) -> tuple[int, ...]:
        """Each axis's exactness degree in its own variable.

        The product is exact on the full ``Q`` space these bound, which is
        strictly larger than the ``P`` space :pyattr:`exactness_degree` reports.
        """
        return tuple(rule.exactness_degree for rule in self.factors_)

    @property
    def exactness_degree(self) -> int:
        """The total-degree exactness: the weakest axis, since ``xi^(p+1)`` fails."""
        return min(self.per_variable_exactness)

    @cached_property
    def _rule(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The cartesian product of the factors' points, with multiplied weights.

        Ordered with the last axis varying fastest, which makes the point list
        of a square read row by row.
        """
        grids = [rule.points.ravel() for rule in self.factors_]
        weight_sets = [rule.weights for rule in self.factors_]
        points = np.array(list(itertools.product(*grids)), dtype=np.float64)
        weights = np.array(
            [float(np.prod(combination)) for combination in itertools.product(*weight_sets)],
            dtype=np.float64,
        )
        points = np.ascontiguousarray(points)
        points.setflags(write=False)
        weights.setflags(write=False)
        return points, weights

    @property
    def points(self) -> NDArray[np.float64]:
        """The product grid, shape ``(prod n_k, dim)``, read-only."""
        return self._rule[0]

    @property
    def weights(self) -> NDArray[np.float64]:
        """The products of the factors' weights, read-only."""
        return self._rule[1]

    # ---- verification -------------------------------------------------------

    def verify_tensor_product_construction(self) -> None:
        """Check the product really is the product, factor by factor.

        Every factor must be a valid rule in its own right, the point count must
        be the product of the factors' counts, and - the identity that matters -
        the product's moment of a separable monomial must equal the product of
        the factors' moments. That last one is the construction's whole claim,
        and it is checked against the factors rather than against the closed
        forms, so it is a different statement from exactness.
        """
        for index, rule in enumerate(self.factors_):
            rule.verify()
            if rule.topological_dimension != 1:
                raise InputValidationError(f"factor {index} is not one-dimensional")
        expected = int(np.prod([rule.num_points for rule in self.factors_]))
        if self.num_points != expected:
            raise InputValidationError(
                f"{self._label()}: {self.num_points} points, expected {expected} from the "
                f"factors"
            )
        for exponents in itertools.product(
            range(self.exactness_degree + 1), repeat=len(self.factors_)
        ):
            product = 1.0
            for rule, power in zip(self.factors_, exponents, strict=True):
                product *= rule.moment((power,))
            combined = self.moment(exponents)
            if not np.isclose(combined, product, atol=1e-12):
                raise InputValidationError(
                    f"{self._label()}: the product's moment of {exponents} is "
                    f"{combined:.16g} but the factors give {product:.16g}"
                )

    def verify(self) -> None:
        """Run the generic identities plus the construction check."""
        super().verify()
        self.verify_tensor_product_construction()


def _resolve(reference: ReferenceElement | CellType | str) -> ReferenceElement:
    """Accept a reference element, a cell type, or a name; require a product domain."""
    element = reference if isinstance(reference, ReferenceElement) else reference_element(reference)
    if element.cell_type not in _PRODUCT_CELLS:
        available = ", ".join(sorted(cell.value for cell in _PRODUCT_CELLS))
        raise InputValidationError(
            f"a tensor product needs a domain that is a product of intervals; the "
            f"{element.cell_type.value} is a simplex, whose integration limits are "
            f"coupled. Product domains: {available}"
        )
    return element


def _bind(
    instance: TensorProductQuadrature,
    element: ReferenceElement,
    factors: tuple[QuadratureRule, ...],
    order: int,
) -> None:
    """Set the frozen fields of a product rule."""
    object.__setattr__(instance, "reference_", element)
    object.__setattr__(instance, "factors_", factors)
    object.__setattr__(instance, "order_", order)
