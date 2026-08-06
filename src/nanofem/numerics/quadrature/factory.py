"""Rule selection by (cell, order); reduced rules only by explicit policy (SDS 2.5).

The policy the phase-0 seam declared, made real: **full integration is the
default and implicit reduction is prohibited**. A request for degree ``p``
returns a rule whose exactness is at least ``p``, never less. Reduced
integration - deliberately under-integrating to soften an element's locking - is
a legitimate technique and a documented one, but it is a *choice*, and a caller
must make it by asking for a lower degree rather than receiving one silently.

Choosing the degree an element needs is not this layer's business. The
``m >= 2p`` rule of thumb, and its mapping-adjusted refinements, belong to the
element layer, which knows the polynomial order of what it is integrating. This
factory only guarantees that what was asked for is what is delivered.
"""

from __future__ import annotations

from functools import lru_cache

from nanofem.numerics.quadrature.dunavant import DunavantQuadrature
from nanofem.numerics.quadrature.enums import QuadratureFamily
from nanofem.numerics.quadrature.gauss import GaussLegendreQuadrature, GaussLobattoQuadrature
from nanofem.numerics.quadrature.rules import QuadratureRule
from nanofem.numerics.quadrature.tensor import TensorProductQuadrature
from nanofem.numerics.reference.cell import ReferenceCell
from nanofem.numerics.reference.element import ReferenceElement
from nanofem.numerics.reference.enums import CellType
from nanofem.numerics.reference.registry import cell_type_of_name
from nanofem.utils.exceptions import InputValidationError

#: The default family for each reference domain: Gauss on the line, its tensor
#: product on the square, Dunavant on the triangle - in each case the fewest
#: points reaching a given degree among the implemented families.
DEFAULT_FAMILIES: dict[CellType, QuadratureFamily] = {
    CellType.LINE: QuadratureFamily.GAUSS_LEGENDRE,
    CellType.QUADRILATERAL: QuadratureFamily.TENSOR_PRODUCT,
    CellType.TRIANGLE: QuadratureFamily.DUNAVANT,
}

#: Which families are available on which domains.
AVAILABLE_QUADRATURES: dict[QuadratureFamily, frozenset[CellType]] = {
    QuadratureFamily.GAUSS_LEGENDRE: frozenset({CellType.LINE}),
    QuadratureFamily.GAUSS_LOBATTO: frozenset({CellType.LINE}),
    QuadratureFamily.TENSOR_PRODUCT: frozenset({CellType.LINE, CellType.QUADRILATERAL}),
    QuadratureFamily.DUNAVANT: frozenset({CellType.TRIANGLE}),
}


def quadrature(
    cell: ReferenceElement | CellType | str,
    order: int,
    family: QuadratureFamily | str | None = None,
) -> QuadratureRule:
    """Return a rule on ``cell`` integrating degree ``order`` exactly.

    With ``family`` omitted, the default for the domain is used. The returned
    rule's exactness is at least ``order`` - never less, per the SDS 2.5
    prohibition on implicit reduction.

    Rules are immutable value objects with read-only arrays, so identical
    requests return the identical cached instance.
    """
    resolved_cell = _resolve_cell(cell)
    resolved_family = _resolve_family(family, resolved_cell)
    return _build(resolved_family, resolved_cell, order)


def available_quadratures() -> tuple[tuple[QuadratureFamily, CellType], ...]:
    """Every (family, domain) combination this phase implements."""
    return tuple(
        (family, cell)
        for family, cells in AVAILABLE_QUADRATURES.items()
        for cell in sorted(cells, key=lambda c: c.value)
    )


class QuadratureFactory:
    """Full integration default (m >= 2p, mapping-adjusted); implicit reduction prohibited.

    The phase-0 seam, filled. Its ``cell`` argument is a ``ReferenceCell`` - a
    mesh-name record - because that is what the declaration said and what the
    mesh layer will hold; :func:`quadrature` is the same thing for callers who
    already have a reference element.
    """

    def __init__(self, family: QuadratureFamily | None = None) -> None:
        """Optionally pin a family; otherwise each domain's default is used."""
        self._family = family

    def rule(self, cell: ReferenceCell, order: int) -> QuadratureRule:
        """Return a rule of at least the requested exactness for the cell."""
        cell_type = cell_type_of_name(cell.name)
        family = _resolve_family(self._family, cell_type)
        built = _build(family, cell_type, order)
        if built.exactness_degree < order:
            raise InputValidationError(
                f"{built._label()} delivers degree {built.exactness_degree} but {order} was "
                f"requested; implicit reduction is prohibited"
            )
        return built


@lru_cache(maxsize=256)
def _build(family: QuadratureFamily, cell: CellType, order: int) -> QuadratureRule:
    """Construct and memoize a rule.

    Memoized because a rule is a frozen value object with read-only arrays: two
    callers asking for the same rule can safely receive the same instance, and
    the Legendre root-finding behind a high-order rule is worth doing once. The
    cache is a pure memo on a pure function of immutable arguments - it changes
    nothing about the result.
    """
    allowed = AVAILABLE_QUADRATURES.get(family)
    if allowed is None:
        implemented = ", ".join(sorted(f.value for f in AVAILABLE_QUADRATURES))
        raise NotImplementedError(
            f"the {family.value} family is not implemented; available: {implemented}"
        )
    if cell not in allowed:
        domains = ", ".join(sorted(c.value for c in allowed))
        raise InputValidationError(
            f"the {family.value} family is not defined on the {cell.value}; it applies to: "
            f"{domains}"
        )
    if family is QuadratureFamily.GAUSS_LEGENDRE:
        return GaussLegendreQuadrature(order)
    if family is QuadratureFamily.GAUSS_LOBATTO:
        return GaussLobattoQuadrature(order)
    if family is QuadratureFamily.DUNAVANT:
        return DunavantQuadrature(order)
    return TensorProductQuadrature(cell, order)


def _resolve_cell(cell: ReferenceElement | CellType | str) -> CellType:
    """Accept a reference element, a cell type, a cell-type name, or a mesh name."""
    if isinstance(cell, ReferenceElement):
        return cell.cell_type
    if isinstance(cell, CellType):
        return cell
    try:
        return CellType(cell)
    except ValueError:
        return cell_type_of_name(cell)


def _resolve_family(family: QuadratureFamily | str | None, cell: CellType) -> QuadratureFamily:
    """Pick the requested family, or the domain's default."""
    if family is None:
        default = DEFAULT_FAMILIES.get(cell)
        if default is None:
            domains = ", ".join(sorted(c.value for c in DEFAULT_FAMILIES))
            raise InputValidationError(
                f"no quadrature is implemented on the {cell.value}; available domains: "
                f"{domains}"
            )
        return default
    if isinstance(family, QuadratureFamily):
        return family
    try:
        return QuadratureFamily(family)
    except ValueError:
        known = ", ".join(sorted(f.value for f in QuadratureFamily))
        raise InputValidationError(
            f"unknown quadrature family {family!r}; known families: {known}"
        ) from None
