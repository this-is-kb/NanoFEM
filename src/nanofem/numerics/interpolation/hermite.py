"""The Hermite interpolation family (SDS 2.4).

Hermite elements attach derivative functionals alongside point values, which
is what buys C1 continuity: neighbouring cells share not only nodal values
but nodal derivatives, so the assembled field has a continuous gradient. This
is the machinery architecture decision D3 provisions for - Euler-Bernoulli
beams need it now, strain-gradient elasticity needs it later.

Implemented elements
--------------------
- **Cubic Hermite line** (``order=3``): DOFs ``u(x_i)`` and ``du/dxi (x_i)``
  at the two vertices, spanning ``P_3`` (dimension 4). C1. This is the
  classical Euler-Bernoulli beam interpolation.
- **Bicubic Hermite quadrilateral** (``order=3``): DOFs ``u``, ``du/dxi``,
  ``du/deta``, and the cross derivative ``d2u/dxi deta`` at each of the four
  vertices, spanning ``Q_3`` (dimension 16). This is the Bogner-Fox-Schmit
  element. It delivers C1 on rectangular (axis-aligned, affinely mapped)
  meshes; on generally distorted quadrilaterals the cross-derivative DOFs no
  longer match across a facet and only C0 survives. The declared continuity
  is the element's *design* class, realized when that mesh condition holds.

Why there is no Hermite triangle
--------------------------------
Requesting one raises, with the reason, rather than silently returning
something weaker than a caller would assume. The natural candidate - the
10-DOF cubic Hermite triangle (vertex values, two vertex derivatives each,
one centroid value on ``P_3``) - is unisolvent but only **C0**: the normal
derivative along an edge is a quadratic in the edge parameter, needing three
conditions, while only the two endpoint normal derivatives are shared. C1 on
simplices therefore requires either the Argyris element (quintic ``P_5``, 21
DOFs, with second derivatives and edge normal derivatives) or a macroelement
such as Hsieh-Clough-Tocher. Both are real future work with a named landing
zone; neither is a cubic Hermite triangle.

This module declares metadata only: no shape function is written down here.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cached_property

from nanofem.numerics.interpolation.base import Interpolation
from nanofem.numerics.interpolation.dofs import DofFunctional, InterpolationNode
from nanofem.numerics.interpolation.enums import InterpolationFamily
from nanofem.numerics.interpolation.polynomial import PolynomialSpace
from nanofem.numerics.operators.base import Continuity
from nanofem.numerics.reference.element import ReferenceElement
from nanofem.numerics.reference.enums import CellType, EntityType
from nanofem.numerics.reference.registry import reference_element
from nanofem.utils.exceptions import InputValidationError

#: Orders implemented for each cell shape.
_SUPPORTED_ORDERS: dict[CellType, tuple[int, ...]] = {
    CellType.LINE: (3,),
    CellType.QUADRILATERAL: (3,),
}

#: Why the triangle is absent, stated where a caller will meet it.
_NO_TRIANGLE_REASON = (
    "Hermite interpolation on a triangle is not implemented: the 10-dof cubic Hermite "
    "triangle is unisolvent but only C0 (an edge normal derivative is quadratic, yet only "
    "its two endpoint values are shared), so it does not deliver what a Hermite element is "
    "chosen for. C1 on simplices needs the Argyris quintic (21 dofs) or an HCT "
    "macroelement, both of which are future work"
)


@dataclass(frozen=True, eq=False, repr=False)
class HermiteInterpolation(Interpolation):
    """Hermite interpolation with derivative degrees of freedom at the vertices.

    Parameters
    ----------
    cell_type
        The reference shape: line or quadrilateral.
    order
        The polynomial order; 3 for both implemented elements.
    """

    cell_type_: CellType
    order_: int

    def __init__(self, cell_type: CellType | str, order: int = 3) -> None:
        """Validate the requested cell/order and freeze the element."""
        resolved = cell_type if isinstance(cell_type, CellType) else CellType(cell_type)
        if resolved is CellType.TRIANGLE:
            raise NotImplementedError(_NO_TRIANGLE_REASON)
        supported = _SUPPORTED_ORDERS.get(resolved)
        if supported is None:
            available = ", ".join(sorted(ct.value for ct in _SUPPORTED_ORDERS))
            raise InputValidationError(
                f"Hermite interpolation is not implemented on {resolved.value!r}; "
                f"available shapes: {available}"
            )
        if order not in supported:
            raise InputValidationError(
                f"Hermite order {order} on {resolved.value} is not implemented; "
                f"supported orders: {supported}"
            )
        object.__setattr__(self, "cell_type_", resolved)
        object.__setattr__(self, "order_", order)

    # ---- declarations -------------------------------------------------------

    @property
    def family(self) -> InterpolationFamily:
        """The Hermite family."""
        return InterpolationFamily.HERMITE

    @property
    def order(self) -> int:
        """The polynomial order ``k``."""
        return self.order_

    @property
    def cell_type(self) -> CellType:
        """The reference shape."""
        return self.cell_type_

    @cached_property
    def reference_element(self) -> ReferenceElement:
        """The reference domain, from the phase-2 registry."""
        return reference_element(self.cell_type_)

    @cached_property
    def polynomial_space(self) -> PolynomialSpace:
        """``P_3`` on the line, ``Q_3`` (bicubic) on the quadrilateral."""
        dimension = self.reference_element.topological_dimension
        if self.cell_type_ is CellType.QUADRILATERAL:
            return PolynomialSpace.tensor_product(self.order_, dimension)
        return PolynomialSpace.total_degree(self.order_, dimension)

    @cached_property
    def nodes(self) -> tuple[InterpolationNode, ...]:
        """The vertices; Hermite adds derivative DOFs rather than extra points."""
        return tuple(
            InterpolationNode(
                index=index,
                coordinates=tuple(float(x) for x in coordinates),
                entity=EntityType.VERTEX,
                entity_index=index,
            )
            for index, coordinates in enumerate(self.reference_element.vertex_coordinates)
        )

    @cached_property
    def dofs(self) -> tuple[DofFunctional, ...]:
        """Value and derivative functionals at each vertex, grouped per vertex.

        The line carries ``(u, du/dxi)`` - the classical beam ordering
        ``(w1, theta1, w2, theta2)``. The quadrilateral carries
        ``(u, du/dxi, du/deta, d2u/dxi deta)``, the Bogner-Fox-Schmit set.
        """
        multi_indices = _derivative_multi_indices(self.reference_element.topological_dimension)
        dofs: list[DofFunctional] = []
        for node in self.nodes:
            for local, derivative in enumerate(multi_indices):
                dofs.append(
                    DofFunctional(
                        index=len(dofs),
                        node_index=node.index,
                        point=node.coordinates,
                        derivative=derivative,
                        entity=node.entity,
                        entity_index=node.entity_index,
                        local_index=local,
                    )
                )
        return tuple(dofs)

    @property
    def continuity(self) -> Continuity:
        """C1 by design: shared nodal values and derivatives match the gradient.

        Realized on the line unconditionally, and on the quadrilateral for
        rectangular (affinely mapped) meshes; see the module docstring.
        """
        return Continuity.C1


def _derivative_multi_indices(dimension: int) -> tuple[tuple[int, ...], ...]:
    """The Hermite functional multi-indices for a cell of the given dimension.

    One variable gives ``(0,), (1,)`` - value and slope. Two variables give
    the tensor product ``(0,0), (1,0), (0,1), (1,1)`` - value, both first
    derivatives, and the cross derivative, which is exactly the
    Bogner-Fox-Schmit set and exactly what the tensor product of two 1-D
    cubic Hermite elements requires.

    Each tuple is reversed because ``itertools.product`` varies the *last*
    index fastest, which would order the quadrilateral functionals
    ``u, d/deta, d/dxi, cross``; the conventional order is
    ``u, d/dxi, d/deta, cross``.
    """
    return tuple(tuple(reversed(alpha)) for alpha in itertools.product((0, 1), repeat=dimension))
