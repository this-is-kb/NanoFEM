"""The Lagrange interpolation family (SDS 2.4).

Lagrange elements attach a single point-value degree of freedom to each node,
so their shape functions will satisfy the classical Kronecker delta
``N_i(x_j) = delta_ij`` and form a partition of unity. Their inter-element
continuity is C0: neighbouring cells share the nodes on their common facet,
so the trace matches, but no derivative is constrained.

Spaces and node counts
----------------------
======================  =======  =============  =====  =========
cell                    order    space          nodes  mesh name
======================  =======  =============  =====  =========
line                    1, 2, 3  ``P_k``        k + 1  line{n}
triangle                1, 2, 3  ``P_k``        (k+1)(k+2)/2  tri{n}
quadrilateral           1, 2, 3  ``Q_k``        (k+1)^2  quad{n}
======================  =======  =============  =====  =========

Node placement is the equispaced lattice: the barycentric lattice on the
triangle, the tensor-product grid on the quadrilateral. Equispaced nodes are
the classical choice and are exact for these orders, but the conditioning of
the unisolvence matrix degrades as the order rises (the Runge phenomenon),
which is why high-order work wants the Gauss-Lobatto-Legendre nodes of the
spectral family instead.

Node ordering follows the *reference element's own* entity numbering:
vertices first in reference vertex order, then the nodes interior to edge 0,
edge 1, ... in reference edge order, then the cell interior. Because the
reference element is the single source of topological truth (SDS C-3), this
ordering is derived rather than restated - and it therefore differs from the
orderings baked into external formats such as gmsh or VTK. Translating
between them is the mesh I/O adapter's job (architecture P5), not this
module's.

This module declares metadata only: no shape function is written down here.
"""

from __future__ import annotations

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
    CellType.LINE: (1, 2, 3),
    CellType.TRIANGLE: (1, 2, 3),
    CellType.QUADRILATERAL: (1, 2, 3),
}


@dataclass(frozen=True, eq=False, repr=False)
class LagrangeInterpolation(Interpolation):
    """Equispaced Lagrange interpolation of order ``k`` on a reference cell.

    Parameters
    ----------
    cell_type
        The reference shape: line, triangle, or quadrilateral.
    order
        The polynomial order ``k`` (1, 2, or 3).
    """

    cell_type_: CellType
    order_: int

    def __init__(self, cell_type: CellType | str, order: int) -> None:
        """Validate the requested cell/order and freeze the element."""
        resolved = cell_type if isinstance(cell_type, CellType) else CellType(cell_type)
        supported = _SUPPORTED_ORDERS.get(resolved)
        if supported is None:
            available = ", ".join(sorted(ct.value for ct in _SUPPORTED_ORDERS))
            raise InputValidationError(
                f"Lagrange interpolation is not implemented on {resolved.value!r}; "
                f"available shapes: {available}"
            )
        if order not in supported:
            raise InputValidationError(
                f"Lagrange order {order} on {resolved.value} is not implemented; "
                f"supported orders: {supported}"
            )
        object.__setattr__(self, "cell_type_", resolved)
        object.__setattr__(self, "order_", order)

    # ---- declarations -------------------------------------------------------

    @property
    def family(self) -> InterpolationFamily:
        """The Lagrange family."""
        return InterpolationFamily.LAGRANGE

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
        """``P_k`` on simplices, ``Q_k`` on tensor-product cells."""
        dimension = self.reference_element.topological_dimension
        if self.cell_type_ is CellType.QUADRILATERAL:
            return PolynomialSpace.tensor_product(self.order_, dimension)
        return PolynomialSpace.total_degree(self.order_, dimension)

    @cached_property
    def nodes(self) -> tuple[InterpolationNode, ...]:
        """The equispaced lattice, ordered vertices then edges then interior."""
        if self.cell_type_ is CellType.LINE:
            placements = _line_placements(self.reference_element, self.order_)
        elif self.cell_type_ is CellType.TRIANGLE:
            placements = _triangle_placements(self.reference_element, self.order_)
        else:
            placements = _quadrilateral_placements(self.reference_element, self.order_)
        return tuple(
            InterpolationNode(index, coordinates, entity, entity_index, local_index)
            for index, (coordinates, entity, entity_index, local_index) in enumerate(placements)
        )

    @cached_property
    def dofs(self) -> tuple[DofFunctional, ...]:
        """One point-value functional per node."""
        zero = (0,) * self.reference_element.topological_dimension
        return tuple(
            DofFunctional(
                index=node.index,
                node_index=node.index,
                point=node.coordinates,
                derivative=zero,
                entity=node.entity,
                entity_index=node.entity_index,
                local_index=node.local_index,
            )
            for node in self.nodes
        )

    @property
    def continuity(self) -> Continuity:
        """C0: shared facet nodes make the trace continuous, derivatives free."""
        return Continuity.C0


# ---- node placement (combinatorial geometry, derived from the reference element) ----

_Placement = tuple[tuple[float, ...], EntityType, int, int]


def _vertex_placements(element: ReferenceElement) -> list[_Placement]:
    """Vertex nodes, in the reference element's own vertex order."""
    return [
        (tuple(float(x) for x in coordinates), EntityType.VERTEX, index, 0)
        for index, coordinates in enumerate(element.vertex_coordinates)
    ]


def _edge_placements(element: ReferenceElement, order: int) -> list[_Placement]:
    """Equispaced nodes interior to each edge, in the reference edge order."""
    placements: list[_Placement] = []
    vertices = element.vertex_coordinates
    for edge_index, (start, end) in enumerate(element.edge_vertex_indices):
        for step in range(1, order):
            fraction = step / order
            point = vertices[start] + fraction * (vertices[end] - vertices[start])
            placements.append(
                (tuple(float(x) for x in point), EntityType.EDGE, edge_index, step - 1)
            )
    return placements


def _line_placements(element: ReferenceElement, order: int) -> list[_Placement]:
    """Line lattice: two vertices, then the interior points of the cell itself.

    The line's only edge *is* the cell, so its interior nodes are associated
    with the cell entity rather than an edge.
    """
    placements = _vertex_placements(element)
    vertices = element.vertex_coordinates
    for step in range(1, order):
        fraction = step / order
        point = vertices[0] + fraction * (vertices[1] - vertices[0])
        placements.append((tuple(float(x) for x in point), EntityType.CELL, 0, step - 1))
    return placements


def _triangle_placements(element: ReferenceElement, order: int) -> list[_Placement]:
    """Barycentric lattice: vertices, then edge interiors, then the cell interior."""
    placements = _vertex_placements(element)
    placements += _edge_placements(element, order)
    vertices = element.vertex_coordinates
    local = 0
    for first in range(1, order):
        for second in range(1, order - first):
            third = order - first - second
            if third < 1:
                continue
            point = (
                (third / order) * vertices[0]
                + (first / order) * vertices[1]
                + (second / order) * vertices[2]
            )
            placements.append((tuple(float(x) for x in point), EntityType.CELL, 0, local))
            local += 1
    return placements


def _quadrilateral_placements(element: ReferenceElement, order: int) -> list[_Placement]:
    """Tensor-product grid: vertices, then edge interiors, then the cell interior."""
    placements = _vertex_placements(element)
    placements += _edge_placements(element, order)
    interior = [-1.0 + 2.0 * step / order for step in range(1, order)]
    local = 0
    for eta in interior:
        for xi in interior:
            placements.append(((float(xi), float(eta)), EntityType.CELL, 0, local))
            local += 1
    return placements
