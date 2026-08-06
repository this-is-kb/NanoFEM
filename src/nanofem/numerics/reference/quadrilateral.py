"""The reference quadrilateral element.

Reference domain: the bi-unit square ``[-1, 1]^2`` with vertices in
counterclockwise order:

- vertex 0 ``(-1, -1)``, vertex 1 ``(1, -1)``, vertex 2 ``(1, 1)``,
  vertex 3 ``(-1, 1)``.

Facets are ordered bottom, right, top, left (SDS convention), each ordered so
its outward normal points away from the cell:

- facet 0 (bottom): vertices ``(0, 1)``, outward normal ``(0, -1)``,
- facet 1 (right): vertices ``(1, 2)``, outward normal ``(1, 0)``,
- facet 2 (top): vertices ``(2, 3)``, outward normal ``(0, 1)``,
- facet 3 (left): vertices ``(3, 0)``, outward normal ``(-1, 0)``.

For a 2-D cell the edges and facets coincide. All geometry and topology are
inherited from :class:`~nanofem.numerics.reference.element.ReferenceElement`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.reference.element import ReferenceElement
from nanofem.numerics.reference.enums import CellType, FacetType

_QUADRILATERAL_VERTICES: NDArray[np.float64] = np.array(
    [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=np.float64
)
_QUADRILATERAL_VERTICES.setflags(write=False)

#: Facets ordered bottom, right, top, left; endpoints ordered for outward normals.
_QUADRILATERAL_FACETS: tuple[tuple[int, ...], ...] = ((0, 1), (1, 2), (2, 3), (3, 0))

#: In 2-D the edges coincide with the facets.
_QUADRILATERAL_EDGES: tuple[tuple[int, ...], ...] = _QUADRILATERAL_FACETS


@dataclass(frozen=True, eq=False, repr=False)
class ReferenceQuadrilateral(ReferenceElement):
    """The bi-unit reference quadrilateral (4 vertices, 4 edges = 4 facets, 1 face)."""

    @property
    def cell_type(self) -> CellType:
        """The quadrilateral cell type."""
        return CellType.QUADRILATERAL

    @property
    def facet_type(self) -> FacetType:
        """A quadrilateral's facets are lines (edges)."""
        return FacetType.LINE

    @property
    def vertex_coordinates(self) -> NDArray[np.float64]:
        """Vertices of ``[-1, 1]^2`` counterclockwise, shape ``(4, 2)``, read-only."""
        return _QUADRILATERAL_VERTICES

    @property
    def facet_vertex_indices(self) -> tuple[tuple[int, ...], ...]:
        """Facets ordered bottom, right, top, left, in outward order."""
        return _QUADRILATERAL_FACETS

    @property
    def edge_vertex_indices(self) -> tuple[tuple[int, ...], ...]:
        """Edges coincide with facets in 2-D."""
        return _QUADRILATERAL_EDGES

    @property
    def reference_measure(self) -> float:
        """Area of ``[-1, 1]^2``."""
        return 4.0
