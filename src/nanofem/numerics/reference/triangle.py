"""The reference triangle element.

Reference domain: the unit right triangle with vertices ``(0, 0)``, ``(1, 0)``,
``(0, 1)`` in counterclockwise order. Facet ``i`` is opposite vertex ``i`` (SDS
convention), and each facet's vertices are ordered so its outward normal points
away from the cell:

- facet 0 (opposite vertex 0): vertices ``(1, 2)`` - the hypotenuse,
- facet 1 (opposite vertex 1): vertices ``(2, 0)`` - the left edge,
- facet 2 (opposite vertex 2): vertices ``(0, 1)`` - the bottom edge.

For a 2-D cell the edges and facets coincide. All geometry and topology are
inherited from :class:`~nanofem.numerics.reference.element.ReferenceElement`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.reference.element import ReferenceElement
from nanofem.numerics.reference.enums import CellType, FacetType

_TRIANGLE_VERTICES: NDArray[np.float64] = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64
)
_TRIANGLE_VERTICES.setflags(write=False)

#: Facet i opposite vertex i, endpoints ordered for an outward normal.
_TRIANGLE_FACETS: tuple[tuple[int, ...], ...] = ((1, 2), (2, 0), (0, 1))

#: In 2-D the edges coincide with the facets.
_TRIANGLE_EDGES: tuple[tuple[int, ...], ...] = _TRIANGLE_FACETS


@dataclass(frozen=True, eq=False, repr=False)
class ReferenceTriangle(ReferenceElement):
    """The unit reference triangle (3 vertices, 3 edges = 3 facets, 1 face)."""

    @property
    def cell_type(self) -> CellType:
        """The triangle cell type."""
        return CellType.TRIANGLE

    @property
    def facet_type(self) -> FacetType:
        """A triangle's facets are lines (edges)."""
        return FacetType.LINE

    @property
    def vertex_coordinates(self) -> NDArray[np.float64]:
        """Vertices ``(0,0), (1,0), (0,1)`` counterclockwise, shape ``(3, 2)``, read-only."""
        return _TRIANGLE_VERTICES

    @property
    def facet_vertex_indices(self) -> tuple[tuple[int, ...], ...]:
        """Facet i opposite vertex i, in outward order."""
        return _TRIANGLE_FACETS

    @property
    def edge_vertex_indices(self) -> tuple[tuple[int, ...], ...]:
        """Edges coincide with facets in 2-D."""
        return _TRIANGLE_EDGES

    @property
    def reference_measure(self) -> float:
        """Area of the unit right triangle."""
        return 0.5
