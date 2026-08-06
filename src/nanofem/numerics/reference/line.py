"""The reference line element.

Reference domain: ``xi`` in ``[-1, 1]``. Vertex 0 sits at ``-1`` and vertex 1
at ``+1``. The line's facets are its two vertices; its single edge is the cell
itself. All geometry and topology are inherited from
:class:`~nanofem.numerics.reference.element.ReferenceElement`; this module only
declares the canonical data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.reference.element import ReferenceElement
from nanofem.numerics.reference.enums import CellType, FacetType

_LINE_VERTICES: NDArray[np.float64] = np.array([[-1.0], [1.0]], dtype=np.float64)
_LINE_VERTICES.setflags(write=False)

#: Facets are the two vertices (codimension-1 entities of a line).
_LINE_FACETS: tuple[tuple[int, ...], ...] = ((0,), (1,))

#: The single edge is the cell itself, spanning both vertices.
_LINE_EDGES: tuple[tuple[int, ...], ...] = ((0, 1),)


@dataclass(frozen=True, eq=False, repr=False)
class ReferenceLine(ReferenceElement):
    """The bi-unit reference line ``[-1, 1]`` (2 vertices, 1 edge, 2 vertex facets)."""

    @property
    def cell_type(self) -> CellType:
        """The line cell type."""
        return CellType.LINE

    @property
    def facet_type(self) -> FacetType:
        """A line's facets are vertices."""
        return FacetType.VERTEX

    @property
    def vertex_coordinates(self) -> NDArray[np.float64]:
        """Vertices at ``xi = -1`` and ``xi = +1``, shape ``(2, 1)``, read-only."""
        return _LINE_VERTICES

    @property
    def facet_vertex_indices(self) -> tuple[tuple[int, ...], ...]:
        """Each facet is a single vertex."""
        return _LINE_FACETS

    @property
    def edge_vertex_indices(self) -> tuple[tuple[int, ...], ...]:
        """The one edge spans vertices 0 and 1."""
        return _LINE_EDGES

    @property
    def reference_measure(self) -> float:
        """Length of ``[-1, 1]``."""
        return 2.0
