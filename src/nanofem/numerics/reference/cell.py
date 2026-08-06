"""Reference cells: single source of truth for topology and orientation (SDS 2.3, C-3).

Line: xi in [-1, 1]. Triangle: (0,0), (1,0), (0,1), counterclockwise; facet i
opposite vertex i. Quad: [-1, 1]^2, facets bottom/right/top/left. Outward
normals. No module may embed its own copies of these conventions; the mesh
validates connectivity against this registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from nanofem.utils.exceptions import InputValidationError


@dataclass(frozen=True)
class ReferenceCell:
    """Topology of one reference cell; hashable (cache key for tabulated data)."""

    name: str
    dimension: int
    num_nodes: int
    num_vertices: int
    num_facets: int
    measure: float


#: Registered mesh cell types, keyed by name. The suffix is the *node* count, so
#: one reference domain hosts several entries: "tri3" and "tri6" are the same
#: triangle at different interpolation orders. "quad8" is the serendipity cell;
#: "quad9" is the Lagrange biquadratic one.
REFERENCE_CELLS: dict[str, ReferenceCell] = {
    "line2": ReferenceCell("line2", 1, 2, 2, 2, 2.0),
    "line3": ReferenceCell("line3", 1, 3, 2, 2, 2.0),
    "line4": ReferenceCell("line4", 1, 4, 2, 2, 2.0),
    "tri3": ReferenceCell("tri3", 2, 3, 3, 3, 0.5),
    "tri6": ReferenceCell("tri6", 2, 6, 3, 3, 0.5),
    "tri10": ReferenceCell("tri10", 2, 10, 3, 3, 0.5),
    "quad4": ReferenceCell("quad4", 2, 4, 4, 4, 4.0),
    "quad8": ReferenceCell("quad8", 2, 8, 4, 4, 4.0),
    "quad9": ReferenceCell("quad9", 2, 9, 4, 4, 4.0),
    "quad16": ReferenceCell("quad16", 2, 16, 4, 4, 4.0),
}


def reference_cell(name: str) -> ReferenceCell:
    """Return the registered reference cell; unknown names list what exists."""
    try:
        return REFERENCE_CELLS[name]
    except KeyError:
        known = ", ".join(sorted(REFERENCE_CELLS))
        raise InputValidationError(f"unknown cell type {name!r}; registered: {known}") from None
