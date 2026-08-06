"""Strongly typed enumerations for the reference element library.

These enums are the shared vocabulary of the reference geometry layer: cell
shapes, entity roles, facet shapes, sub-entity orientations, and spatial
dimension. They are deliberately free of any finite element mathematics -
they name topological and geometric concepts only.

Design notes
------------
- ``Dimension`` is an ``IntEnum`` so it compares and arithmetics as a plain
  integer while remaining a named, bounded type.
- ``CellType`` carries a ``topological_dimension`` and an ``is_implemented``
  flag so callers (and the registry) can reason about which shapes exist in
  this phase without a separate table.
- ``EntityType`` names entities by their role/absolute dimension (vertex 0,
  edge 1, face 2, and the cell itself whose dimension is contextual). The
  codimension-1 role ("facet") is not a fixed absolute dimension - it maps to
  vertex/edge/face depending on the cell - so it is exposed per element via
  ``ReferenceElement.facet_type`` rather than as an ``EntityType`` member.
- ``Orientation`` covers the only non-trivial case needed for 1-D facets
  (edges): forward (canonical) or reversed. Higher orientation groups for the
  2-D faces of 3-D cells (rotations and reflections) arrive with the volume
  elements and will extend this enum.
"""

from __future__ import annotations

from enum import Enum, IntEnum, unique


@unique
class Dimension(IntEnum):
    """Spatial dimension as a bounded, named integer type."""

    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3


@unique
class CellType(Enum):
    """The topological shape of a reference cell.

    ``VERTEX`` (a 0-D point) is included because it is the facet shape of a
    line; the 3-D shapes are declared now so interfaces, registries, and enums
    anticipate them, but only line/triangle/quadrilateral are implemented in
    this phase (see :pyattr:`is_implemented`).
    """

    VERTEX = "vertex"
    LINE = "line"
    TRIANGLE = "triangle"
    QUADRILATERAL = "quadrilateral"
    TETRAHEDRON = "tetrahedron"
    HEXAHEDRON = "hexahedron"
    PRISM = "prism"
    PYRAMID = "pyramid"

    @property
    def topological_dimension(self) -> int:
        """Intrinsic dimension of this shape (point 0, line 1, surface 2, volume 3)."""
        return _CELL_DIMENSION[self]

    @property
    def is_implemented(self) -> bool:
        """Whether a concrete ``ReferenceElement`` exists for this shape in this phase."""
        return self in _IMPLEMENTED_CELL_TYPES


_CELL_DIMENSION: dict[CellType, int] = {
    CellType.VERTEX: 0,
    CellType.LINE: 1,
    CellType.TRIANGLE: 2,
    CellType.QUADRILATERAL: 2,
    CellType.TETRAHEDRON: 3,
    CellType.HEXAHEDRON: 3,
    CellType.PRISM: 3,
    CellType.PYRAMID: 3,
}

_IMPLEMENTED_CELL_TYPES: frozenset[CellType] = frozenset(
    {CellType.LINE, CellType.TRIANGLE, CellType.QUADRILATERAL}
)


@unique
class EntityType(Enum):
    """An entity by its role/absolute dimension within a cell's boundary lattice."""

    VERTEX = "vertex"
    EDGE = "edge"
    FACE = "face"
    CELL = "cell"

    @property
    def absolute_dimension(self) -> int | None:
        """Absolute dimension (vertex 0, edge 1, face 2); ``None`` for the cell.

        The cell entity's dimension equals the cell's own topological dimension
        and is therefore contextual, reported as ``None`` here.
        """
        return _ENTITY_DIMENSION[self]


_ENTITY_DIMENSION: dict[EntityType, int | None] = {
    EntityType.VERTEX: 0,
    EntityType.EDGE: 1,
    EntityType.FACE: 2,
    EntityType.CELL: None,
}


@unique
class FacetType(Enum):
    """The shape of a codimension-1 entity (facet).

    A line's facets are vertices; a 2-D cell's facets are lines (edges); a 3-D
    cell's facets are triangles or quadrilaterals.
    """

    VERTEX = "vertex"
    LINE = "line"
    TRIANGLE = "triangle"
    QUADRILATERAL = "quadrilateral"


@unique
class Orientation(Enum):
    """Relative orientation of a shared sub-entity against its canonical listing.

    ``FORWARD`` is the identity (vertices in listed order); ``REVERSED`` flips
    the order (the only non-trivial orientation for a 1-D facet). The richer
    orientation group for 2-D faces of 3-D cells is a future extension.
    """

    FORWARD = "forward"
    REVERSED = "reversed"

    @property
    def sign(self) -> int:
        """+1 for forward, -1 for reversed (handy for tangent/normal bookkeeping)."""
        return 1 if self is Orientation.FORWARD else -1
