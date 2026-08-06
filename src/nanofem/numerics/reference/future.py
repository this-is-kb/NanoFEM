"""Declared placeholders for the 3-D reference elements.

These shapes are named now so registries, enumerations, and interfaces
anticipate them, but they are not implemented in this phase: constructing one
raises ``NotImplementedError``. Their intended canonical topology is recorded
in each class's ``PROVISIONAL_TOPOLOGY`` mapping and docstring so the eventual
implementation has a fixed target. No finite element mathematics is implied.

Building these out requires the 3-D machinery the 1-D/2-D layer defers:
volume measures, face enumeration (dimension-2 sub-entities), triangular and
quadrilateral facet normals via cross products, and the larger orientation
groups (rotations and reflections) of 2-D faces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from numpy.typing import NDArray

from nanofem.numerics.reference.element import ReferenceElement
from nanofem.numerics.reference.enums import CellType, FacetType


@dataclass(frozen=True, eq=False)
class _FutureReferenceElement(ReferenceElement):
    """Shared placeholder base: satisfies the interface but refuses construction."""

    #: Intended canonical topology, for documentation and the future build target.
    PROVISIONAL_TOPOLOGY: ClassVar[dict[str, Any]] = {}

    def __post_init__(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} is a declared placeholder; 3-D reference elements "
            "arrive in a later phase (intended topology in PROVISIONAL_TOPOLOGY)"
        )

    @property
    def cell_type(self) -> CellType:  # pragma: no cover - construction raises first
        """Not available for a placeholder."""
        raise NotImplementedError

    @property
    def facet_type(self) -> FacetType:  # pragma: no cover - construction raises first
        """Not available for a placeholder."""
        raise NotImplementedError

    @property
    def vertex_coordinates(self) -> NDArray[Any]:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    @property
    def facet_vertex_indices(self) -> tuple[tuple[int, ...], ...]:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    @property
    def edge_vertex_indices(self) -> tuple[tuple[int, ...], ...]:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    @property
    def reference_measure(self) -> float:  # pragma: no cover - construction raises first
        """Not available for a placeholder."""
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class ReferenceTetrahedron(_FutureReferenceElement):
    """Placeholder for the unit tetrahedron.

    Intended: vertices ``(0,0,0), (1,0,0), (0,1,0), (0,0,1)``; 4 vertices,
    6 edges, 4 triangular faces (= facets); measure ``1/6``.
    """

    PROVISIONAL_TOPOLOGY: ClassVar[dict[str, Any]] = {
        "cell_type": CellType.TETRAHEDRON.value,
        "num_vertices": 4,
        "num_edges": 6,
        "num_faces": 4,
        "num_facets": 4,
        "facet_type": FacetType.TRIANGLE.value,
        "reference_measure": 1.0 / 6.0,
    }


@dataclass(frozen=True, eq=False)
class ReferenceHexahedron(_FutureReferenceElement):
    """Placeholder for the bi-unit hexahedron.

    Intended: the cube ``[-1, 1]^3``; 8 vertices, 12 edges, 6 quadrilateral
    faces (= facets); measure ``8``.
    """

    PROVISIONAL_TOPOLOGY: ClassVar[dict[str, Any]] = {
        "cell_type": CellType.HEXAHEDRON.value,
        "num_vertices": 8,
        "num_edges": 12,
        "num_faces": 6,
        "num_facets": 6,
        "facet_type": FacetType.QUADRILATERAL.value,
        "reference_measure": 8.0,
    }


@dataclass(frozen=True, eq=False)
class ReferencePrism(_FutureReferenceElement):
    """Placeholder for the triangular prism (wedge).

    Intended: a triangular cross-section extruded along an axis; 6 vertices,
    9 edges, 5 faces (2 triangles + 3 quadrilaterals). Facets are of mixed
    shape, so a per-facet facet-type will be needed rather than a single one.
    """

    PROVISIONAL_TOPOLOGY: ClassVar[dict[str, Any]] = {
        "cell_type": CellType.PRISM.value,
        "num_vertices": 6,
        "num_edges": 9,
        "num_faces": 5,
        "num_facets": 5,
        "facet_type": "mixed (triangle and quadrilateral)",
    }


@dataclass(frozen=True, eq=False)
class ReferencePyramid(_FutureReferenceElement):
    """Placeholder for the square-based pyramid.

    Intended: a quadrilateral base with an apex; 5 vertices, 8 edges, 5 faces
    (4 triangles + 1 quadrilateral). Facets are of mixed shape.
    """

    PROVISIONAL_TOPOLOGY: ClassVar[dict[str, Any]] = {
        "cell_type": CellType.PYRAMID.value,
        "num_vertices": 5,
        "num_edges": 8,
        "num_faces": 5,
        "num_facets": 5,
        "facet_type": "mixed (triangle and quadrilateral)",
    }
