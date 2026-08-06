"""The ``ReferenceElement`` abstract base class.

A reference element is the canonical geometric domain of a finite element -
the unit/bi-unit shape in which interpolation, quadrature, and mapping will
later be defined - together with its topological lattice (vertices, edges,
faces, facets) and orientation conventions. This module implements *only*
geometry and topology: there are no shape functions, no Jacobians, no
quadrature, no interpolation, and no mapping anywhere in this layer.

Design
------
The base class is data-driven. A concrete shape overrides a small set of
abstract members that state its *data*:

- :pyattr:`cell_type`, :pyattr:`facet_type`,
- :pyattr:`vertex_coordinates` (the reference coordinates, one row per vertex),
- :pyattr:`facet_vertex_indices` and :pyattr:`edge_vertex_indices` (incidence),
- :pyattr:`reference_measure` (the canonical length/area/volume).

Every other property and operation - entity counts, centroid, bounding box,
edge lengths, diameter, reference tangents and outward normals, point
containment, distance to the boundary, sub-entity permutation, validation,
and serialization - is a generic algorithm derived from those data and shared
by all shapes. This mirrors how professional libraries (deal.II's
``ReferenceCell``/``GeometryInfo``, MFEM's ``Geometry``) centralize the
canonical data and derive the rest.

Conventions (frozen, per SDS C-3 / 2.3)
---------------------------------------
- Line: xi in [-1, 1]; vertices at -1 and +1.
- Triangle: vertices (0, 0), (1, 0), (0, 1), counterclockwise; facet i is
  opposite vertex i.
- Quadrilateral: [-1, 1]^2; facets ordered bottom, right, top, left.
- Outward facet normals: for a 2-D cell the outward normal of edge (a, b) is
  its tangent rotated clockwise, n = (t_y, -t_x); for a 1-D cell the facet
  (a vertex) points away from the centroid.

Immutability and value semantics
--------------------------------
Concrete reference elements are stateless per shape (all instances of a shape
are identical), so they are frozen dataclasses with no fields. Equality and
hashing are by :pyattr:`cell_type`, defined once here and inherited, so that
``ReferenceTriangle() == ReferenceTriangle()`` and reference elements are
usable as dictionary keys and set members.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.reference.enums import CellType, EntityType, FacetType, Orientation
from nanofem.numerics.reference.errors import OrientationError, TopologyError
from nanofem.utils.exceptions import InputValidationError

#: Tolerance for topological/geometric self-consistency checks (unit reference scale).
_VALIDATION_ATOL: float = 1.0e-12

#: Expected corner-vertex counts, used by validation to catch malformed data.
_EXPECTED_VERTEX_COUNT: dict[CellType, int] = {
    CellType.LINE: 2,
    CellType.TRIANGLE: 3,
    CellType.QUADRILATERAL: 4,
}


class ReferenceElement(ABC):
    """Canonical geometric domain and topological lattice of a finite element.

    Subclasses declare data (coordinates, incidence, measure); this base class
    derives all topological and geometric queries. No finite element
    mathematics (shape functions, Jacobians, quadrature, mapping) lives here.
    """

    # ---- abstract data (declared by each concrete shape) --------------------

    @property
    @abstractmethod
    def cell_type(self) -> CellType:
        """The topological shape of this reference element."""

    @property
    @abstractmethod
    def facet_type(self) -> FacetType:
        """The shape of this element's codimension-1 entities (facets)."""

    @property
    @abstractmethod
    def vertex_coordinates(self) -> NDArray[np.float64]:
        """Reference coordinates of the vertices, shape ``(n_vertices, dim)``, read-only."""

    @property
    @abstractmethod
    def facet_vertex_indices(self) -> tuple[tuple[int, ...], ...]:
        """Vertex indices bounding each facet, in canonical (outward) order."""

    @property
    @abstractmethod
    def edge_vertex_indices(self) -> tuple[tuple[int, ...], ...]:
        """Vertex indices bounding each edge (dimension-1 entity)."""

    @property
    @abstractmethod
    def reference_measure(self) -> float:
        """Canonical measure of the reference domain (length, area, or volume)."""

    # ---- dimensions ---------------------------------------------------------

    @property
    def topological_dimension(self) -> int:
        """Intrinsic dimension of the reference domain (from the coordinate array)."""
        return int(self.vertex_coordinates.shape[1])

    @property
    def embedding_dimension(self) -> int:
        """Dimension of the space the reference domain lives in.

        For a reference element this equals the topological dimension: the
        canonical shapes are defined in their natural space. Embedding a lower
        dimensional element in a higher dimensional physical space (a bar in a
        plane, a shell in 3-D) is a property of the *physical* element and its
        mapping - a later layer - not of the reference element.
        """
        return self.topological_dimension

    @property
    def reference_coordinates(self) -> NDArray[np.float64]:
        """Alias for :pyattr:`vertex_coordinates` (the reference coordinate array)."""
        return self.vertex_coordinates

    # ---- entity counts ------------------------------------------------------

    @property
    def num_vertices(self) -> int:
        """Number of vertices (dimension-0 entities)."""
        return int(self.vertex_coordinates.shape[0])

    @property
    def num_edges(self) -> int:
        """Number of edges (dimension-1 entities); for a line this is 1 (the cell itself)."""
        return len(self.edge_vertex_indices)

    @property
    def num_faces(self) -> int:
        """Number of faces (dimension-2 entities): 0 for a line, 1 for a 2-D cell.

        For 3-D cells this counts the bounding 2-D entities and arrives with the
        volume elements.
        """
        dim = self.topological_dimension
        if dim < 2:
            return 0
        if dim == 2:
            return 1
        raise NotImplementedError("3-D face enumeration arrives with the volume elements")

    @property
    def num_facets(self) -> int:
        """Number of facets (codimension-1 entities)."""
        return len(self.facet_vertex_indices)

    @property
    def facet_entity_type(self) -> EntityType:
        """Which entity role this element's facets play (vertex, edge, or face)."""
        dim = self.topological_dimension
        if dim == 1:
            return EntityType.VERTEX
        if dim == 2:
            return EntityType.EDGE
        return EntityType.FACE

    # ---- geometry -----------------------------------------------------------

    @property
    def centroid(self) -> NDArray[np.float64]:
        """Centroid of the vertices, shape ``(dim,)``."""
        return np.asarray(self.vertex_coordinates.mean(axis=0), dtype=np.float64)

    def bounding_box(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Axis-aligned bounding box as ``(lower, upper)`` corner coordinates."""
        vertices = self.vertex_coordinates
        return (
            np.asarray(vertices.min(axis=0), dtype=np.float64),
            np.asarray(vertices.max(axis=0), dtype=np.float64),
        )

    def edge_lengths(self) -> NDArray[np.float64]:
        """Euclidean length of each edge, shape ``(n_edges,)``."""
        vertices = self.vertex_coordinates
        lengths = [
            float(np.linalg.norm(vertices[b] - vertices[a])) for a, b in self.edge_vertex_indices
        ]
        return np.asarray(lengths, dtype=np.float64)

    @property
    def min_edge_length(self) -> float:
        """Shortest edge length."""
        return float(self.edge_lengths().min())

    @property
    def max_edge_length(self) -> float:
        """Longest edge length."""
        return float(self.edge_lengths().max())

    @property
    def diameter(self) -> float:
        """Largest distance between any two vertices (the cell diameter)."""
        vertices = self.vertex_coordinates
        diffs = vertices[:, None, :] - vertices[None, :, :]
        return float(np.sqrt((diffs**2).sum(axis=-1)).max())

    @property
    def characteristic_length(self) -> float:
        """A single representative size of the reference cell (its diameter)."""
        return self.diameter

    def local_axes(self) -> NDArray[np.float64]:
        """Unit basis vectors of the reference coordinate system, shape ``(dim, dim)``."""
        return np.eye(self.topological_dimension, dtype=np.float64)

    def facet_centroids(self) -> NDArray[np.float64]:
        """Centroid of each facet, shape ``(n_facets, dim)``."""
        vertices = self.vertex_coordinates
        rows = [vertices[list(facet)].mean(axis=0) for facet in self.facet_vertex_indices]
        return np.asarray(rows, dtype=np.float64)

    def reference_tangents(self) -> NDArray[np.float64]:
        """Unit tangent of each edge (from first to second vertex), shape ``(n_edges, dim)``."""
        vertices = self.vertex_coordinates
        rows: list[NDArray[np.float64]] = []
        for a, b in self.edge_vertex_indices:
            tangent = vertices[b] - vertices[a]
            rows.append(tangent / np.linalg.norm(tangent))
        return np.asarray(rows, dtype=np.float64)

    def reference_normals(self) -> NDArray[np.float64]:
        """Outward unit normal of each facet, shape ``(n_facets, dim)``.

        In 1-D the facet is a vertex and its normal points away from the
        centroid; in 2-D the outward normal of edge ``(a, b)`` is the tangent
        rotated clockwise, ``n = (t_y, -t_x)``. 3-D facet normals arrive with
        the volume elements.
        """
        dim = self.topological_dimension
        vertices = self.vertex_coordinates
        if dim == 1:
            centroid = self.centroid
            rows = []
            for (v,) in self.facet_vertex_indices:
                direction = vertices[v] - centroid
                rows.append(direction / np.linalg.norm(direction))
            return np.asarray(rows, dtype=np.float64)
        if dim == 2:
            rows = []
            for a, b in self.facet_vertex_indices:
                tangent = vertices[b] - vertices[a]
                normal = np.array([tangent[1], -tangent[0]], dtype=np.float64)
                rows.append(normal / np.linalg.norm(normal))
            return np.asarray(rows, dtype=np.float64)
        raise NotImplementedError("3-D reference normals arrive with the volume elements")

    # ---- point queries ------------------------------------------------------

    def _as_point(
        self, point: NDArray[np.float64] | tuple[float, ...] | list[float]
    ) -> NDArray[np.float64]:
        """Coerce ``point`` to a float array and validate its dimension."""
        array = np.asarray(point, dtype=np.float64)
        if array.shape != (self.topological_dimension,):
            raise InputValidationError(
                f"point must have shape ({self.topological_dimension},), got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise InputValidationError("point contains non-finite entries")
        return array

    def signed_distance_to_boundary(
        self, point: NDArray[np.float64] | tuple[float, ...] | list[float]
    ) -> float:
        """Signed distance to the boundary: negative inside, zero on it, positive outside.

        Computed from the supporting half-spaces of the (convex) reference cell
        as ``max_i n_i . (p - v_i)`` over facets, with ``n_i`` the outward unit
        normal and ``v_i`` a vertex of facet ``i``. For an interior point this
        is the perpendicular distance to the nearest facet's plane.
        """
        p = self._as_point(point)
        vertices = self.vertex_coordinates
        normals = self.reference_normals()
        best = -np.inf
        for i, facet in enumerate(self.facet_vertex_indices):
            support = float(np.dot(normals[i], p - vertices[facet[0]]))
            best = max(best, support)
        return float(best)

    def distance_to_boundary(
        self, point: NDArray[np.float64] | tuple[float, ...] | list[float]
    ) -> float:
        """Unsigned distance to the boundary (``abs`` of the signed distance)."""
        return abs(self.signed_distance_to_boundary(point))

    def contains(
        self,
        point: NDArray[np.float64] | tuple[float, ...] | list[float],
        tol: float = 1.0e-10,
    ) -> bool:
        """Whether ``point`` lies inside the reference domain (within ``tol``)."""
        return self.signed_distance_to_boundary(point) <= tol

    # ---- orientation --------------------------------------------------------

    def facet_orientations(self) -> tuple[Orientation, ...]:
        """Orientations a facet of this element can take.

        A vertex facet (1-D cell) has a single orientation; an edge facet
        (2-D cell) is forward or reversed. The larger orientation groups of
        2-D faces (3-D cells) arrive with the volume elements.
        """
        arity = len(self.facet_vertex_indices[0])
        if arity == 1:
            return (Orientation.FORWARD,)
        if arity == 2:
            return (Orientation.FORWARD, Orientation.REVERSED)
        raise NotImplementedError("face orientation groups arrive with the volume elements")

    def permute_facet(self, facet_index: int, orientation: Orientation) -> tuple[int, ...]:
        """Vertex indices of a facet reordered for the given orientation.

        ``FORWARD`` returns the canonical order; ``REVERSED`` reverses it (the
        vertex-preserving flip a neighbouring cell sees across a shared edge).
        """
        if not 0 <= facet_index < self.num_facets:
            raise InputValidationError(
                f"facet_index {facet_index} out of range [0, {self.num_facets})"
            )
        if orientation not in self.facet_orientations():
            raise InputValidationError(
                f"{orientation} is not a valid orientation for a {self.facet_type.value} facet"
            )
        vertices = self.facet_vertex_indices[facet_index]
        return vertices if orientation is Orientation.FORWARD else tuple(reversed(vertices))

    # ---- validation ---------------------------------------------------------

    def is_valid(self) -> bool:
        """Return ``True`` if :meth:`validate` passes, ``False`` otherwise."""
        try:
            self.validate()
        except (TopologyError, OrientationError):
            return False
        return True

    def validate(self) -> None:
        """Check topology, geometry, and orientation consistency; raise on any failure.

        Verifies dimension consistency, vertex count and uniqueness, facet and
        edge incidence (valid indices, consistent arity, no duplicates),
        outward orientation of every facet normal, unit-length normals and
        tangents, and agreement between the declared and computed measure.
        """
        self._validate_dimensions()
        self._validate_vertices()
        self._validate_incidence()
        self._validate_orientation()
        self._validate_measure()

    def _validate_dimensions(self) -> None:
        dim = self.topological_dimension
        if not 1 <= dim <= 3:
            raise TopologyError(f"{self.cell_type.value}: topological dimension {dim} not in 1..3")
        if dim != self.cell_type.topological_dimension:
            raise TopologyError(
                f"{self.cell_type.value}: coordinate dimension {dim} disagrees with "
                f"cell-type dimension {self.cell_type.topological_dimension}"
            )
        if self.embedding_dimension != dim:
            raise TopologyError(f"{self.cell_type.value}: embedding dimension must equal {dim}")
        if self.vertex_coordinates.ndim != 2:
            raise TopologyError(f"{self.cell_type.value}: vertex_coordinates must be 2-D")
        if not np.all(np.isfinite(self.vertex_coordinates)):
            raise TopologyError(f"{self.cell_type.value}: vertex coordinates are not all finite")

    def _validate_vertices(self) -> None:
        expected = _EXPECTED_VERTEX_COUNT.get(self.cell_type)
        if expected is not None and self.num_vertices != expected:
            raise TopologyError(
                f"{self.cell_type.value}: expected {expected} vertices, got {self.num_vertices}"
            )
        unique_rows = np.unique(self.vertex_coordinates, axis=0)
        if unique_rows.shape[0] != self.num_vertices:
            raise TopologyError(f"{self.cell_type.value}: duplicate vertex coordinates")

    def _validate_incidence(self) -> None:
        for label, groups, count in (
            ("facet", self.facet_vertex_indices, self.num_facets),
            ("edge", self.edge_vertex_indices, self.num_edges),
        ):
            if len(groups) != count:
                raise TopologyError(f"{self.cell_type.value}: inconsistent {label} count")
            arities = {len(g) for g in groups}
            if len(arities) != 1:
                raise TopologyError(f"{self.cell_type.value}: mixed {label} arities {arities}")
            seen: set[frozenset[int]] = set()
            for group in groups:
                if any(not 0 <= i < self.num_vertices for i in group):
                    raise TopologyError(
                        f"{self.cell_type.value}: {label} {group} references an unknown vertex"
                    )
                if len(set(group)) != len(group):
                    raise TopologyError(f"{self.cell_type.value}: {label} {group} repeats a vertex")
                key = frozenset(group)
                if key in seen:
                    raise TopologyError(
                        f"{self.cell_type.value}: duplicate {label} on vertices {sorted(key)}"
                    )
                seen.add(key)

    def _validate_orientation(self) -> None:
        normals = self.reference_normals()
        facet_centroids = self.facet_centroids()
        centroid = self.centroid
        for i in range(self.num_facets):
            outward = float(np.dot(normals[i], facet_centroids[i] - centroid))
            if outward <= _VALIDATION_ATOL:
                raise OrientationError(
                    f"{self.cell_type.value}: facet {i} normal does not point outward"
                )
        if not np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=_VALIDATION_ATOL):
            raise OrientationError(f"{self.cell_type.value}: facet normals are not unit length")
        tangents = self.reference_tangents()
        if not np.allclose(np.linalg.norm(tangents, axis=1), 1.0, atol=_VALIDATION_ATOL):
            raise OrientationError(f"{self.cell_type.value}: edge tangents are not unit length")

    def _validate_measure(self) -> None:
        if self.reference_measure <= 0.0:
            raise TopologyError(f"{self.cell_type.value}: reference measure must be positive")
        computed = self._computed_measure()
        if not np.isclose(computed, self.reference_measure, atol=_VALIDATION_ATOL):
            raise TopologyError(
                f"{self.cell_type.value}: declared measure {self.reference_measure} "
                f"disagrees with computed measure {computed}"
            )

    def _computed_measure(self) -> float:
        """Measure recomputed from the vertices (length in 1-D, shoelace area in 2-D)."""
        dim = self.topological_dimension
        vertices = self.vertex_coordinates
        if dim == 1:
            return float(np.linalg.norm(vertices[1] - vertices[0]))
        if dim == 2:
            x = vertices[:, 0]
            y = vertices[:, 1]
            shoelace = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
            return float(0.5 * abs(shoelace))
        raise NotImplementedError("3-D volume computation arrives with the volume elements")

    # ---- serialization and utilities ---------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible record of the reference element's geometry and topology."""
        return {
            "schema": "nanofem-reference-element/1",
            "cell_type": self.cell_type.value,
            "facet_type": self.facet_type.value,
            "topological_dimension": self.topological_dimension,
            "embedding_dimension": self.embedding_dimension,
            "num_vertices": self.num_vertices,
            "num_edges": self.num_edges,
            "num_faces": self.num_faces,
            "num_facets": self.num_facets,
            "vertex_coordinates": self.vertex_coordinates.tolist(),
            "edge_vertex_indices": [list(e) for e in self.edge_vertex_indices],
            "facet_vertex_indices": [list(f) for f in self.facet_vertex_indices],
            "reference_measure": self.reference_measure,
            "centroid": self.centroid.tolist(),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize :meth:`to_dict` to a JSON string (keys sorted for determinism)."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    def visualization_data(self) -> dict[str, Any]:
        """Plotting-ready data (points, edge segments, facet centroids and normals).

        Returns plain data only; no plotting library is imported here, keeping
        this module a dependency-light geometric leaf. A caller may feed the
        arrays to matplotlib or pyvista in the visualization layer.
        """
        return {
            "cell_type": self.cell_type.value,
            "points": self.vertex_coordinates.tolist(),
            "edges": [list(e) for e in self.edge_vertex_indices],
            "facet_centroids": self.facet_centroids().tolist(),
            "facet_normals": self.reference_normals().tolist(),
        }

    def pretty(self) -> str:
        """A compact, human-readable multi-line summary of the reference element."""
        lower, upper = self.bounding_box()
        lines = [
            f"{type(self).__name__} [{self.cell_type.value}]",
            f"  dimension        : topological={self.topological_dimension} "
            f"embedding={self.embedding_dimension}",
            f"  entities         : {self.num_vertices} vertices, {self.num_edges} edges, "
            f"{self.num_faces} faces, {self.num_facets} facets ({self.facet_type.value})",
            f"  measure          : {self.reference_measure:g}",
            f"  centroid         : {np.array2string(self.centroid, precision=6)}",
            f"  bounding box     : {np.array2string(lower, precision=6)} .. "
            f"{np.array2string(upper, precision=6)}",
            f"  characteristic L : {self.characteristic_length:g}",
        ]
        return "\n".join(lines)

    def debug_summary(self) -> str:
        """A verbose dump: the summary plus full incidence, tangents, and normals."""
        parts = [self.pretty(), "  facets:"]
        normals = self.reference_normals()
        for i, facet in enumerate(self.facet_vertex_indices):
            parts.append(
                f"    [{i}] vertices={facet} normal={np.array2string(normals[i], precision=6)}"
            )
        parts.append("  edges:")
        tangents = self.reference_tangents()
        for i, edge in enumerate(self.edge_vertex_indices):
            parts.append(
                f"    [{i}] vertices={edge} tangent={np.array2string(tangents[i], precision=6)}"
            )
        return "\n".join(parts)

    # ---- value semantics ----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(cell_type={self.cell_type.value!r}, "
            f"dim={self.topological_dimension}, vertices={self.num_vertices}, "
            f"facets={self.num_facets})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReferenceElement):
            return NotImplemented
        return self.cell_type == other.cell_type

    def __hash__(self) -> int:
        return hash((ReferenceElement, self.cell_type))
