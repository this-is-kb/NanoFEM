"""Mesh: nodes, homogeneous cell blocks, named regions; topology queries (SDS 2.1).

Storage is structure-of-arrays (coordinate array + connectivity arrays per
block, SDS C-8); Node and Cell are per-entity record views. Global cell ids
run block by block in declaration order (deterministic, SDS C-5). The mesh is
frozen after construction: coordinate and connectivity arrays are marked
read-only. Spatial *algorithms* live in numerics.search (ADR-007).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.mesh.facet_region import FacetRegion
from nanofem.mesh.node import Node
from nanofem.mesh.region import Region
from nanofem.numerics.reference.registry import reference_element_for_name
from nanofem.utils.exceptions import InputValidationError, MeshError
from nanofem.utils.validation import require_finite_array, require_identifier


@dataclass(frozen=True, eq=False)
class CellBlock:
    """Homogeneous connectivity for one cell type plus its region tag."""

    cell_type: str
    connectivity: NDArray[np.int64]
    region: str

    def __post_init__(self) -> None:
        require_identifier(self.cell_type, "cell_type")
        require_identifier(self.region, "cell block region")
        conn = np.asarray(self.connectivity, dtype=np.int64)
        if conn.ndim != 2 or conn.shape[0] < 1 or conn.shape[1] < 1:
            raise InputValidationError(
                f"block '{self.region}': connectivity must be (n_cells, n_en), got {conn.shape}"
            )
        if int(conn.min()) < 0:
            raise InputValidationError(f"block '{self.region}': negative node index")
        conn.setflags(write=False)
        object.__setattr__(self, "connectivity", conn)

    @property
    def num_cells(self) -> int:
        """Cells in this block."""
        return int(self.connectivity.shape[0])

    @property
    def nodes_per_cell(self) -> int:
        """Nodes per cell in this block."""
        return int(self.connectivity.shape[1])


@dataclass(frozen=True)
class Cell:
    """Per-entity cell record view (authoritative storage is the CellBlock).

    material_id / geometry_id are optional convenience tags; the canonical
    binding is region-level via Model domain definitions (dev note N-8).
    """

    cell_id: int
    cell_type: str
    connectivity: tuple[int, ...]
    region: str
    material_id: str | None = None
    geometry_id: str | None = None


class Mesh:
    """Container and query interface; immutable after construction."""

    def __init__(
        self,
        coordinates: NDArray[np.float64],
        blocks: tuple[CellBlock, ...],
        regions: tuple[Region, ...] = (),
        facet_regions: tuple[FacetRegion, ...] = (),
    ) -> None:
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] < 1 or not 1 <= coords.shape[1] <= 3:
            raise MeshError(f"coordinates must be (n_nodes, d) with d in 1..3, got {coords.shape}")
        require_finite_array(coords, "mesh coordinates")
        if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
            raise MeshError("mesh contains duplicate node coordinates")
        coords.setflags(write=False)
        self._coordinates = coords

        self._blocks = tuple(blocks)
        n_nodes = self.num_nodes
        offset = 0
        self._block_offsets: list[int] = []
        referenced = np.zeros(n_nodes, dtype=bool)
        for b in self._blocks:
            if int(b.connectivity.max(initial=-1)) >= n_nodes:
                raise MeshError(f"block '{b.region}' references node id >= {n_nodes}")
            referenced[np.unique(b.connectivity)] = True
            self._block_offsets.append(offset)
            offset += b.num_cells
        self._num_cells = offset

        if self._blocks:
            orphans = np.flatnonzero(~referenced)
            if orphans.size:
                raise MeshError(f"orphan nodes (unreferenced by any cell): {orphans.tolist()}")

        self._regions: dict[str, Region] = {}
        for r in regions:
            if r.name in self._regions:
                raise MeshError(f"duplicate region name '{r.name}'")
            if r.dimension == 0:
                bound, kind = n_nodes, "node"
            elif r.dimension == self.dimension:
                bound, kind = self._num_cells, "cell"
            else:
                raise MeshError(
                    f"region '{r.name}': dimension {r.dimension} unsupported in phase 1 "
                    "(facet/edge regions arrive with the phase-2 orientation machinery)"
                )
            bad = [i for i in r.entity_ids if i >= bound]
            if bad:
                raise MeshError(f"region '{r.name}': {kind} ids out of range: {bad}")
            self._regions[r.name] = r

        # implicit cell regions from block tags (declaration order, SDS C-5)
        for b, off in zip(self._blocks, self._block_offsets, strict=True):
            ids = tuple(range(off, off + b.num_cells))
            if b.region in self._regions:
                existing = self._regions[b.region]
                if existing.dimension != self.dimension:
                    raise MeshError(f"region '{b.region}' used for both nodes and cells")
                merged = tuple(sorted(set(existing.entity_ids) | set(ids)))
                self._regions[b.region] = Region(b.region, self.dimension, merged)
            else:
                self._regions[b.region] = Region(b.region, self.dimension, ids)

        # node -> cells adjacency (connectivity query; plain bookkeeping)
        adjacency: dict[int, list[int]] = {}
        for b, off in zip(self._blocks, self._block_offsets, strict=True):
            for local, row in enumerate(b.connectivity):
                cid = off + local
                for nid in row.tolist():
                    adjacency.setdefault(int(nid), []).append(cid)
        self._node_to_cells: dict[int, tuple[int, ...]] = {
            k: tuple(v) for k, v in adjacency.items()
        }

        self._facet_regions: dict[str, FacetRegion] = {}
        for fr in facet_regions:
            if fr.name in self._facet_regions:
                raise MeshError(f"duplicate facet region name '{fr.name}'")
            for cell_id, local_index in fr.facets:
                if cell_id >= self._num_cells:
                    raise MeshError(f"facet region '{fr.name}': cell id {cell_id} out of range")
                num_facets = reference_element_for_name(self.cell(cell_id).cell_type).num_facets
                if local_index >= num_facets:
                    raise MeshError(
                        f"facet region '{fr.name}': cell {cell_id} has {num_facets} facets, "
                        f"local index {local_index} out of range"
                    )
            self._facet_regions[fr.name] = fr

    # ---- sizes ----
    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return int(self._coordinates.shape[0])

    @property
    def num_cells(self) -> int:
        """Number of cells across all blocks."""
        return self._num_cells

    @property
    def dimension(self) -> int:
        """Spatial dimension d."""
        return int(self._coordinates.shape[1])

    # ---- entity views ----
    @property
    def coordinates(self) -> NDArray[np.float64]:
        """Read-only (n_nodes, d) coordinate array."""
        return self._coordinates

    @property
    def blocks(self) -> tuple[CellBlock, ...]:
        """Cell blocks in declaration order."""
        return self._blocks

    def node(self, node_id: int) -> Node:
        """Per-entity node record view."""
        if not 0 <= node_id < self.num_nodes:
            raise MeshError(f"node id {node_id} out of range [0, {self.num_nodes})")
        return Node(node_id, self._coordinates[node_id].copy())

    def cell(self, cell_id: int) -> Cell:
        """Per-entity cell record view (global id across blocks)."""
        if not 0 <= cell_id < self._num_cells:
            raise MeshError(f"cell id {cell_id} out of range [0, {self._num_cells})")
        for b, off in zip(self._blocks, self._block_offsets, strict=True):
            if cell_id < off + b.num_cells:
                row = b.connectivity[cell_id - off]
                return Cell(cell_id, b.cell_type, tuple(int(n) for n in row), b.region)
        raise MeshError(f"cell id {cell_id} not resolved")  # pragma: no cover

    # ---- region and topology queries ----
    def region(self, name: str) -> Region:
        """Named region; unknown names list what exists (never silently drop)."""
        try:
            return self._regions[name]
        except KeyError:
            known = ", ".join(sorted(self._regions)) or "(none)"
            raise MeshError(f"unknown region '{name}'; known regions: {known}") from None

    @property
    def region_names(self) -> tuple[str, ...]:
        """All region names, sorted."""
        return tuple(sorted(self._regions))

    def cells_in_region(self, name: str) -> tuple[int, ...]:
        """Cell ids of a cell-dimension region."""
        r = self.region(name)
        if r.dimension != self.dimension:
            raise MeshError(f"region '{name}' is not a cell region (dimension {r.dimension})")
        return r.entity_ids

    def nodes_in_region(self, name: str) -> tuple[int, ...]:
        """Node ids of a node region (dimension 0)."""
        r = self.region(name)
        if r.dimension != 0:
            raise MeshError(f"region '{name}' is not a node region (dimension {r.dimension})")
        return r.entity_ids

    def cells_of_node(self, node_id: int) -> tuple[int, ...]:
        """Cells adjacent to a node (connectivity query)."""
        if not 0 <= node_id < self.num_nodes:
            raise MeshError(f"node id {node_id} out of range [0, {self.num_nodes})")
        return self._node_to_cells.get(node_id, ())

    # ---- facet regions ----
    def facet_region(self, name: str) -> FacetRegion:
        """Named facet region; unknown names list what exists (never silently drop)."""
        try:
            return self._facet_regions[name]
        except KeyError:
            known = ", ".join(sorted(self._facet_regions)) or "(none)"
            raise MeshError(f"unknown facet region '{name}'; known: {known}") from None

    @property
    def facet_region_names(self) -> tuple[str, ...]:
        """All facet region names, sorted."""
        return tuple(sorted(self._facet_regions))

    def facets_in_region(self, name: str) -> tuple[tuple[int, int], ...]:
        """``(cell_id, local_facet_index)`` pairs of a facet region."""
        return self.facet_region(name).facets

    def facet_node_ids(self, cell_id: int, local_facet_index: int) -> tuple[int, ...]:
        """Global node ids of one cell's facet, in the reference element's own local order."""
        cell = self.cell(cell_id)
        local_vertices = reference_element_for_name(cell.cell_type).facet_vertex_indices[
            local_facet_index
        ]
        return tuple(cell.connectivity[i] for i in local_vertices)

    # ---- serialization ----
    def to_dict(self) -> dict[str, object]:
        """JSON-safe mesh record (serialization strategy, docs/design/OBJECT_MODEL.md)."""
        return {
            "coordinates": [[float(c) for c in row] for row in self._coordinates],
            "blocks": [
                {
                    "cell_type": b.cell_type,
                    "connectivity": [[int(n) for n in row] for row in b.connectivity],
                    "region": b.region,
                }
                for b in self._blocks
            ],
            "regions": [r.to_dict() for _, r in sorted(self._regions.items()) if r.dimension == 0],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Mesh:
        """Rebuild a mesh from its to_dict record."""
        coords = np.asarray(data["coordinates"], dtype=np.float64)
        raw_blocks = data["blocks"]
        assert isinstance(raw_blocks, list)
        blocks = tuple(
            CellBlock(
                str(b["cell_type"]),
                np.asarray(b["connectivity"], dtype=np.int64),
                str(b["region"]),
            )
            for b in raw_blocks
        )
        raw_regions = data.get("regions", [])
        assert isinstance(raw_regions, list)
        regions = tuple(
            Region(
                str(r["name"]),
                int(r["dimension"]),
                tuple(int(i) for i in r["entity_ids"]),
                {str(k): str(v) for k, v in dict(r.get("metadata", {})).items()},
            )
            for r in raw_regions
        )
        return cls(coords, blocks, regions)
