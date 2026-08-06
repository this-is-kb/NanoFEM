"""Unit tests for Node, CellBlock, Cell, Region, and Mesh (requirements 1-4)."""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.node import Node
from nanofem.mesh.region import Region
from nanofem.utils.exceptions import InputValidationError, MeshError


def two_quad_mesh() -> Mesh:
    """Two quad4 cells side by side, six nodes, with a node region 'left'."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    block = CellBlock("quad4", np.array([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=np.int64), "body")
    left = Region("left", 0, (0, 3))
    return Mesh(coords, (block,), (left,))


def test_node_validation_immutability_and_metadata() -> None:
    """Node validates id/coords, freezes the array, and carries tags/metadata."""
    n = Node(3, np.array([1.0, 2.0]), tags=frozenset({"support"}), metadata={"origin": "gmsh"})
    assert n.dimension == 2 and "support" in n.tags and n.metadata["origin"] == "gmsh"
    with pytest.raises(ValueError):
        n.coordinates[0] = 9.0  # read-only array
    with pytest.raises(InputValidationError):
        Node(-1, np.array([0.0]))
    with pytest.raises(InputValidationError):
        Node(0, np.array([np.nan]))


def test_region_validation() -> None:
    """Regions reject empty, duplicate, and negative entity ids."""
    Region("edge", 0, (0, 1))
    with pytest.raises(InputValidationError):
        Region("edge", 0, ())
    with pytest.raises(InputValidationError):
        Region("edge", 0, (0, 0))
    with pytest.raises(InputValidationError):
        Region("edge", 5, (0,))


def test_mesh_queries_and_cell_view() -> None:
    """Topology/connectivity queries and the per-entity Cell view."""
    m = two_quad_mesh()
    assert m.num_nodes == 6 and m.num_cells == 2 and m.dimension == 2
    assert m.cells_in_region("body") == (0, 1)
    assert m.nodes_in_region("left") == (0, 3)
    with pytest.raises(MeshError):
        m.nodes_in_region("body")  # node-region query on a cell region
    assert m.cells_of_node(1) == (0, 1)  # shared edge node
    c = m.cell(1)
    assert c.cell_type == "quad4" and c.connectivity == (1, 2, 5, 4) and c.region == "body"
    with pytest.raises(MeshError):
        m.region("nope")


def test_mesh_rejects_bad_connectivity_orphans_duplicates() -> None:
    """Fail-fast integrity: out-of-range indices, orphan nodes, duplicate coords."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    with pytest.raises(MeshError):
        Mesh(coords, (CellBlock("line2", np.array([[0, 9]], dtype=np.int64), "b"),))
    with pytest.raises(MeshError):  # node 2 orphaned
        Mesh(coords, (CellBlock("line2", np.array([[0, 1]], dtype=np.int64), "b"),))
    dup = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(MeshError):
        Mesh(dup, (CellBlock("line2", np.array([[0, 1], [1, 2]], dtype=np.int64), "b"),))


def test_mesh_is_frozen_and_round_trips() -> None:
    """Coordinates are read-only; to_dict/from_dict is an exact inverse."""
    m = two_quad_mesh()
    with pytest.raises(ValueError):
        m.coordinates[0, 0] = 5.0
    m2 = Mesh.from_dict(m.to_dict())
    assert np.array_equal(m2.coordinates, m.coordinates)
    assert m2.region_names == m.region_names
    assert m2.cell(0).connectivity == m.cell(0).connectivity
