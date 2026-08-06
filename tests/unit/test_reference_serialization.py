"""Serialization, immutability, utility helpers, and frozen-convention regressions.

The regression tests pin the canonical numbering and orientation conventions
(SDS C-3) so a future refactor cannot silently renumber the geometric
foundation every element will be built on.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from nanofem.numerics.reference import (
    CellType,
    ReferenceLine,
    ReferenceQuadrilateral,
    ReferenceTriangle,
    reference_element_from_dict,
)
from nanofem.numerics.reference.element import ReferenceElement
from nanofem.utils.exceptions import InputValidationError

BUILT: list[ReferenceElement] = [ReferenceLine(), ReferenceTriangle(), ReferenceQuadrilateral()]


# ---- dictionary and JSON serialization -------------------------------------


@pytest.mark.parametrize("element", BUILT)
def test_to_dict_is_complete_and_json_safe(element: ReferenceElement) -> None:
    """The payload carries the full topology/geometry record and survives json.dumps."""
    data = element.to_dict()
    assert data["schema"] == "nanofem-reference-element/1"
    assert data["cell_type"] == element.cell_type.value
    assert data["num_vertices"] == element.num_vertices
    assert data["num_facets"] == element.num_facets
    assert len(data["vertex_coordinates"]) == element.num_vertices
    assert len(data["facet_vertex_indices"]) == element.num_facets
    json.dumps(data)  # must not raise


@pytest.mark.parametrize("element", BUILT)
def test_to_json_is_deterministic(element: ReferenceElement) -> None:
    """JSON output is stable across calls (keys sorted)."""
    assert element.to_json() == element.to_json()
    assert json.loads(element.to_json())["cell_type"] == element.cell_type.value


@pytest.mark.parametrize("element", BUILT)
def test_dict_round_trip_reconstructs_equal_element(element: ReferenceElement) -> None:
    """from_dict(to_dict(x)) == x for every implemented shape."""
    restored = reference_element_from_dict(element.to_dict())
    assert restored == element
    assert restored.to_dict() == element.to_dict()


def test_from_dict_rejects_missing_key_and_tampered_geometry() -> None:
    """A payload without cell_type, or with non-canonical vertices, raises."""
    with pytest.raises(InputValidationError):
        reference_element_from_dict({"num_vertices": 3})
    tampered = ReferenceTriangle().to_dict()
    tampered["vertex_coordinates"] = [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]
    with pytest.raises(InputValidationError):
        reference_element_from_dict(tampered)


# ---- immutability -----------------------------------------------------------


@pytest.mark.parametrize("element", BUILT)
def test_vertex_coordinates_are_read_only(element: ReferenceElement) -> None:
    """Canonical coordinate arrays cannot be mutated through the property."""
    with pytest.raises(ValueError):
        element.vertex_coordinates[0, 0] = 99.0


@pytest.mark.parametrize("element", BUILT)
def test_instances_are_frozen(element: ReferenceElement) -> None:
    """Reference elements are frozen dataclasses: no attribute assignment."""
    with pytest.raises(FrozenInstanceError):
        element.cell_type = CellType.LINE  # type: ignore[misc]


def test_derived_arrays_are_fresh_copies() -> None:
    """Mutating a returned derived array cannot corrupt the canonical data."""
    tri = ReferenceTriangle()
    normals = tri.reference_normals()
    normals[0, 0] = 123.0
    assert tri.reference_normals()[0, 0] != 123.0


# ---- utility helpers --------------------------------------------------------


@pytest.mark.parametrize("element", BUILT)
def test_pretty_and_debug_summaries(element: ReferenceElement) -> None:
    """Pretty printing and the debug dump report the shape's key facts."""
    pretty = element.pretty()
    assert type(element).__name__ in pretty
    assert element.cell_type.value in pretty
    debug = element.debug_summary()
    assert "facets:" in debug and "edges:" in debug
    assert debug.count("\n") > pretty.count("\n")


@pytest.mark.parametrize("element", BUILT)
def test_repr_is_informative(element: ReferenceElement) -> None:
    """repr names the class, shape, dimension, and entity counts."""
    text = repr(element)
    assert type(element).__name__ in text
    assert element.cell_type.value in text
    assert f"vertices={element.num_vertices}" in text


@pytest.mark.parametrize("element", BUILT)
def test_visualization_data_is_plain_and_consistent(element: ReferenceElement) -> None:
    """Visualization helper returns JSON-safe plotting data, importing no plot library."""
    data = element.visualization_data()
    json.dumps(data)
    assert len(data["points"]) == element.num_vertices
    assert len(data["edges"]) == element.num_edges
    assert len(data["facet_normals"]) == element.num_facets
    assert len(data["facet_centroids"]) == element.num_facets


# ---- regression: the frozen conventions -------------------------------------


def test_regression_line_convention() -> None:
    """Line: xi in [-1, 1], vertex 0 at -1, vertex 1 at +1."""
    line = ReferenceLine()
    assert np.array_equal(line.vertex_coordinates, [[-1.0], [1.0]])
    assert line.facet_vertex_indices == ((0,), (1,))
    assert line.edge_vertex_indices == ((0, 1),)


def test_regression_triangle_convention() -> None:
    """Triangle: (0,0), (1,0), (0,1) counterclockwise; facet i opposite vertex i."""
    tri = ReferenceTriangle()
    assert np.array_equal(tri.vertex_coordinates, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert tri.facet_vertex_indices == ((1, 2), (2, 0), (0, 1))
    for i, facet in enumerate(tri.facet_vertex_indices):
        assert i not in facet  # facet i is opposite vertex i


def test_regression_quadrilateral_convention() -> None:
    """Quad: [-1,1]^2 counterclockwise from (-1,-1); facets bottom, right, top, left."""
    quad = ReferenceQuadrilateral()
    assert np.array_equal(
        quad.vertex_coordinates, [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    )
    assert quad.facet_vertex_indices == ((0, 1), (1, 2), (2, 3), (3, 0))
    assert np.allclose(quad.reference_normals(), [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])


def test_regression_serialized_payloads_are_frozen() -> None:
    """Full payload snapshots pin the conventions against silent drift."""
    assert ReferenceTriangle().to_dict() == {
        "schema": "nanofem-reference-element/1",
        "cell_type": "triangle",
        "facet_type": "line",
        "topological_dimension": 2,
        "embedding_dimension": 2,
        "num_vertices": 3,
        "num_edges": 3,
        "num_faces": 1,
        "num_facets": 3,
        "vertex_coordinates": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        "edge_vertex_indices": [[1, 2], [2, 0], [0, 1]],
        "facet_vertex_indices": [[1, 2], [2, 0], [0, 1]],
        "reference_measure": 0.5,
        "centroid": [1.0 / 3.0, 1.0 / 3.0],
    }
    assert ReferenceLine().to_dict()["vertex_coordinates"] == [[-1.0], [1.0]]
    assert ReferenceQuadrilateral().to_dict()["reference_measure"] == 4.0


# ---- independence -----------------------------------------------------------


def test_module_needs_no_other_nanofem_module() -> None:
    """Success criterion: the reference library stands alone.

    A fresh interpreter importing only the reference package must build,
    query, validate, and serialize a triangle - proving the geometric
    foundation has no dependency on mesh, core, physics, or any other layer
    (beyond the shared exception base).
    """
    import subprocess
    import sys

    script = (
        "from nanofem.numerics.reference import ReferenceTriangle;"
        "t = ReferenceTriangle();"
        "assert t.num_vertices == 3 and t.num_edges == 3;"
        "assert t.facet_vertex_indices == ((1, 2), (2, 0), (0, 1));"
        "t.validate();"
        "assert t.to_json();"
        "print('standalone OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "standalone OK" in result.stdout
