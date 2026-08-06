"""Topology, orientation, validation, equality, and hashing of reference elements.

These tests exercise only geometry and topology - there is no finite element
mathematics in the module under test.
"""

from __future__ import annotations

import pytest

from nanofem.numerics.reference import (
    CellType,
    Dimension,
    EntityType,
    FacetType,
    Orientation,
    ReferenceHexahedron,
    ReferenceLine,
    ReferencePrism,
    ReferencePyramid,
    ReferenceQuadrilateral,
    ReferenceTetrahedron,
    ReferenceTriangle,
    cell_type_of_name,
    reference_element,
    reference_element_for_name,
)
from nanofem.numerics.reference.element import ReferenceElement
from nanofem.numerics.reference.errors import OrientationError, TopologyError
from nanofem.numerics.reference.future import _FutureReferenceElement
from nanofem.utils.exceptions import InputValidationError

BUILT = [ReferenceLine(), ReferenceTriangle(), ReferenceQuadrilateral()]


# ---- enumerations -----------------------------------------------------------


def test_dimension_is_int_enum() -> None:
    """Dimension compares and arithmetics as an integer."""
    assert int(Dimension.TWO) == 2
    assert Dimension.THREE - Dimension.ONE == 2
    assert isinstance(Dimension.TWO, int)


def test_cell_type_dimension_and_implemented_flag() -> None:
    """CellType knows its dimension and whether a concrete element exists."""
    assert CellType.LINE.topological_dimension == 1
    assert CellType.TRIANGLE.topological_dimension == 2
    assert CellType.HEXAHEDRON.topological_dimension == 3
    assert CellType.TRIANGLE.is_implemented
    assert not CellType.TETRAHEDRON.is_implemented


def test_entity_and_facet_and_orientation_enums() -> None:
    """The remaining enums expose their expected members and helpers."""
    assert EntityType.EDGE.absolute_dimension == 1
    assert EntityType.CELL.absolute_dimension is None
    assert {f.value for f in FacetType} == {"vertex", "line", "triangle", "quadrilateral"}
    assert Orientation.FORWARD.sign == 1 and Orientation.REVERSED.sign == -1


# ---- entity counts and incidence -------------------------------------------


@pytest.mark.parametrize(
    ("element", "n_vert", "n_edge", "n_face", "n_facet", "facet_type"),
    [
        (ReferenceLine(), 2, 1, 0, 2, FacetType.VERTEX),
        (ReferenceTriangle(), 3, 3, 1, 3, FacetType.LINE),
        (ReferenceQuadrilateral(), 4, 4, 1, 4, FacetType.LINE),
    ],
)
def test_entity_counts(
    element: ReferenceElement,
    n_vert: int,
    n_edge: int,
    n_face: int,
    n_facet: int,
    facet_type: FacetType,
) -> None:
    """Vertex/edge/face/facet counts and facet type match the shape."""
    assert element.num_vertices == n_vert
    assert element.num_edges == n_edge
    assert element.num_faces == n_face
    assert element.num_facets == n_facet
    assert element.facet_type is facet_type


def test_facet_entity_type_by_dimension() -> None:
    """A facet is a vertex in 1-D and an edge in 2-D."""
    assert ReferenceLine().facet_entity_type is EntityType.VERTEX
    assert ReferenceTriangle().facet_entity_type is EntityType.EDGE


def test_triangle_incidence_follows_opposite_vertex_convention() -> None:
    """Facet i is opposite vertex i, ordered for an outward normal."""
    assert ReferenceTriangle().facet_vertex_indices == ((1, 2), (2, 0), (0, 1))


def test_quadrilateral_incidence_is_bottom_right_top_left() -> None:
    """Quad facets are ordered bottom, right, top, left."""
    assert ReferenceQuadrilateral().facet_vertex_indices == ((0, 1), (1, 2), (2, 3), (3, 0))


def test_every_vertex_is_shared_by_two_facets_in_2d() -> None:
    """Incidence check: each 2-D vertex lies on exactly two facets."""
    for element in (ReferenceTriangle(), ReferenceQuadrilateral()):
        counts = [0] * element.num_vertices
        for facet in element.facet_vertex_indices:
            for v in facet:
                counts[v] += 1
        assert all(c == 2 for c in counts)


# ---- orientation ------------------------------------------------------------


def test_facet_orientations_depend_on_facet_arity() -> None:
    """Vertex facets have one orientation; edge facets are forward or reversed."""
    assert ReferenceLine().facet_orientations() == (Orientation.FORWARD,)
    assert ReferenceTriangle().facet_orientations() == (
        Orientation.FORWARD,
        Orientation.REVERSED,
    )


def test_permute_facet_reverses_edge_vertices() -> None:
    """Reversing an edge facet flips its vertex order (the neighbour's view)."""
    tri = ReferenceTriangle()
    assert tri.permute_facet(0, Orientation.FORWARD) == (1, 2)
    assert tri.permute_facet(0, Orientation.REVERSED) == (2, 1)


def test_permute_facet_rejects_bad_index_and_orientation() -> None:
    """Out-of-range facets and unsupported orientations raise."""
    with pytest.raises(InputValidationError):
        ReferenceTriangle().permute_facet(9, Orientation.FORWARD)
    with pytest.raises(InputValidationError):
        ReferenceLine().permute_facet(0, Orientation.REVERSED)  # vertex facet has no reversal


# ---- validation -------------------------------------------------------------


def test_all_built_elements_validate() -> None:
    """Every implemented reference element passes its own consistency checks."""
    for element in BUILT:
        element.validate()
        assert element.is_valid()


def test_validation_catches_inward_normal() -> None:
    """A facet ordered so its normal points inward fails orientation validation."""

    class _InwardTriangle(ReferenceTriangle):
        @property
        def facet_vertex_indices(self) -> tuple[tuple[int, ...], ...]:
            return ((1, 2), (2, 0), (1, 0))  # facet 2 reversed -> inward normal

    bad = _InwardTriangle()
    assert not bad.is_valid()
    with pytest.raises(OrientationError):
        bad.validate()


def test_validation_catches_duplicate_vertices() -> None:
    """Two coincident vertices fail topology validation."""

    import numpy as np

    class _DegenerateQuad(ReferenceQuadrilateral):
        @property
        def vertex_coordinates(self) -> np.ndarray:
            v = np.array([[-1.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
            v.setflags(write=False)
            return v

    with pytest.raises(TopologyError):
        _DegenerateQuad().validate()


def test_validation_catches_bad_vertex_index() -> None:
    """A facet referencing a non-existent vertex fails topology validation."""

    class _BadIndexTriangle(ReferenceTriangle):
        @property
        def facet_vertex_indices(self) -> tuple[tuple[int, ...], ...]:
            return ((1, 2), (2, 0), (0, 9))

    with pytest.raises(TopologyError):
        _BadIndexTriangle().validate()


def test_validation_catches_measure_mismatch() -> None:
    """A declared measure disagreeing with the geometry fails validation."""

    class _WrongMeasureLine(ReferenceLine):
        @property
        def reference_measure(self) -> float:
            return 5.0

    with pytest.raises(TopologyError):
        _WrongMeasureLine().validate()


# ---- registry and placeholders ---------------------------------------------


def test_registry_factory_by_enum_and_string() -> None:
    """The factory accepts a CellType or its string value."""
    assert isinstance(reference_element(CellType.LINE), ReferenceLine)
    assert isinstance(reference_element("triangle"), ReferenceTriangle)
    assert isinstance(reference_element("quadrilateral"), ReferenceQuadrilateral)


def test_registry_rejects_unknown_and_unimplemented() -> None:
    """Unknown names raise InputValidationError; 3-D shapes raise NotImplementedError."""
    with pytest.raises(InputValidationError):
        reference_element("banana")
    with pytest.raises(NotImplementedError):
        reference_element(CellType.HEXAHEDRON)


def test_cell_name_bridge_strips_order_suffix() -> None:
    """Mesh cell-type names map to the topological cell type regardless of order."""
    assert cell_type_of_name("tri3") is CellType.TRIANGLE
    assert cell_type_of_name("tri6") is CellType.TRIANGLE
    assert cell_type_of_name("quad8") is CellType.QUADRILATERAL
    assert isinstance(reference_element_for_name("line3"), ReferenceLine)
    with pytest.raises(InputValidationError):
        cell_type_of_name("hex27")


@pytest.mark.parametrize(
    "placeholder",
    [ReferenceTetrahedron, ReferenceHexahedron, ReferencePrism, ReferencePyramid],
)
def test_future_placeholders_refuse_construction(
    placeholder: type[_FutureReferenceElement],
) -> None:
    """3-D placeholders raise on construction but document their intended topology."""
    with pytest.raises(NotImplementedError):
        placeholder()
    assert placeholder.PROVISIONAL_TOPOLOGY["num_vertices"] >= 4


# ---- equality and hashing ---------------------------------------------------


def test_equality_is_by_shape() -> None:
    """Same-shape elements are equal; different shapes are not."""
    assert ReferenceTriangle() == ReferenceTriangle()
    assert ReferenceTriangle() != ReferenceLine()
    assert ReferenceTriangle() != object()


def test_hashing_enables_set_and_dict_use() -> None:
    """Equal elements share a hash and collapse in sets; they work as dict keys."""
    assert hash(ReferenceTriangle()) == hash(ReferenceTriangle())
    assert len({ReferenceTriangle(), ReferenceTriangle(), ReferenceLine()}) == 2
    lookup = {ReferenceLine(): 1, ReferenceTriangle(): 2, ReferenceQuadrilateral(): 3}
    assert lookup[ReferenceTriangle()] == 2


# ---- validation: every rule is proven to fire -------------------------------


def test_validation_catches_wrong_vertex_count() -> None:
    """A shape whose vertex count contradicts its cell type fails validation."""

    import numpy as np

    class _FourVertexTriangle(ReferenceTriangle):
        @property
        def vertex_coordinates(self) -> np.ndarray:
            v = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
            v.setflags(write=False)
            return v

    with pytest.raises(TopologyError, match="expected 3 vertices"):
        _FourVertexTriangle().validate()


def test_validation_catches_dimension_disagreement() -> None:
    """Coordinates whose dimension contradicts the cell type fail validation."""

    import numpy as np

    class _OneDTriangle(ReferenceTriangle):
        @property
        def vertex_coordinates(self) -> np.ndarray:
            v = np.array([[0.0], [1.0], [2.0]])
            v.setflags(write=False)
            return v

    with pytest.raises(TopologyError, match="disagrees with cell-type dimension"):
        _OneDTriangle().validate()


def test_validation_catches_non_finite_coordinates() -> None:
    """Non-finite vertex coordinates fail validation."""

    import numpy as np

    class _NanQuad(ReferenceQuadrilateral):
        @property
        def vertex_coordinates(self) -> np.ndarray:
            v = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, np.nan], [-1.0, 1.0]])
            v.setflags(write=False)
            return v

    with pytest.raises(TopologyError, match="not all finite"):
        _NanQuad().validate()


def test_validation_catches_duplicate_facet() -> None:
    """Entity uniqueness: the same facet listed twice fails validation."""

    class _DuplicateFacetTriangle(ReferenceTriangle):
        @property
        def facet_vertex_indices(self) -> tuple[tuple[int, ...], ...]:
            return ((1, 2), (2, 0), (2, 1))  # facet 2 duplicates facet 0's vertex set

    with pytest.raises(TopologyError, match="duplicate facet"):
        _DuplicateFacetTriangle().validate()


def test_validation_catches_repeated_vertex_within_a_facet() -> None:
    """A degenerate facet naming one vertex twice fails validation."""

    class _RepeatedVertexQuad(ReferenceQuadrilateral):
        @property
        def facet_vertex_indices(self) -> tuple[tuple[int, ...], ...]:
            return ((0, 1), (1, 2), (2, 2), (3, 0))

    with pytest.raises(TopologyError, match="repeats a vertex"):
        _RepeatedVertexQuad().validate()


def test_validation_catches_mixed_facet_arity() -> None:
    """Facets of inconsistent arity fail validation."""

    class _MixedArityTriangle(ReferenceTriangle):
        @property
        def facet_vertex_indices(self) -> tuple[tuple[int, ...], ...]:
            return ((1, 2), (2, 0), (0,))

    with pytest.raises(TopologyError, match="mixed facet arities"):
        _MixedArityTriangle().validate()


def test_validation_catches_non_positive_measure() -> None:
    """A non-positive declared measure fails validation before the geometry check."""

    class _ZeroMeasureTriangle(ReferenceTriangle):
        @property
        def reference_measure(self) -> float:
            return 0.0

    with pytest.raises(TopologyError, match="measure must be positive"):
        _ZeroMeasureTriangle().validate()
