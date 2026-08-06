"""Geometric operations on reference elements: measure, normals, containment.

Pure computational geometry - no shape functions, Jacobians, or quadrature.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nanofem.numerics.reference import (
    ReferenceLine,
    ReferenceQuadrilateral,
    ReferenceTriangle,
)
from nanofem.numerics.reference.element import ReferenceElement
from nanofem.utils.exceptions import InputValidationError

SQRT2 = math.sqrt(2.0)


# ---- centroid, measure, bounding box ---------------------------------------


def test_centroids() -> None:
    """Centroid is the mean of the vertices."""
    assert np.allclose(ReferenceLine().centroid, [0.0])
    assert np.allclose(ReferenceTriangle().centroid, [1.0 / 3.0, 1.0 / 3.0])
    assert np.allclose(ReferenceQuadrilateral().centroid, [0.0, 0.0])


def test_reference_measures_match_sds_conventions() -> None:
    """Canonical measures: line 2, triangle 1/2, quad 4."""
    assert ReferenceLine().reference_measure == 2.0
    assert ReferenceTriangle().reference_measure == 0.5
    assert ReferenceQuadrilateral().reference_measure == 4.0


def test_bounding_boxes() -> None:
    """Axis-aligned bounding boxes span the reference domains."""
    lo, hi = ReferenceQuadrilateral().bounding_box()
    assert np.allclose(lo, [-1.0, -1.0]) and np.allclose(hi, [1.0, 1.0])
    lo, hi = ReferenceTriangle().bounding_box()
    assert np.allclose(lo, [0.0, 0.0]) and np.allclose(hi, [1.0, 1.0])


# ---- edge lengths and characteristic size ----------------------------------


def test_edge_lengths_and_extremes() -> None:
    """Edge lengths and their min/max are geometric distances between endpoints."""
    assert np.allclose(ReferenceQuadrilateral().edge_lengths(), [2.0, 2.0, 2.0, 2.0])
    tri = ReferenceTriangle()
    assert np.allclose(sorted(tri.edge_lengths()), [1.0, 1.0, SQRT2])
    assert tri.min_edge_length == pytest.approx(1.0)
    assert tri.max_edge_length == pytest.approx(SQRT2)


def test_diameter_and_characteristic_length() -> None:
    """Diameter is the largest vertex-pair distance; characteristic length equals it."""
    assert ReferenceLine().diameter == pytest.approx(2.0)
    assert ReferenceTriangle().diameter == pytest.approx(SQRT2)
    assert ReferenceQuadrilateral().diameter == pytest.approx(2.0 * SQRT2)
    assert ReferenceQuadrilateral().characteristic_length == pytest.approx(2.0 * SQRT2)


def test_local_axes_are_identity() -> None:
    """The reference coordinate system uses the identity basis."""
    assert np.allclose(ReferenceTriangle().local_axes(), np.eye(2))
    assert np.allclose(ReferenceLine().local_axes(), np.eye(1))


# ---- tangents and outward normals ------------------------------------------


def test_reference_tangents_are_unit_vectors() -> None:
    """Edge tangents are unit length and point from the first to the second vertex."""
    tri = ReferenceTriangle()
    tangents = tri.reference_tangents()
    assert np.allclose(np.linalg.norm(tangents, axis=1), 1.0)
    assert np.allclose(tangents[2], [1.0, 0.0])  # bottom edge V0 -> V1


def test_line_outward_normals_point_along_axis() -> None:
    """The line's vertex facets point outward along the axis."""
    assert np.allclose(ReferenceLine().reference_normals(), [[-1.0], [1.0]])


def test_triangle_outward_normals() -> None:
    """Hypotenuse normal is (1,1)/sqrt2; left and bottom normals point out."""
    normals = ReferenceTriangle().reference_normals()
    assert np.allclose(normals[0], [1.0 / SQRT2, 1.0 / SQRT2])
    assert np.allclose(normals[1], [-1.0, 0.0])
    assert np.allclose(normals[2], [0.0, -1.0])


def test_quadrilateral_outward_normals() -> None:
    """Quad normals point down, right, up, left."""
    normals = ReferenceQuadrilateral().reference_normals()
    assert np.allclose(normals, [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])


def test_all_normals_are_unit_and_outward() -> None:
    """Every facet normal is unit length and points away from the centroid."""
    for element in (ReferenceLine(), ReferenceTriangle(), ReferenceQuadrilateral()):
        normals = element.reference_normals()
        centroids = element.facet_centroids()
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
        outward = np.einsum("ij,ij->i", normals, centroids - element.centroid)
        assert np.all(outward > 0.0)


# ---- containment and distance ----------------------------------------------


@pytest.mark.parametrize(
    ("element", "inside", "outside", "boundary"),
    [
        (ReferenceLine(), (0.0,), (1.5,), (1.0,)),
        (ReferenceTriangle(), (0.25, 0.25), (0.6, 0.6), (0.0, 0.0)),
        (ReferenceQuadrilateral(), (0.0, 0.0), (2.0, 0.0), (1.0, 1.0)),
    ],
)
def test_point_containment(
    element: ReferenceElement,
    inside: tuple[float, ...],
    outside: tuple[float, ...],
    boundary: tuple[float, ...],
) -> None:
    """Interior points are contained, exterior points are not, boundary is within tol."""
    assert element.contains(inside)
    assert not element.contains(outside)
    assert element.contains(boundary)


def test_signed_distance_to_boundary_values() -> None:
    """Signed distance is negative inside, zero on the boundary, positive outside."""
    quad = ReferenceQuadrilateral()
    assert quad.signed_distance_to_boundary((0.0, 0.0)) == pytest.approx(-1.0)
    assert quad.signed_distance_to_boundary((1.0, 0.0)) == pytest.approx(0.0)
    assert quad.signed_distance_to_boundary((2.0, 0.0)) == pytest.approx(1.0)

    tri = ReferenceTriangle()
    assert tri.signed_distance_to_boundary((1.0 / 3.0, 1.0 / 3.0)) == pytest.approx(
        -1.0 / (3.0 * SQRT2)
    )
    assert tri.distance_to_boundary((1.0 / 3.0, 1.0 / 3.0)) == pytest.approx(1.0 / (3.0 * SQRT2))


def test_point_of_wrong_dimension_raises() -> None:
    """A query point whose dimension mismatches the cell raises."""
    with pytest.raises(InputValidationError):
        ReferenceTriangle().contains((0.1,))
    with pytest.raises(InputValidationError):
        ReferenceLine().signed_distance_to_boundary((0.1, 0.2))
    with pytest.raises(InputValidationError):
        ReferenceQuadrilateral().contains((float("nan"), 0.0))


# ---- embedding dimension ----------------------------------------------------


def test_embedding_equals_topological_for_reference_elements() -> None:
    """Reference elements live in their natural space; physical embedding is separate."""
    for element in (ReferenceLine(), ReferenceTriangle(), ReferenceQuadrilateral()):
        assert element.embedding_dimension == element.topological_dimension
