"""Isoparametric mappings: bilinear geometry, Newton inversion, physical Hessians.

The load-bearing test here is the last group: the mapping-Hessian correction the
SDS 2.6 note calls mandatory is shown to change the answer on a bilinear
element, and the corrected physical Hessian is checked against a finite
difference taken in *physical* space, which shares no code with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.numerics.interpolation import LagrangeInterpolation, shape_functions
from nanofem.numerics.mapping import (
    AffineMapping,
    InverseMapError,
    IsoparametricMapping,
    MappingType,
)
from nanofem.numerics.reference.enums import CellType
from nanofem.utils.exceptions import InputValidationError

# A genuinely non-parallelogram quadrilateral: bilinear, not affine.
TRAPEZOID = [[0.0, 0.0], [2.0, 0.0], [3.0, 2.0], [0.0, 1.0]]


def trapezoid_map() -> IsoparametricMapping:
    """The bilinear map onto the trapezoid above."""
    return IsoparametricMapping(LagrangeInterpolation(CellType.QUADRILATERAL, 1), TRAPEZOID)


# ---- construction and metadata -----------------------------------------------


def test_isoparametric_accepts_a_basis_or_an_interpolation() -> None:
    """Either the shape functions or the triple they come from."""
    interpolation = LagrangeInterpolation(CellType.TRIANGLE, 1)
    nodes = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    from_interpolation = IsoparametricMapping(interpolation, nodes)
    from_basis = IsoparametricMapping(shape_functions(interpolation), nodes)
    assert from_interpolation == from_basis
    assert hash(from_interpolation) == hash(from_basis)
    assert from_basis.mapping_type is MappingType.ISOPARAMETRIC


def test_isoparametric_refuses_a_non_nodal_geometry_basis() -> None:
    """Hermite dofs are slopes, which are not node coordinates."""
    from nanofem.numerics.interpolation import HermiteInterpolation

    with pytest.raises(InputValidationError, match="must be nodal"):
        IsoparametricMapping(HermiteInterpolation(CellType.LINE, 3), [[0.0], [1.0], [2.0], [3.0]])


def test_isoparametric_validates_its_node_count() -> None:
    """The node array must match the geometry basis, node for node."""
    with pytest.raises(InputValidationError, match="shape"):
        IsoparametricMapping(
            LagrangeInterpolation(CellType.QUADRILATERAL, 1), [[0.0, 0.0], [1.0, 0.0]]
        )
    with pytest.raises(InputValidationError, match="non-finite"):
        IsoparametricMapping(
            LagrangeInterpolation(CellType.TRIANGLE, 1),
            [[0.0, 0.0], [1.0, 0.0], [np.inf, 1.0]],
        )


def test_geometry_basis_is_exposed_not_duplicated() -> None:
    """The map holds a phase-4 basis and re-derives nothing."""
    mapping = trapezoid_map()
    assert mapping.geometry_basis.num_functions == 4
    assert mapping.geometry_order == 1
    assert mapping.reference_element.cell_type is CellType.QUADRILATERAL


# ---- affineness is a property of the geometry --------------------------------


def test_bilinear_quadrilateral_is_not_affine() -> None:
    """A general quadrilateral has a varying Jacobian."""
    mapping = trapezoid_map()
    assert not mapping.is_affine
    corners = mapping.reference_element.vertex_coordinates
    determinants = mapping.jacobian_determinant(corners)
    assert not np.allclose(determinants, determinants[0])


def test_parallelogram_quadrilateral_is_detected_as_affine() -> None:
    """Affineness falls out of the vanishing mapping Hessian, not the element type."""
    mapping = IsoparametricMapping(
        LagrangeInterpolation(CellType.QUADRILATERAL, 1),
        [[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [1.0, 1.0]],
    )
    assert mapping.is_affine
    assert np.allclose(mapping.mapping_hessian(mapping.sample_points()), 0.0)
    points = mapping.sample_points()
    determinants = mapping.jacobian_determinant(points)
    assert np.allclose(determinants, determinants[0])


def test_straight_sided_quadratic_triangle_is_affine() -> None:
    """A P2 triangle with mid-side nodes at the midpoints is still an affine map.

    The basis is quadratic but the geometry is not: the quadratic terms cancel.
    Assuming non-affineness from the element order alone would be wrong here,
    which is why the question is asked of the mapping Hessian.
    """
    corners = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 6.0]])
    interpolation = LagrangeInterpolation(CellType.TRIANGLE, 2)
    nodes = np.vstack(
        [
            corners,
            0.5 * (corners[1] + corners[2]),  # edge 0 = (V1, V2)
            0.5 * (corners[2] + corners[0]),  # edge 1 = (V2, V0)
            0.5 * (corners[0] + corners[1]),  # edge 2 = (V0, V1)
        ]
    )
    mapping = IsoparametricMapping(interpolation, nodes)
    mapping.validate()
    mapping.verify()
    assert mapping.is_affine
    assert np.allclose(mapping.jacobian_determinant(mapping.sample_points()), 24.0)


def test_curved_quadratic_triangle_is_not_affine() -> None:
    """Pulling one mid-side node off the midpoint curves the edge."""
    corners = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 6.0]])
    nodes = np.vstack(
        [
            corners,
            0.5 * (corners[1] + corners[2]) + np.array([0.4, 0.3]),  # bulged
            0.5 * (corners[2] + corners[0]),
            0.5 * (corners[0] + corners[1]),
        ]
    )
    mapping = IsoparametricMapping(LagrangeInterpolation(CellType.TRIANGLE, 2), nodes)
    mapping.validate()
    mapping.verify()
    assert not mapping.is_affine


# ---- the map itself ----------------------------------------------------------


def test_bilinear_map_reproduces_its_corners() -> None:
    """The reference vertices land on the physical ones."""
    mapping = trapezoid_map()
    assert np.allclose(mapping.physical_vertices, TRAPEZOID)


def test_bilinear_edges_are_straight() -> None:
    """A Q1 map sends each reference edge to a straight segment.

    On an edge two of the four shape functions vanish, so the trace is linear -
    which is why a Q1 quadrilateral is straight-sided despite being non-affine,
    and why its area is still the shoelace of its corners.
    """
    mapping = trapezoid_map()
    element = mapping.reference_element
    for facet_index, (first, second) in enumerate(element.facet_vertex_indices):
        vertices = element.vertex_coordinates
        fractions = np.linspace(0.0, 1.0, 7).reshape(-1, 1)
        along = vertices[first] + fractions * (vertices[second] - vertices[first])
        images = mapping.map(along)
        start, end = images[0], images[-1]
        expected = start + fractions * (end - start)
        assert np.allclose(images, expected, atol=1e-12), f"facet {facet_index} bows"


def test_bilinear_centroid_is_the_mean_of_the_corners() -> None:
    """For Q1 each corner function is 1/4 at the centre, so the images coincide."""
    mapping = trapezoid_map()
    assert np.allclose(mapping.centroid, np.mean(TRAPEZOID, axis=0))


def test_non_affine_measure_is_deferred_to_quadrature() -> None:
    """The measure of a non-affine element is an integral, so it is not computed here."""
    with pytest.raises(NotImplementedError, match="quadrature"):
        trapezoid_map().physical_measure()


# ---- Newton inversion --------------------------------------------------------


def test_newton_inverse_round_trips_on_a_bilinear_element() -> None:
    """A polynomial system has no closed-form inverse, so Newton solves it."""
    mapping = trapezoid_map()
    rng = np.random.default_rng(4)
    points = rng.uniform(-0.9, 0.9, size=(20, 2))
    assert np.allclose(mapping.inverse_map(mapping.map(points)), points, atol=1e-10)


def test_newton_inverse_is_exact_for_affine_geometry() -> None:
    """An affine residual is linear, so Newton lands in a single step."""
    mapping = IsoparametricMapping(
        LagrangeInterpolation(CellType.TRIANGLE, 1),
        [[1.0, 1.0], [3.0, 1.0], [1.0, 4.0]],
    )
    points = np.array([[0.2, 0.3], [0.0, 0.0], [1 / 3, 1 / 3]])
    assert np.allclose(mapping.inverse_map(mapping.map(points)), points, atol=1e-14)
    # And it agrees with the closed-form affine inverse.
    affine = AffineMapping(CellType.TRIANGLE, [[1.0, 1.0], [3.0, 1.0], [1.0, 4.0]])
    physical = mapping.map(points)
    assert np.allclose(mapping.inverse_map(physical), affine.inverse_map(physical))


def test_inverse_map_may_land_outside_the_reference_cell() -> None:
    """A physical point outside the element gets reference coordinates outside it.

    The map is a polynomial, so it is defined beyond the reference cell and
    Newton will happily find a preimage there. That is the right behaviour and
    the same choice deal.II makes: the inverse answers "which reference
    coordinates map here", and whether the answer lies *in* the cell is the
    caller's question, asked with ReferenceElement.contains.
    """
    mapping = trapezoid_map()
    reference = mapping.inverse_map([[4.0, 3.0]])
    assert not mapping.reference_element.contains(reference[0])
    assert np.allclose(mapping.map(reference), [[4.0, 3.0]], atol=1e-9)
    far = mapping.inverse_map([[1e9, -1e9]])
    assert np.allclose(mapping.map(far), [[1e9, -1e9]], rtol=1e-9)


def test_newton_reports_its_own_failure_and_does_not_blame_the_element() -> None:
    """When the iterate hits a fold, the message names the real cause.

    A folded quadrilateral has a Jacobian that changes sign, so the map is
    degenerate along a line inside it. A Newton iterate that reaches that line
    cannot continue - but propagating the degeneracy would report it as an
    element defect, sending the user to inspect geometry when the real answer is
    that the target point has no preimage on this branch.
    """
    folded = IsoparametricMapping(
        LagrangeInterpolation(CellType.QUADRILATERAL, 1),
        [[0.0, 0.0], [2.0, 0.0], [-1.0, 1.0], [1.0, 1.0]],
    )
    determinants = folded.jacobian_determinant(folded.reference_element.vertex_coordinates)
    assert determinants.min() < 0.0 < determinants.max()  # it really does fold
    with pytest.raises(InverseMapError) as excinfo:
        folded.inverse_map([[100.0, 100.0]])
    assert "preimage" in str(excinfo.value)


def test_a_sound_element_stays_sound_after_a_failed_inverse() -> None:
    """A failed solve says nothing about the element, and leaves it untouched."""
    mapping = trapezoid_map()
    assert mapping.is_valid()
    with pytest.raises(InverseMapError):
        IsoparametricMapping(
            LagrangeInterpolation(CellType.QUADRILATERAL, 1),
            [[0.0, 0.0], [2.0, 0.0], [-1.0, 1.0], [1.0, 1.0]],
        ).inverse_map([[100.0, 100.0]])
    assert mapping.is_valid()


# ---- the physical Hessian and its correction ---------------------------------


def physical_finite_difference_hessian(
    mapping: IsoparametricMapping, physical_point: np.ndarray, step: float
) -> np.ndarray:
    """Differentiate the physical gradient in *physical* space, independently.

    Uses only ``inverse_map`` and ``physical_gradient``; it never touches the
    mapping Hessian, so it is a genuine second opinion on the transformation.
    """
    dimension = mapping.embedding_dimension
    columns = []
    for axis in range(dimension):
        offset = np.zeros(dimension)
        offset[axis] = step
        gradients = []
        for sign in (+1.0, -1.0):
            shifted = (physical_point + sign * offset).reshape(1, -1)
            reference = mapping.inverse_map(shifted)
            gradients.append(
                mapping.physical_gradient(mapping.geometry_basis.gradient(reference), reference)[0]
            )
        columns.append((gradients[0] - gradients[1]) / (2.0 * step))
    return np.stack(columns, axis=-1)  # (n_functions, emb, emb)


def test_physical_hessian_matches_a_physical_space_finite_difference() -> None:
    """The corrected Hessian agrees with an independent derivative in physical space."""
    mapping = trapezoid_map()
    basis = mapping.geometry_basis
    for reference_point in ([[0.1, -0.2]], [[-0.3, 0.4]], [[0.0, 0.0]]):
        reference = np.asarray(reference_point)
        analytic = mapping.physical_hessian(
            basis.gradient(reference), basis.hessian(reference), reference
        )[0]
        numeric = physical_finite_difference_hessian(mapping, mapping.map(reference)[0], step=1e-5)
        assert np.allclose(analytic, numeric, atol=1e-6)


def test_dropping_the_mapping_hessian_gives_a_different_answer() -> None:
    """The correction the SDS calls mandatory is shown to matter, not merely asserted.

    On a bilinear element the naive ``J^-T H_xi J^-1`` disagrees with the truth
    while the gradient is untouched - which is exactly why the error hides until
    a C1 theory needs second derivatives.
    """
    mapping = trapezoid_map()
    basis = mapping.geometry_basis
    reference = np.array([[0.2, -0.1]])
    correct = mapping.physical_hessian(
        basis.gradient(reference), basis.hessian(reference), reference
    )[0]
    inverse = mapping.inverse_jacobian(reference)[0]
    naive = np.einsum("ai,ab,bj->ij", inverse, basis.hessian(reference)[0][0], inverse)
    assert not np.allclose(correct[0], naive, atol=1e-6)
    assert not np.allclose(mapping.mapping_hessian(reference), 0.0)


def test_the_correction_vanishes_for_an_affine_map() -> None:
    """On an affine element the naive formula is right, because K is zero."""
    mapping = AffineMapping(CellType.TRIANGLE, [[1.0, 1.0], [3.0, 1.0], [1.0, 4.0]])
    basis = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 2))
    reference = np.array([[0.2, 0.3]])
    correct = mapping.physical_hessian(
        basis.gradient(reference), basis.hessian(reference), reference
    )
    inverse = mapping.inverse_jacobian(reference)[0]
    naive = np.einsum("ai,nab,bj->nij", inverse, basis.hessian(reference)[0], inverse)
    assert np.allclose(correct[0], naive, atol=1e-12)
    assert np.allclose(mapping.mapping_hessian(reference), 0.0)


def test_physical_hessian_reproduces_a_known_quadratic_field() -> None:
    """On an affine element, a P2 field's physical Hessian is exact and known.

    Interpolate f(x, y) = x^2 + 3*x*y at the nodes; its physical Hessian is the
    constant [[2, 3], [3, 0]]. Recovering that exercises the whole chain -
    shape functions, Jacobian, its inverse, and the transformation.
    """
    mapping = AffineMapping(CellType.TRIANGLE, [[1.0, 1.0], [3.0, 1.0], [1.0, 4.0]])
    basis = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 2))
    nodal_points = mapping.map(basis.interpolation.node_locations)
    values = nodal_points[:, 0] ** 2 + 3.0 * nodal_points[:, 0] * nodal_points[:, 1]
    reference = np.array([[0.15, 0.25], [0.0, 0.0], [1 / 3, 1 / 3]])
    hessians = mapping.physical_hessian(
        basis.gradient(reference), basis.hessian(reference), reference
    )
    field_hessian = np.einsum("n,pnij->pij", values, hessians)
    assert np.allclose(field_hessian, [[2.0, 3.0], [3.0, 0.0]], atol=1e-9)


def test_physical_hessian_validates_its_input_shape() -> None:
    """A Hessian of the wrong rank is rejected rather than broadcast into nonsense."""
    mapping = trapezoid_map()
    with pytest.raises(InputValidationError, match="reference_hessian"):
        mapping.physical_hessian(np.zeros((1, 4, 2)), np.zeros((1, 4, 2)), [[0.0, 0.0]])


# ---- the full suite ----------------------------------------------------------


def test_bilinear_map_passes_every_identity() -> None:
    """A non-affine map satisfies the whole verification suite."""
    mapping = trapezoid_map()
    mapping.validate()
    mapping.verify()
    assert mapping.is_valid()
