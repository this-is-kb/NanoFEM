"""Affine and identity mappings against closed forms, random maps, and embeddings.

Every reference here is written from the definition ``x = A xi + b`` and shares
no code with the least-squares fit under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.numerics.mapping import (
    AffineMapping,
    IdentityMapping,
    InverseMapError,
    MappingType,
    NonAffineError,
)
from nanofem.numerics.reference.enums import CellType
from nanofem.utils.exceptions import InputValidationError

# A right triangle mapped to (1,1), (3,1), (1,4): x = 1 + 2*xi, y = 1 + 3*eta.
TRIANGLE_NODES = [[1.0, 1.0], [3.0, 1.0], [1.0, 4.0]]


# ---- analytical affine maps --------------------------------------------------


def test_triangle_map_matches_the_closed_form() -> None:
    """A diagonal stretch plus a shift, checked term by term."""
    mapping = AffineMapping(CellType.TRIANGLE, TRIANGLE_NODES)
    assert np.allclose(mapping.linear, [[2.0, 0.0], [0.0, 3.0]])
    assert np.allclose(mapping.translation, [1.0, 1.0])
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1 / 3, 1 / 3]])
    expected = np.column_stack([1.0 + 2.0 * points[:, 0], 1.0 + 3.0 * points[:, 1]])
    assert np.allclose(mapping.map(points), expected)
    assert np.allclose(mapping.map(points), points @ mapping.linear.T + mapping.translation)


def test_triangle_jacobian_determinant_and_measure() -> None:
    """det J = 6 constant; area = reference measure (1/2) times |det J| = 3."""
    mapping = AffineMapping(CellType.TRIANGLE, TRIANGLE_NODES)
    points = mapping.sample_points()
    assert np.allclose(mapping.jacobian_determinant(points), 6.0)
    assert np.allclose(mapping.volume_scale(points), 6.0)
    assert mapping.physical_measure() == pytest.approx(3.0)
    # Independent: the shoelace area of the physical triangle.
    x, y = np.array(TRIANGLE_NODES)[:, 0], np.array(TRIANGLE_NODES)[:, 1]
    shoelace = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert mapping.physical_measure() == pytest.approx(shoelace)


def test_line_map_uses_the_bi_unit_reference() -> None:
    """The reference line is [-1, 1], so J is half the physical length."""
    mapping = AffineMapping(CellType.LINE, [[2.0], [8.0]])
    assert np.allclose(mapping.linear, [[3.0]])  # (x1 - x0) / 2
    assert np.allclose(mapping.translation, [5.0])  # the midpoint
    assert np.allclose(mapping.map([[-1.0], [0.0], [1.0]]), [[2.0], [5.0], [8.0]])
    assert mapping.physical_measure() == pytest.approx(6.0)
    assert np.allclose(mapping.jacobian_determinant([[0.0]]), 3.0)


def test_parallelogram_quadrilateral_is_affine() -> None:
    """A parallelogram is the one quadrilateral an affine map can reach."""
    mapping = AffineMapping(
        CellType.QUADRILATERAL, [[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [1.0, 1.0]]
    )
    mapping.validate()
    mapping.verify()
    assert mapping.is_affine
    assert mapping.physical_measure() == pytest.approx(2.0)  # base 2, height 1


def test_non_parallelogram_quadrilateral_is_refused_with_the_reason() -> None:
    """Four corners impose eight conditions on six unknowns; only a parallelogram fits.

    The rule is not hard-coded: it emerges as a non-zero residual in the
    least-squares fit.
    """
    with pytest.raises(NonAffineError) as excinfo:
        AffineMapping(CellType.QUADRILATERAL, [[0.0, 0.0], [2.0, 0.0], [3.0, 2.0], [0.0, 1.0]])
    message = str(excinfo.value)
    assert "parallelogram" in message and "IsoparametricMapping" in message


# ---- random affine maps ------------------------------------------------------


@pytest.mark.parametrize("cell", [CellType.LINE, CellType.TRIANGLE])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_random_affine_maps_verify(cell: CellType, seed: int) -> None:
    """Randomly generated well-conditioned affine maps satisfy every identity."""
    rng = np.random.default_rng(seed)
    dimension = 1 if cell is CellType.LINE else 2
    while True:
        linear = rng.normal(size=(dimension, dimension))
        if abs(np.linalg.det(linear)) > 0.2:  # reject near-singular draws
            break
    if np.linalg.det(linear) < 0.0:  # keep orientation positive
        linear[:, 0] *= -1.0
    translation = rng.normal(size=dimension)
    mapping = AffineMapping.from_linear_map(cell, linear, translation)
    mapping.validate()
    mapping.verify()
    assert np.allclose(mapping.linear, linear)
    assert np.allclose(mapping.translation, translation)
    points = rng.uniform(0.0, 0.4, size=(6, dimension))
    assert np.allclose(mapping.map(points), points @ linear.T + translation)
    assert np.allclose(mapping.inverse_map(mapping.map(points)), points)


def test_random_affine_jacobian_matches_the_generating_matrix() -> None:
    """The recovered Jacobian is the matrix the map was generated from."""
    rng = np.random.default_rng(99)
    linear = np.array([[2.0, 0.5], [-0.25, 1.5]])
    translation = rng.normal(size=2)
    mapping = AffineMapping.from_linear_map(CellType.TRIANGLE, linear, translation)
    points = rng.uniform(0.0, 0.4, size=(4, 2))
    assert np.allclose(mapping.jacobian(points), linear)
    assert np.allclose(mapping.jacobian_determinant(points), np.linalg.det(linear))
    assert np.allclose(mapping.metric_tensor(points), linear.T @ linear)
    assert np.allclose(mapping.inverse_jacobian(points), np.linalg.inv(linear))


# ---- the identity map --------------------------------------------------------


@pytest.mark.parametrize("cell", [CellType.LINE, CellType.TRIANGLE, CellType.QUADRILATERAL])
def test_identity_mapping_is_the_fixed_point(cell: CellType) -> None:
    """The identity leaves everything alone: the control every other map is read against."""
    mapping = IdentityMapping(cell)
    mapping.validate()
    mapping.verify()
    assert mapping.mapping_type is MappingType.IDENTITY
    assert mapping.is_affine
    dimension = mapping.topological_dimension
    points = mapping.sample_points()
    assert np.allclose(mapping.map(points), points)
    assert np.allclose(mapping.inverse_map(points), points)
    assert np.allclose(mapping.linear, np.eye(dimension))
    assert np.allclose(mapping.translation, 0.0)
    assert np.allclose(mapping.jacobian(points), np.eye(dimension))
    assert np.allclose(mapping.metric_tensor(points), np.eye(dimension))
    assert np.allclose(mapping.volume_scale(points), 1.0)
    assert mapping.physical_measure() == pytest.approx(mapping.reference_element.reference_measure)


def test_identity_leaves_gradients_untouched() -> None:
    """Under the identity a physical gradient is the reference gradient."""
    mapping = IdentityMapping(CellType.TRIANGLE)
    points = mapping.sample_points()
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(points.shape[0], 3, 2))
    assert np.allclose(mapping.physical_gradient(reference, points), reference)
    assert np.allclose(mapping.reference_gradient(reference, points), reference)


def test_identity_is_an_affine_mapping() -> None:
    """A genuine special case, so it inherits every check rather than repeating them."""
    assert isinstance(IdentityMapping(CellType.TRIANGLE), AffineMapping)


# ---- embedded elements -------------------------------------------------------


def test_bar_in_a_plane_has_a_tall_jacobian() -> None:
    """A 1-D element in 2-D: no determinant, but a well-defined measure scaling."""
    mapping = AffineMapping(CellType.LINE, [[0.0, 0.0], [3.0, 4.0]])
    mapping.validate()
    mapping.verify()
    assert mapping.is_embedded
    assert mapping.topological_dimension == 1 and mapping.embedding_dimension == 2
    assert mapping.jacobian([[0.0]]).shape == (1, 2, 1)
    assert mapping.physical_measure() == pytest.approx(5.0)  # the 3-4-5 triangle
    assert np.allclose(mapping.volume_scale([[0.0]]), 2.5)  # half the length
    assert np.allclose(mapping.metric_tensor([[0.0]]), [[6.25]])


def test_embedded_elements_refuse_a_signed_determinant() -> None:
    """Orientation is undefined for a bar in a plane, and the error says why."""
    from nanofem.numerics.mapping import EmbeddedMappingError

    mapping = AffineMapping(CellType.LINE, [[0.0, 0.0], [3.0, 4.0]])
    with pytest.raises(EmbeddedMappingError, match="square Jacobian"):
        mapping.jacobian_determinant([[0.0]])
    with pytest.raises(EmbeddedMappingError, match="second fundamental form"):
        mapping.physical_hessian(np.zeros((1, 2, 1)), np.zeros((1, 2, 1, 1)), [[0.0]])


def test_triangle_in_space_measures_by_the_gram_determinant() -> None:
    """A 2-D element in 3-D: area from sqrt(det(J^T J)), no determinant needed."""
    mapping = AffineMapping(CellType.TRIANGLE, [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    mapping.validate()
    mapping.verify()
    assert mapping.is_embedded
    assert mapping.jacobian([[0.25, 0.25]]).shape == (1, 3, 2)
    assert mapping.physical_measure() == pytest.approx(3.0)  # half of 2 by 3


def test_embedded_inverse_rejects_a_point_off_the_subspace() -> None:
    """A point that is not on the bar has no reference preimage, and is not projected."""
    mapping = AffineMapping(CellType.LINE, [[0.0, 0.0], [4.0, 0.0]])
    assert np.allclose(mapping.inverse_map([[2.0, 0.0]]), [[0.0]])
    with pytest.raises(InverseMapError, match="off the element's affine subspace"):
        mapping.inverse_map([[2.0, 1.0]])


def test_embedded_gradient_is_tangential() -> None:
    """J J^+ is the tangent projector, not the identity, for an embedded element."""
    mapping = AffineMapping(CellType.LINE, [[0.0, 0.0], [3.0, 4.0]])
    points = [[0.0]]
    jacobian = mapping.jacobian(points)[0]
    inverse = mapping.inverse_jacobian(points)[0]
    projector = jacobian @ inverse
    assert np.allclose(projector, projector.T)  # symmetric
    assert np.allclose(projector @ projector, projector)  # idempotent
    assert not np.allclose(projector, np.eye(2))  # but not the identity
    direction = np.array([3.0, 4.0]) / 5.0
    assert np.allclose(projector, np.outer(direction, direction))


# ---- construction ------------------------------------------------------------


def test_affine_mapping_accepts_a_reference_element_a_cell_type_or_a_name() -> None:
    """Three ways to name the reference domain, one result."""
    from nanofem.numerics.reference import ReferenceTriangle

    by_element = AffineMapping(ReferenceTriangle(), TRIANGLE_NODES)
    by_type = AffineMapping(CellType.TRIANGLE, TRIANGLE_NODES)
    by_name = AffineMapping("triangle", TRIANGLE_NODES)
    assert by_element == by_type == by_name
    assert hash(by_element) == hash(by_name)


def test_affine_mapping_validates_its_vertex_array() -> None:
    """The wrong number of vertices, or non-finite ones, are rejected at construction."""
    with pytest.raises(InputValidationError, match="shape"):
        AffineMapping(CellType.TRIANGLE, [[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(InputValidationError, match="non-finite"):
        AffineMapping(CellType.TRIANGLE, [[0.0, 0.0], [1.0, 0.0], [np.nan, 1.0]])


def test_from_linear_map_validates_its_shapes() -> None:
    """A and b must be consistent with the reference element and each other."""
    with pytest.raises(InputValidationError, match="linear must have shape"):
        AffineMapping.from_linear_map(CellType.TRIANGLE, [[1.0], [0.0]], [0.0, 0.0])
    with pytest.raises(InputValidationError, match="translation must have shape"):
        AffineMapping.from_linear_map(CellType.TRIANGLE, np.eye(2), [0.0])


def test_mapping_is_frozen_and_arrays_are_read_only() -> None:
    """Mappings are immutable value objects."""
    from dataclasses import FrozenInstanceError

    mapping = AffineMapping(CellType.TRIANGLE, TRIANGLE_NODES)
    with pytest.raises(FrozenInstanceError):
        mapping.vertices_ = np.zeros((3, 2))  # type: ignore[misc]
    for array in (mapping.physical_nodes, mapping.linear, mapping.translation):
        with pytest.raises(ValueError):
            array[0] = 99.0
