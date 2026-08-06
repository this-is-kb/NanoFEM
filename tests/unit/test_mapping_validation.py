"""Geometric validation, verification trip tests, caching, and frozen regressions.

Each detector is shown firing on the geometry it exists to catch, and each
verification is broken on purpose. A check that never fires is a decoration.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.numerics.interpolation import LagrangeInterpolation
from nanofem.numerics.mapping import (
    AffineMapping,
    CurvilinearMapping,
    GeometricMapping,
    HighOrderMapping,
    IdentityMapping,
    IsoparametricMapping,
    MappingError,
    MappingType,
    NURBSMapping,
)
from nanofem.numerics.mapping.future import _FutureMapping
from nanofem.numerics.reference.enums import CellType
from nanofem.utils.exceptions import DegenerateCellError

GOOD_TRIANGLE = [[1.0, 1.0], [3.0, 1.0], [1.0, 4.0]]
TRAPEZOID = [[0.0, 0.0], [2.0, 0.0], [3.0, 2.0], [0.0, 1.0]]


def all_mappings() -> list[GeometricMapping]:
    """One of every implemented mapping kind, all well formed."""
    return [
        IdentityMapping(CellType.TRIANGLE),
        AffineMapping(CellType.TRIANGLE, GOOD_TRIANGLE),
        AffineMapping(CellType.LINE, [[0.0, 0.0], [3.0, 4.0]]),  # embedded
        IsoparametricMapping(LagrangeInterpolation(CellType.QUADRILATERAL, 1), TRAPEZOID),
    ]


IDS = ["identity-tri", "affine-tri", "affine-bar-2d", "iso-quad"]


# ---- geometric validation: each detector fires -------------------------------


def test_collinear_triangle_is_degenerate() -> None:
    """Three collinear points have no area, so the Jacobian collapses."""
    mapping = AffineMapping(CellType.TRIANGLE, [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert not mapping.is_valid()
    with pytest.raises(DegenerateCellError, match="rank deficient"):
        mapping.validate()


def test_zero_length_line_is_degenerate() -> None:
    """A member with coincident ends is caught as coincident nodes first."""
    mapping = AffineMapping(CellType.LINE, [[2.0, 3.0], [2.0, 3.0]])
    with pytest.raises(DegenerateCellError, match="coincident"):
        mapping.validate()


def test_coincident_nodes_are_detected() -> None:
    """Two nodes at the same place collapse the map."""
    mapping = AffineMapping(CellType.TRIANGLE, [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    with pytest.raises(DegenerateCellError, match="coincident"):
        mapping.validate_nodes()


def test_reversed_node_order_inverts_the_element() -> None:
    """Listing a triangle clockwise gives a negative determinant, and the message says why."""
    mapping = AffineMapping(CellType.TRIANGLE, [[1.0, 1.0], [1.0, 4.0], [3.0, 1.0]])
    assert np.all(mapping.jacobian_determinant(mapping.sample_points()) < 0.0)
    with pytest.raises(DegenerateCellError) as excinfo:
        mapping.validate()
    assert "inverted" in str(excinfo.value)
    assert "ordered the wrong way" in str(excinfo.value)
    with pytest.raises(MappingError, match="orientation is not preserved"):
        mapping.verify_orientation()


def test_orientation_reversal_flips_only_the_sign() -> None:
    """A reversed element has the same measure scaling, so |det| cannot detect it."""
    forward = AffineMapping(CellType.TRIANGLE, GOOD_TRIANGLE)
    reversed_ = AffineMapping(
        CellType.TRIANGLE, [GOOD_TRIANGLE[0], GOOD_TRIANGLE[2], GOOD_TRIANGLE[1]]
    )
    points = forward.sample_points()
    assert np.allclose(
        forward.jacobian_determinant(points), -reversed_.jacobian_determinant(points)
    )
    assert np.allclose(forward.volume_scale(points), reversed_.volume_scale(points))
    assert forward.is_valid() and not reversed_.is_valid()


def test_sliver_triangle_is_near_singular() -> None:
    """A needle has a huge Jacobian condition number and is reported as a sliver."""
    mapping = AffineMapping(CellType.TRIANGLE, [[0.0, 0.0], [1.0, 0.0], [0.5, 1e-10]])
    with pytest.raises(DegenerateCellError, match="near singular|sliver|degenerate"):
        mapping.validate()


def test_inverted_bilinear_quadrilateral_is_detected() -> None:
    """A self-overlapping quadrilateral inverts somewhere, which the corners reveal."""
    mapping = IsoparametricMapping(
        LagrangeInterpolation(CellType.QUADRILATERAL, 1),
        [[0.0, 0.0], [2.0, 0.0], [-1.0, 1.0], [1.0, 1.0]],  # nodes 2 and 3 swapped
    )
    assert not mapping.is_valid()
    with pytest.raises(DegenerateCellError):
        mapping.validate()


def test_a_wrong_dimensional_element_is_rejected() -> None:
    """A 2-D element cannot live in 1-D space."""
    mapping = AffineMapping.__new__(AffineMapping)
    object.__setattr__(mapping, "reference_", IdentityMapping(CellType.TRIANGLE).reference_)
    nodes = np.array([[0.0], [1.0], [2.0]])
    nodes.setflags(write=False)
    object.__setattr__(mapping, "vertices_", nodes)
    with pytest.raises(MappingError, match="cannot live in"):
        mapping.validate_nodes()


@pytest.mark.parametrize("mapping", all_mappings(), ids=IDS)
def test_well_formed_mappings_validate_and_verify(mapping: GeometricMapping) -> None:
    """Every good mapping passes both suites."""
    mapping.validate()
    mapping.verify()
    assert mapping.is_valid()


# ---- verification trip tests -------------------------------------------------


def test_verify_jacobian_catches_a_wrong_jacobian() -> None:
    """A plausible, systematic Jacobian error is caught by the finite difference.

    The finite-difference route shares no code with the analytic Jacobian, which
    is the point: an identity built from the same J would agree with itself.
    """

    class _BadJacobian(AffineMapping):
        def jacobian(self, points: object) -> np.ndarray:
            return np.asarray(super().jacobian(points)) * 1.5  # type: ignore[arg-type]

    broken = _BadJacobian(CellType.TRIANGLE, GOOD_TRIANGLE)
    with pytest.raises(MappingError, match="finite difference"):
        broken.verify_jacobian()


def test_verify_gradient_transformation_catches_a_transposed_index() -> None:
    """The classic bug at this seam: J^-1 used where J^-T belongs."""

    class _TransposedPushForward(AffineMapping):
        def physical_gradient(self, reference_gradient: object, points: object) -> np.ndarray:
            array = self._as_reference(points)  # type: ignore[arg-type]
            inverse = self.inverse_jacobian(array)
            gradients = np.asarray(reference_gradient, dtype=np.float64)
            # Contract the wrong index of J^+ - dimensionally fine, silently wrong.
            return np.einsum("pai,pni->pna", inverse, gradients)  # type: ignore[no-any-return]

    # The Jacobian must be non-symmetric, or J^T J^-1 collapses to the identity and
    # the misindexed contraction would accidentally be correct.
    broken = _TransposedPushForward(CellType.TRIANGLE, [[0.0, 0.0], [2.0, 0.5], [1.5, 3.0]])
    with pytest.raises(MappingError, match="pulling back|misindexed"):
        broken.verify_gradient_transformation()


def test_verify_metric_tensor_catches_an_asymmetric_metric() -> None:
    """A metric that is not J^T J is caught."""

    class _BadMetric(AffineMapping):
        def metric_tensor(self, points: object) -> np.ndarray:
            array = np.asarray(super().metric_tensor(points)).copy()  # type: ignore[arg-type]
            array[:, 0, 1] += 0.5
            return array

    broken = _BadMetric(CellType.TRIANGLE, GOOD_TRIANGLE)
    with pytest.raises(MappingError, match="not J\\^T J"):
        broken.verify_metric_tensor()


def test_verify_inverse_consistency_catches_a_broken_inverse() -> None:
    """An inverse that does not undo the map is caught."""

    class _BadInverse(AffineMapping):
        def inverse_map(self, points: object) -> np.ndarray:
            return np.asarray(super().inverse_map(points)) + 0.1  # type: ignore[arg-type]

    broken = _BadInverse(CellType.TRIANGLE, GOOD_TRIANGLE)
    with pytest.raises(MappingError, match="inverse_map"):
        broken.verify_inverse_consistency()


def test_verify_measure_scaling_catches_a_wrong_scaling() -> None:
    """The closed-form shoelace area is an independent standard for an affine map."""

    class _BadScale(AffineMapping):
        def volume_scale(self, points: object) -> np.ndarray:
            return np.asarray(super().volume_scale(points)) * 2.0  # type: ignore[arg-type]

    broken = _BadScale(CellType.TRIANGLE, GOOD_TRIANGLE)
    with pytest.raises(MappingError, match="closed-form physical measure"):
        broken.verify_measure_scaling()


def test_verify_affine_exactness_catches_a_non_affine_map() -> None:
    """An affine map whose image is not A xi + b is caught."""

    class _Bent(AffineMapping):
        def map(self, points: object) -> np.ndarray:
            array = self._as_reference(points)  # type: ignore[arg-type]
            return np.asarray(super().map(array)) + 0.1 * array[:, :1] ** 2

    broken = _Bent(CellType.TRIANGLE, GOOD_TRIANGLE)
    with pytest.raises(MappingError, match="is not A xi \\+ b"):
        broken.verify_affine_exactness()


# ---- caching -----------------------------------------------------------------


def test_derived_quantities_are_cached_per_point_set() -> None:
    """The same points are transformed once; the pattern SDS C-8 wants."""
    mapping = IsoparametricMapping(LagrangeInterpolation(CellType.QUADRILATERAL, 1), TRAPEZOID)
    mapping.clear_cache()
    assert mapping.cache_info()["entries"] == 0
    points = np.array([[0.1, 0.2], [-0.3, 0.4]])
    first = mapping.jacobian(points)
    second = mapping.jacobian(points)
    assert first is second
    assert mapping.cache_info()["entries"] == 1
    mapping.metric_tensor(points)
    mapping.inverse_jacobian(points)
    mapping.jacobian_determinant(points)
    mapping.jacobian_transpose(points)
    assert mapping.cache_info()["entries"] == 5
    mapping.jacobian(points + 0.05)
    assert mapping.cache_info()["entries"] == 6
    mapping.clear_cache()
    assert mapping.cache_info()["entries"] == 0


@pytest.mark.parametrize("mapping", all_mappings(), ids=IDS)
def test_cached_arrays_are_read_only(mapping: GeometricMapping) -> None:
    """A shared cached batch cannot be corrupted by one of its consumers."""
    points = mapping.sample_points()
    for array in (
        mapping.jacobian(points),
        mapping.metric_tensor(points),
        mapping.inverse_jacobian(points),
    ):
        with pytest.raises(ValueError):
            array[0, 0, 0] = 99.0


# ---- element geometry --------------------------------------------------------


def test_element_geometry_of_a_known_triangle() -> None:
    """Centroid, bounding box, edges, diameter, aspect ratio, quality."""
    mapping = AffineMapping(CellType.TRIANGLE, GOOD_TRIANGLE)
    assert np.allclose(mapping.centroid, [5 / 3, 2.0])
    lower, upper = mapping.bounding_box()
    assert np.allclose(lower, [1.0, 1.0]) and np.allclose(upper, [3.0, 4.0])
    # Reference edges are ((1,2), (2,0), (0,1)): hypotenuse, left, bottom.
    assert np.allclose(mapping.edge_lengths(), [np.hypot(2.0, 3.0), 3.0, 2.0])
    assert mapping.diameter == pytest.approx(np.hypot(2.0, 3.0))
    assert mapping.characteristic_length == pytest.approx(mapping.diameter)
    assert mapping.aspect_ratio == pytest.approx(np.hypot(2.0, 3.0) / 2.0)
    assert 0.0 < mapping.quality <= 1.0


def test_identity_has_unit_quality_and_unit_scaling() -> None:
    """The reference element mapped to itself is the perfectly conditioned case."""
    for cell in (CellType.LINE, CellType.TRIANGLE, CellType.QUADRILATERAL):
        mapping = IdentityMapping(cell)
        assert mapping.quality == pytest.approx(1.0)
        assert np.allclose(mapping.volume_scale(mapping.sample_points()), 1.0)


def test_quality_degrades_with_distortion_and_goes_negative_when_inverted() -> None:
    """The scaled Jacobian falls as the element skews, and changes sign when inverted."""
    good = AffineMapping(CellType.TRIANGLE, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    skewed = AffineMapping(CellType.TRIANGLE, [[0.0, 0.0], [1.0, 0.0], [0.9, 0.2]])
    inverted = AffineMapping(CellType.TRIANGLE, [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert good.quality == pytest.approx(1.0)
    assert 0.0 < skewed.quality < good.quality
    assert inverted.quality < 0.0


def test_centroid_is_the_image_of_the_reference_centroid_not_an_integral() -> None:
    """Documented choice: no integration is available at this layer."""
    mapping = IsoparametricMapping(LagrangeInterpolation(CellType.QUADRILATERAL, 1), TRAPEZOID)
    assert np.allclose(
        mapping.centroid, mapping.map(mapping.reference_element.centroid.reshape(1, -1))[0]
    )


# ---- placeholders ------------------------------------------------------------


@pytest.mark.parametrize("placeholder", [CurvilinearMapping, NURBSMapping, HighOrderMapping])
def test_placeholders_refuse_and_name_their_blocker(
    placeholder: type[_FutureMapping],
) -> None:
    """Each declared mapping raises on construction and says what it needs."""
    with pytest.raises(NotImplementedError, match="declared placeholder"):
        placeholder()
    assert placeholder.BLOCKED_BY
    assert placeholder.PROVISIONAL_METADATA["mapping_type"]
    assert placeholder.PROVISIONAL_METADATA["needs"]


def test_nurbs_is_blocked_in_the_interpolation_layer_not_here() -> None:
    """A rational basis is not a polynomial ShapeFunctionFamily; that comes first."""
    assert "rational" in NURBSMapping.BLOCKED_BY
    assert "interpolation layer" in NURBSMapping.BLOCKED_BY


def test_high_order_geometry_already_works_mathematically() -> None:
    """The placeholder is honest that IsoparametricMapping already covers the maths.

    A cubic geometry maps today. What HighOrderMapping would add is what makes it
    trustworthy at high order, which is why the class documents its blockers
    rather than pretending the mathematics is missing.
    """
    assert "already handles high-order geometry" in HighOrderMapping.BLOCKED_BY
    corners = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    interpolation = LagrangeInterpolation(CellType.TRIANGLE, 3)
    nodes = np.array(
        [
            corners[0] * (1 - a - b) + corners[1] * a + corners[2] * b
            for a, b in interpolation.node_locations
        ]
    )
    mapping = IsoparametricMapping(interpolation, nodes)
    mapping.validate()
    mapping.verify()
    assert mapping.is_affine  # straight-sided cubic geometry is still affine


def test_mapping_type_flags() -> None:
    """The enum reports what is implemented and what has a constant Jacobian."""
    assert MappingType.AFFINE.is_implemented and MappingType.IDENTITY.is_implemented
    assert MappingType.ISOPARAMETRIC.is_implemented
    assert not MappingType.NURBS.is_implemented
    assert MappingType.AFFINE.has_constant_jacobian
    assert not MappingType.ISOPARAMETRIC.has_constant_jacobian


# ---- regression: the frozen maps ---------------------------------------------


def test_regression_affine_triangle_parameters() -> None:
    """A = diag(2, 3), b = (1, 1) for the canonical test triangle."""
    mapping = AffineMapping(CellType.TRIANGLE, GOOD_TRIANGLE)
    assert np.allclose(mapping.linear, [[2.0, 0.0], [0.0, 3.0]], atol=1e-12)
    assert np.allclose(mapping.translation, [1.0, 1.0], atol=1e-12)
    assert np.allclose(mapping.jacobian_determinant([[0.0, 0.0]]), 6.0)


def test_regression_bilinear_jacobian_at_the_centre() -> None:
    """The trapezoid's Jacobian at the reference centre, pinned."""
    mapping = IsoparametricMapping(LagrangeInterpolation(CellType.QUADRILATERAL, 1), TRAPEZOID)
    jacobian = mapping.jacobian([[0.0, 0.0]])[0]
    # dx/dxi = ((x1-x0) + (x2-x3))/4, dx/deta = ((x3-x0) + (x2-x1))/4
    assert np.allclose(jacobian, [[1.25, 0.25], [0.25, 0.75]], atol=1e-12)
    assert mapping.jacobian_determinant([[0.0, 0.0]])[0] == pytest.approx(0.875)


def test_regression_line_mapping_uses_the_bi_unit_convention() -> None:
    """J = (x1 - x0)/2 follows from the reference line being [-1, 1], not [0, 1]."""
    mapping = AffineMapping(CellType.LINE, [[0.0], [10.0]])
    assert np.allclose(mapping.jacobian([[0.0]]), [[[5.0]]])
    assert mapping.physical_measure() == pytest.approx(10.0)


# ---- independence ------------------------------------------------------------


def test_module_needs_no_quadrature() -> None:
    """The success criterion: mapping is independent of quadrature.

    Only quadrature is asserted, not assembly or materials: the top-level
    ``nanofem`` package eagerly re-exports the analysis classes, so importing
    *any* submodule loads those regardless of what this layer depends on. The
    layer rule they belong to is enforced by the import-linter contracts
    instead; this test makes the claim that is actually about this module.
    """
    import subprocess
    import sys

    script = (
        "import sys;"
        "from nanofem.numerics.mapping import AffineMapping;"
        "m = AffineMapping('triangle', [[1.0, 1.0], [3.0, 1.0], [1.0, 4.0]]);"
        "m.map([[0.25, 0.25]]);"
        "m.inverse_map([[2.0, 2.0]]);"
        "m.jacobian([[0.25, 0.25]]);"
        "m.validate();"
        "m.verify();"
        "loaded = set(sys.modules);"
        "assert not [x for x in loaded if 'quadrature' in x], 'quadrature imported';"
        "print('standalone OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "standalone OK" in result.stdout


# ---- scale invariance --------------------------------------------------------


def test_a_nanometre_element_is_not_degenerate() -> None:
    """Degeneracy is scale-invariant, which this package cannot do without.

    A 1 nm triangle in SI units has an area scaling near 1e-18. Any absolute
    threshold on the measure scaling - the obvious way to write this check -
    would reject it as degenerate, in a library built for MEMS and NEMS
    geometry. The criterion is the ratio of singular values, which is
    dimensionless.
    """
    for scale in (1.0, 1e-3, 1e-6, 1e-9, 1e-12):
        mapping = AffineMapping(CellType.TRIANGLE, [[0.0, 0.0], [scale, 0.0], [0.0, scale]])
        mapping.validate()
        mapping.verify()
        assert mapping.physical_measure() == pytest.approx(0.5 * scale**2)
        assert mapping.quality == pytest.approx(1.0)
        assert mapping.aspect_ratio == pytest.approx(np.sqrt(2.0))


def test_degeneracy_survives_the_precision_loss_of_the_gram_determinant() -> None:
    """A rank-deficient map is caught despite det(J^T J) landing near machine epsilon.

    Squaring the condition number leaves the determinant around 1e-16 and the
    measure scaling around its square root, 1e-8 - far above any threshold a
    reader would think to write. The singular values lose nothing, so the
    collapse is unambiguous.
    """
    mapping = AffineMapping(CellType.TRIANGLE, [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    point = np.array([[0.25, 0.25]])
    gram_determinant = float(np.linalg.det(mapping.metric_tensor(point)[0]))
    assert abs(gram_determinant) < 1e-14  # not identically zero
    assert float(mapping.volume_scale(point)[0]) > 1e-9  # and its root is much larger
    singular = np.linalg.svd(mapping.jacobian(point)[0], compute_uv=False)
    assert singular[-1] < 1e-15 * singular[0]  # yet the rank deficiency is plain
    with pytest.raises(DegenerateCellError, match="rank deficient"):
        mapping.validate()
