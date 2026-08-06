"""The verification suite, and a deliberately broken basis to trip each check.

A verification that never fires is a decoration. Each identity here is broken on
purpose and the corresponding check is required to catch it - and the
corruptions are chosen to show *which* check catches *which* bug class, since
they are not interchangeable.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from nanofem.numerics.interpolation import (
    HermiteInterpolation,
    HermiteShapeFunctions,
    InterpolationNode,
    LagrangeInterpolation,
    LagrangeShapeFunctions,
    PointsLike,
    ShapeFunctionError,
    ShapeFunctionFamily,
    UnisolvenceError,
    available_interpolations,
    interpolation,
    shape_functions,
)
from nanofem.numerics.interpolation.polynomial import PolynomialSpace
from nanofem.numerics.reference.enums import CellType, EntityType

ALL: list[ShapeFunctionFamily] = [
    shape_functions(interpolation(f, c, k)) for f, c, k in available_interpolations()
]
IDS = [f"{b.family.value}-{b.interpolation.cell_type.value}-{b.order}" for b in ALL]


# ---- the suite passes on every implemented element ---------------------------


@pytest.mark.parametrize("basis", ALL, ids=IDS)
def test_every_basis_verifies(basis: ShapeFunctionFamily) -> None:
    """Every implemented element satisfies every identity."""
    basis.verify()
    assert basis.is_valid()


@pytest.mark.parametrize("basis", ALL, ids=IDS)
def test_duality_matrix_is_the_identity(basis: ShapeFunctionFamily) -> None:
    """l_k(N_i) = delta_ki: the definition the construction is supposed to satisfy."""
    assert np.allclose(basis.duality_matrix(), np.eye(basis.num_functions), atol=1e-10)


@pytest.mark.parametrize("basis", ALL, ids=IDS)
def test_construction_satisfies_c_equals_m_inverse_transpose(
    basis: ShapeFunctionFamily,
) -> None:
    """C M^T = I and M^T C = I, the two faces of C = M^-T."""
    matrix = basis.interpolation.unisolvence_matrix()
    identity = np.eye(basis.num_functions)
    assert np.allclose(basis.coefficients @ matrix.T, identity, atol=1e-9)
    assert np.allclose(matrix.T @ basis.coefficients, identity, atol=1e-9)


@pytest.mark.parametrize("basis", ALL, ids=IDS)
def test_partition_of_unity_holds_pointwise(basis: ShapeFunctionFamily) -> None:
    """Value-dual functions sum to 1; their gradients sum to 0."""
    points = basis.verification_points()
    weights = np.array([1.0 if dof.order == 0 else 0.0 for dof in basis.interpolation.dofs])
    assert np.allclose(basis.evaluate(points) @ weights, 1.0, atol=1e-10)
    assert np.allclose(np.einsum("pid,i->pd", basis.gradient(points), weights), 0.0, atol=1e-9)


def test_hermite_derivative_duals_do_not_join_the_partition_of_unity() -> None:
    """The Hermite slope functions sum to something non-constant, as theory says.

    Phase 3 predicted this from the functionals alone (l_k(1) = 0 on derivative
    dofs); here it is on the built basis. H2 + H4 = (xi^3 - xi)/2.
    """
    basis = shape_functions(HermiteInterpolation(CellType.LINE, 3))
    xi = np.linspace(-1.0, 1.0, 9).reshape(-1, 1)
    values = basis.evaluate(xi)
    slope_sum = values[:, 1] + values[:, 3]
    assert np.allclose(slope_sum, (xi[:, 0] ** 3 - xi[:, 0]) / 2.0, atol=1e-12)
    assert not np.allclose(slope_sum, 0.0)


@pytest.mark.parametrize("basis", ALL, ids=IDS)
def test_polynomial_reproduction_of_every_monomial(basis: ShapeFunctionFamily) -> None:
    """Interpolating each monomial's dof vector returns that monomial."""
    interp = basis.interpolation
    points = basis.verification_points()
    matrix = interp.unisolvence_matrix()
    values = basis.evaluate(points)
    for column, alpha in enumerate(interp.polynomial_space.exponents):
        reproduced = values @ matrix[:, column]
        exact = np.prod(points ** np.asarray(alpha, dtype=np.float64), axis=1)
        assert np.allclose(
            reproduced, exact, atol=1e-9
        ), f"{interp.basis_ids[column]} not reproduced"


@pytest.mark.parametrize("basis", ALL, ids=IDS)
def test_gradient_matches_central_finite_differences(basis: ShapeFunctionFamily) -> None:
    """An independent numerical derivative agrees with the analytic one."""
    points = basis.verification_points()
    analytic = basis.gradient(points)
    step = 1e-6
    for axis in range(basis.interpolation.support_dimension):
        offset = np.zeros(basis.interpolation.support_dimension)
        offset[axis] = step
        numeric = (basis.evaluate(points + offset) - basis.evaluate(points - offset)) / (2.0 * step)
        assert np.allclose(analytic[:, :, axis], numeric, atol=1e-6)


@pytest.mark.parametrize("basis", ALL, ids=IDS)
def test_hessian_matches_central_finite_differences_of_the_gradient(
    basis: ShapeFunctionFamily,
) -> None:
    """Mixed partials computed by two independent numerical routes agree."""
    points = basis.verification_points()
    analytic = basis.hessian(points)
    step = 1e-5
    for axis in range(basis.interpolation.support_dimension):
        offset = np.zeros(basis.interpolation.support_dimension)
        offset[axis] = step
        numeric = (basis.gradient(points + offset) - basis.gradient(points - offset)) / (2.0 * step)
        assert np.allclose(analytic[:, :, axis, :], numeric, atol=1e-6)


@pytest.mark.parametrize("basis", ALL, ids=IDS)
def test_hessian_is_symmetric(basis: ShapeFunctionFamily) -> None:
    """Mixed partials commute."""
    hessian = basis.hessian(basis.verification_points())
    assert np.allclose(hessian, np.swapaxes(hessian, -1, -2), atol=1e-12)


@pytest.mark.parametrize("basis", ALL, ids=IDS)
def test_boundary_restriction_makes_conformity_possible(
    basis: ShapeFunctionFamily,
) -> None:
    """Functions from nodes off a facet vanish on it, so neighbours can agree."""
    interp = basis.interpolation
    element = interp.reference_element
    for facet_index in range(element.num_facets):
        facet = set(element.facet_vertex_indices[facet_index])
        points = basis._facet_sample_points(facet_index)
        values = basis.evaluate(points)
        for index, dof in enumerate(interp.dofs):
            node = interp.nodes[dof.node_index]
            on_facet = (node.entity is EntityType.VERTEX and node.entity_index in facet) or (
                node.entity is EntityType.EDGE
                and element.facet_entity_type is EntityType.EDGE
                and node.entity_index == facet_index
            )
            if not on_facet:
                assert np.allclose(
                    values[:, index], 0.0, atol=1e-10
                ), f"{interp.shape_function_ids[index]} survives on facet {facet_index}"


def test_tri6_edge_trace_is_the_line3_basis() -> None:
    """The restriction to an edge *is* the lower-dimensional element's basis.

    This is conformity made concrete: along edge 0 the three surviving functions
    reproduce the quadratic line basis in the edge parameter, so a neighbouring
    triangle sharing that edge sees exactly the same trace.
    """
    basis = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 2))
    line = shape_functions(LagrangeInterpolation(CellType.LINE, 2))
    element = basis.interpolation.reference_element
    start, end = element.vertex_coordinates[1], element.vertex_coordinates[2]
    t = np.linspace(0.0, 1.0, 9)
    edge_points = start + t.reshape(-1, 1) * (end - start)
    values = basis.evaluate(edge_points)
    # Edge 0 runs V1 -> V2 and carries nodes 1, 2 (its ends) and 3 (its midpoint);
    # the line element orders its nodes as (start, end, midpoint) too.
    line_values = line.evaluate((2.0 * t - 1.0).reshape(-1, 1))
    assert np.allclose(values[:, 1], line_values[:, 0], atol=1e-12)
    assert np.allclose(values[:, 2], line_values[:, 1], atol=1e-12)
    assert np.allclose(values[:, 3], line_values[:, 2], atol=1e-12)


@pytest.mark.parametrize("basis", ALL, ids=IDS)
def test_interpolation_operator_is_exact_on_the_space(basis: ShapeFunctionFamily) -> None:
    """The public interpolate() reproduces an arbitrary member of the space."""
    interp = basis.interpolation
    rng = np.random.default_rng(20260717)
    weights = rng.normal(size=interp.space_dimension)
    dof_values = interp.unisolvence_matrix() @ weights
    points = basis.verification_points()
    interpolated = basis.interpolate(dof_values, points)
    exact = np.array(
        [
            sum(
                w * float(np.prod(point ** np.asarray(alpha, dtype=np.float64)))
                for w, alpha in zip(weights, interp.polynomial_space.exponents, strict=True)
            )
            for point in points
        ]
    )
    assert np.allclose(interpolated, exact, atol=1e-8)


def test_interpolate_supports_vector_fields() -> None:
    """A (n_dofs, n_components) dof array interpolates componentwise."""
    basis = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 1))
    dof_values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    at_centroid = basis.interpolate(dof_values, [[1 / 3, 1 / 3]])
    assert at_centroid.shape == (1, 2)
    assert np.allclose(at_centroid, [[2.0, 20.0]])


def test_interpolate_rejects_a_wrong_dof_count() -> None:
    """The operator checks its input against the basis size."""
    from nanofem.utils.exceptions import InputValidationError

    basis = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 1))
    with pytest.raises(InputValidationError, match="expected 3"):
        basis.interpolate([1.0, 2.0], [[0.25, 0.25]])


# ---- trip tests: each check catches a distinct bug class ----------------------


def _corrupt_lagrange(interp: LagrangeInterpolation, delta: np.ndarray) -> LagrangeShapeFunctions:
    """A Lagrange basis whose coefficient matrix is wrong by ``delta``."""

    class _Corrupted(LagrangeShapeFunctions):
        @property
        def coefficients(self) -> NDArray[np.float64]:
            perturbed: NDArray[np.float64] = (
                np.asarray(super().coefficients, dtype=np.float64) + delta
            )
            perturbed.setflags(write=False)
            return perturbed

    return _Corrupted(interp)


def _corrupt_hermite(interp: HermiteInterpolation, delta: np.ndarray) -> HermiteShapeFunctions:
    """A Hermite basis whose coefficient matrix is wrong by ``delta``."""

    class _Corrupted(HermiteShapeFunctions):
        @property
        def coefficients(self) -> NDArray[np.float64]:
            perturbed: NDArray[np.float64] = (
                np.asarray(super().coefficients, dtype=np.float64) + delta
            )
            perturbed.setflags(write=False)
            return perturbed

    return _Corrupted(interp)


def test_wrong_coefficients_break_kronecker_and_reproduction() -> None:
    """A bad C is caught by duality and by reproduction against the naive reference."""
    interp = LagrangeInterpolation(CellType.TRIANGLE, 1)
    delta = np.zeros((3, 3))
    delta[0, 0] = 0.1  # shift N_0 by a constant
    broken = _corrupt_lagrange(interp, delta)
    with pytest.raises(ShapeFunctionError, match="duality"):
        broken.verify_kronecker_delta()
    with pytest.raises(ShapeFunctionError, match="not.*reproduced"):
        broken.verify_polynomial_reproduction()
    with pytest.raises(ShapeFunctionError, match="sum to 1"):
        broken.verify_partition_of_unity()
    with pytest.raises(ShapeFunctionError, match="vanish on facet"):
        broken.verify_boundary_restriction()
    with pytest.raises(ShapeFunctionError):
        broken.verify_interpolation_exactness()
    assert not broken.is_valid()


def test_finite_differences_catch_a_bug_the_algebra_cannot() -> None:
    """A wrong derivative tabulation passes the identity but fails the FD check.

    The analytic identity ``sum_i M[i,j] D^a N_i = D^a m_j`` collapses to
    ``table C^T M = table`` on both sides, so it holds for *any* derivative
    tabulation. Only an independent derivative catches this, which is why the
    finite-difference route is the standard rather than a convenience.
    """
    interp = LagrangeInterpolation(CellType.TRIANGLE, 2)

    class _BadDerivatives(LagrangeShapeFunctions):
        def derivative(
            self, points: PointsLike, multi_index: tuple[int, ...] | None = None
        ) -> np.ndarray:
            values = super().derivative(points, multi_index)
            if multi_index is not None and any(multi_index):
                return np.asarray(values) * 1.5  # plausible, systematic, and wrong
            return values

    broken = _BadDerivatives(interp)
    broken.verify_kronecker_delta()  # the value dofs are untouched, so duality holds
    broken.verify_polynomial_reproduction()  # values are untouched too
    with pytest.raises(ShapeFunctionError, match="finite difference"):
        broken.verify_derivative_consistency()


def test_partition_of_unity_catches_a_broken_gradient_sum() -> None:
    """A gradient that does not sum to zero is caught even if the values do."""
    interp = LagrangeInterpolation(CellType.LINE, 2)

    class _TiltedGradient(LagrangeShapeFunctions):
        def gradient(self, points: PointsLike) -> np.ndarray:
            return np.asarray(super().gradient(points)) + 0.25

    with pytest.raises(ShapeFunctionError, match="partition of unity is not zero"):
        _TiltedGradient(interp).verify_partition_of_unity()


def test_symmetry_catches_an_asymmetric_hessian() -> None:
    """An asymmetric Hessian is caught."""
    interp = LagrangeInterpolation(CellType.QUADRILATERAL, 2)

    class _AsymmetricHessian(LagrangeShapeFunctions):
        def hessian(self, points: PointsLike) -> np.ndarray:
            values = np.asarray(super().hessian(points)).copy()
            values[:, :, 0, 1] += 0.5
            return values

    with pytest.raises(ShapeFunctionError, match="not symmetric"):
        _AsymmetricHessian(interp).verify_symmetry()


def test_a_singular_interpolation_cannot_produce_a_basis() -> None:
    """A rank-deficient element raises when the dual basis is requested.

    Phase 3 proves unisolvence before this layer runs, so this guards a
    third-party family that skipped validation - the failure is named, not a
    LinAlgError from deep inside numpy.
    """

    class _CollinearTriangle(LagrangeInterpolation):
        @property
        def nodes(self) -> tuple[InterpolationNode, ...]:
            base = list(super().nodes)
            base[2] = InterpolationNode(2, (0.5, 0.0), EntityType.VERTEX, 2)
            return tuple(base)

    broken = shape_functions(_CollinearTriangle(CellType.TRIANGLE, 1))
    with pytest.raises(UnisolvenceError, match="singular"):
        _ = broken.coefficients


def test_a_non_square_interpolation_is_reported_before_the_solve() -> None:
    """Too few dofs for the space is caught with a clear message."""

    class _UnderDetermined(LagrangeInterpolation):
        @property
        def polynomial_space(self) -> PolynomialSpace:
            return PolynomialSpace.total_degree(2, 2)

    broken = shape_functions(_UnderDetermined(CellType.TRIANGLE, 1))
    with pytest.raises(UnisolvenceError, match="not square"):
        _ = broken.coefficients


def test_hermite_family_check_catches_a_broken_slope_pattern() -> None:
    """The Hermite idiom check fires when the value/slope pattern breaks."""
    interp = HermiteInterpolation(CellType.LINE, 3)
    delta = np.zeros((4, 4))
    delta[1, 0] = 0.2
    broken = _corrupt_hermite(interp, delta)
    with pytest.raises(ShapeFunctionError, match="at node"):
        broken.verify_nodal_value_and_slope_pattern()


def test_lagrange_family_check_catches_a_broken_nodal_identity() -> None:
    """The classical Kronecker check fires on a corrupted nodal basis."""
    interp = LagrangeInterpolation(CellType.LINE, 2)
    delta = np.zeros((3, 3))
    delta[0, 0] = 0.3
    broken = _corrupt_lagrange(interp, delta)
    with pytest.raises(ShapeFunctionError, match="delta_ij"):
        broken.verify_classical_kronecker_delta()
