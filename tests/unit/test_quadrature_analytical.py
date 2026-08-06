"""Quadrature against analytical integrals, and exactness at every order.

Every reference here is a closed form derived by hand - the Legendre roots, the
classical Newton-Cotes weights, textbook integrals of specific functions - and
shares no code with the rule under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nanofem.numerics.quadrature import (
    DunavantQuadrature,
    GaussLegendreQuadrature,
    GaussLobattoQuadrature,
    QuadratureRule,
    TensorProductQuadrature,
    exact_monomial_integral,
)
from nanofem.numerics.reference import (
    ReferenceLine,
    ReferenceQuadrilateral,
    ReferenceTriangle,
)
from nanofem.numerics.reference.enums import CellType

SQRT3 = math.sqrt(3.0)


# ---- the closed-form standard -----------------------------------------------


def test_exact_line_integrals_by_hand() -> None:
    """On [-1, 1]: even powers give 2/(n+1), odd powers vanish by antisymmetry."""
    line = ReferenceLine()
    assert exact_monomial_integral(line, (0,)) == pytest.approx(2.0)
    assert exact_monomial_integral(line, (1,)) == 0.0
    assert exact_monomial_integral(line, (2,)) == pytest.approx(2.0 / 3.0)
    assert exact_monomial_integral(line, (3,)) == 0.0
    assert exact_monomial_integral(line, (4,)) == pytest.approx(2.0 / 5.0)


def test_exact_triangle_integrals_by_hand() -> None:
    """On the unit triangle: p! q! / (p + q + 2)!."""
    triangle = ReferenceTriangle()
    assert exact_monomial_integral(triangle, (0, 0)) == pytest.approx(0.5)  # the area
    assert exact_monomial_integral(triangle, (1, 0)) == pytest.approx(1 / 6)
    assert exact_monomial_integral(triangle, (0, 1)) == pytest.approx(1 / 6)
    assert exact_monomial_integral(triangle, (1, 1)) == pytest.approx(1 / 24)
    assert exact_monomial_integral(triangle, (2, 0)) == pytest.approx(1 / 12)
    assert exact_monomial_integral(triangle, (2, 2)) == pytest.approx(1 / 180)
    for p in range(4):
        for q in range(4):
            expected = math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)
            assert exact_monomial_integral(triangle, (p, q)) == pytest.approx(expected)


def test_exact_quadrilateral_integrals_separate() -> None:
    """On [-1, 1]^2 the integral is the product of the one-dimensional ones."""
    quad = ReferenceQuadrilateral()
    line = ReferenceLine()
    assert exact_monomial_integral(quad, (0, 0)) == pytest.approx(4.0)  # the area
    assert exact_monomial_integral(quad, (1, 0)) == 0.0  # one odd factor kills it
    assert exact_monomial_integral(quad, (2, 2)) == pytest.approx(4.0 / 9.0)
    for p in range(5):
        for q in range(5):
            expected = exact_monomial_integral(line, (p,)) * exact_monomial_integral(line, (q,))
            assert exact_monomial_integral(quad, (p, q)) == pytest.approx(expected)


def test_exact_integrals_validate_their_input() -> None:
    """A mismatched or negative exponent tuple raises."""
    from nanofem.utils.exceptions import InputValidationError

    with pytest.raises(InputValidationError, match="entries"):
        exact_monomial_integral(ReferenceTriangle(), (1,))
    with pytest.raises(InputValidationError, match="negative"):
        exact_monomial_integral(ReferenceLine(), (-1,))


# ---- Gauss-Legendre against the textbook ------------------------------------


def test_gauss_one_point_is_the_midpoint_rule() -> None:
    """One point at the origin with weight 2: exact through degree 1."""
    rule = GaussLegendreQuadrature(1)
    assert rule.num_points == 1
    assert np.allclose(rule.points, [[0.0]])
    assert np.allclose(rule.weights, [2.0])
    assert rule.exactness_degree == 1


def test_gauss_two_point_matches_the_closed_form() -> None:
    """Points at +-1/sqrt(3), weights 1: exact through degree 3."""
    rule = GaussLegendreQuadrature(3)
    assert rule.num_points == 2
    assert np.allclose(rule.points.ravel(), [-1.0 / SQRT3, 1.0 / SQRT3])
    assert np.allclose(rule.weights, [1.0, 1.0])
    assert rule.exactness_degree == 3


def test_gauss_three_point_matches_the_closed_form() -> None:
    """Points 0 and +-sqrt(3/5); weights 8/9 and 5/9: exact through degree 5."""
    rule = GaussLegendreQuadrature(5)
    assert rule.num_points == 3
    expected = math.sqrt(3.0 / 5.0)
    assert np.allclose(rule.points.ravel(), [-expected, 0.0, expected])
    assert np.allclose(rule.weights, [5 / 9, 8 / 9, 5 / 9])
    assert rule.exactness_degree == 5


def test_gauss_points_are_the_legendre_roots() -> None:
    """The defining property, checked against an independent Legendre evaluation."""
    for order in (1, 3, 5, 7, 9):
        rule = GaussLegendreQuadrature(order)
        count = rule.num_points
        coefficients = np.zeros(count + 1)
        coefficients[count] = 1.0  # P_n
        values = np.polynomial.legendre.legval(rule.points.ravel(), coefficients)
        assert np.allclose(values, 0.0, atol=1e-12), f"order {order}"


@pytest.mark.parametrize("order", range(0, 12))
def test_gauss_point_count_and_exactness(order: int) -> None:
    """n = ceil((p+1)/2), and n points reach degree 2n-1: the theoretical maximum."""
    rule = GaussLegendreQuadrature(order)
    assert rule.num_points == max(1, math.ceil((order + 1) / 2))
    assert rule.exactness_degree == 2 * rule.num_points - 1
    assert rule.exactness_degree >= order  # never less than asked
    rule.verify()


# ---- Gauss-Lobatto against the textbook -------------------------------------


def test_lobatto_two_point_is_the_trapezoid_rule() -> None:
    """The endpoints with weight 1 each: exact through degree 1."""
    rule = GaussLobattoQuadrature(1)
    assert np.allclose(rule.points.ravel(), [-1.0, 1.0])
    assert np.allclose(rule.weights, [1.0, 1.0])
    assert rule.exactness_degree == 1


def test_lobatto_three_point_is_simpsons_rule() -> None:
    """Weights 1/3, 4/3, 1/3 at -1, 0, 1: Simpson's rule, exact through degree 3."""
    rule = GaussLobattoQuadrature(3)
    assert np.allclose(rule.points.ravel(), [-1.0, 0.0, 1.0])
    assert np.allclose(rule.weights, [1 / 3, 4 / 3, 1 / 3])
    assert rule.exactness_degree == 3


def test_lobatto_four_point_matches_the_closed_form() -> None:
    """Points +-1 and +-1/sqrt(5); weights 1/6 and 5/6: exact through degree 5."""
    rule = GaussLobattoQuadrature(5)
    inner = 1.0 / math.sqrt(5.0)
    assert np.allclose(rule.points.ravel(), [-1.0, -inner, inner, 1.0])
    assert np.allclose(rule.weights, [1 / 6, 5 / 6, 5 / 6, 1 / 6])
    assert rule.exactness_degree == 5


def test_lobatto_always_includes_the_endpoints() -> None:
    """The property the family exists for: collocation with nodal endpoints."""
    for order in range(0, 10):
        rule = GaussLobattoQuadrature(order)
        assert rule.points[0, 0] == pytest.approx(-1.0)
        assert rule.points[-1, 0] == pytest.approx(1.0)


def test_lobatto_interior_points_are_roots_of_the_legendre_derivative() -> None:
    """The defining property, checked independently."""
    for order in (2, 4, 6, 8):
        rule = GaussLobattoQuadrature(order)
        count = rule.num_points
        coefficients = np.zeros(count)
        coefficients[count - 1] = 1.0  # P_{n-1}
        derivative = np.polynomial.legendre.legder(coefficients)
        interior = rule.points.ravel()[1:-1]
        values = np.polynomial.legendre.legval(interior, derivative)
        assert np.allclose(values, 0.0, atol=1e-12), f"order {order}"


@pytest.mark.parametrize("order", range(0, 12))
def test_lobatto_point_count_and_exactness(order: int) -> None:
    """n points reach 2n-3: two degrees behind Gauss, the price of the endpoints."""
    rule = GaussLobattoQuadrature(order)
    assert rule.num_points == max(2, math.ceil((order + 3) / 2))
    assert rule.exactness_degree == 2 * rule.num_points - 3
    assert rule.exactness_degree >= order
    rule.verify()


def test_lobatto_costs_two_degrees_against_gauss_at_equal_points() -> None:
    """The trade the family makes, stated as a test."""
    for count in range(2, 8):
        gauss = GaussLegendreQuadrature(2 * count - 1)
        lobatto = GaussLobattoQuadrature(2 * count - 3)
        assert gauss.num_points == lobatto.num_points == count
        assert gauss.exactness_degree - lobatto.exactness_degree == 2


# ---- Dunavant against the published tables -----------------------------------


def test_dunavant_degree_1_is_the_centroid_rule() -> None:
    """One point at the centroid with weight equal to the area."""
    rule = DunavantQuadrature(1)
    assert rule.num_points == 1
    assert np.allclose(rule.points, [[1 / 3, 1 / 3]])
    assert np.allclose(rule.weights, [0.5])


def test_dunavant_degree_2_is_the_three_point_rule() -> None:
    """Permutations of barycentric (2/3, 1/6, 1/6), each weighted a third of the area."""
    rule = DunavantQuadrature(2)
    assert rule.num_points == 3
    assert np.allclose(
        np.sort(rule.points, axis=0),
        np.sort([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]], axis=0),
    )
    assert np.allclose(rule.weights, 0.5 / 3.0)


def test_dunavant_degree_3_has_the_famous_negative_weight() -> None:
    """Four points reaching degree 3 costs a negative centroid weight of -27/48.

    Not a transcription error - a known property. The rule reports it rather
    than hiding it, so a caller who needs positivity can ask for degree 4.
    """
    rule = DunavantQuadrature(3)
    assert rule.num_points == 4
    assert not rule.has_positive_weights
    normalized = rule.weights / rule.total_weight
    assert normalized.min() == pytest.approx(-27.0 / 48.0)
    assert sorted(normalized)[1] == pytest.approx(25.0 / 48.0)
    rule.verify()  # a legitimate rule: it is not failed for being negative
    assert DunavantQuadrature(4).has_positive_weights  # the positive alternative


@pytest.mark.parametrize(("degree", "count"), [(1, 1), (2, 3), (3, 4), (4, 6), (5, 7)])
def test_dunavant_point_counts_match_the_paper(degree: int, count: int) -> None:
    """The published point counts for degrees 1 to 5."""
    rule = DunavantQuadrature(degree)
    assert rule.num_points == count
    assert rule.exactness_degree == degree  # tables are per degree: no overshoot
    rule.verify()


def test_dunavant_points_are_inside_and_barycentric_consistent() -> None:
    """Every point lies in the triangle with barycentric coordinates summing to one.

    Within one unit in the last place. Deriving ``b = (1-a)/2`` from the
    generator removes the *transcription* error - a table listing ``a`` and ``b``
    to fifteen digits leaves their sum off by around 1e-15 - but recovering the
    barycentric coordinates from the stored cartesian ones costs half an epsilon
    back, and nothing can avoid that.
    """
    for degree in (1, 2, 3, 4, 5):
        rule = DunavantQuadrature(degree)
        for point in rule.points:
            barycentric = np.array([1.0 - point[0] - point[1], point[0], point[1]])
            assert barycentric.sum() == pytest.approx(1.0, abs=1e-15)
            assert rule.reference_element.contains(point)


def test_dunavant_rejects_untabulated_degrees() -> None:
    """A degree the table does not cover raises, and says where the rest are."""
    from nanofem.utils.exceptions import InputValidationError

    with pytest.raises(InputValidationError) as excinfo:
        DunavantQuadrature(6)
    assert "degree 20" in str(excinfo.value)
    with pytest.raises(InputValidationError):
        DunavantQuadrature(0)


# ---- integrating actual functions --------------------------------------------


def test_integrate_a_polynomial_on_the_line() -> None:
    """integral of 3x^2 + 2x + 1 over [-1, 1] is 4, by hand."""
    rule = GaussLegendreQuadrature(2)
    value = rule.integrate(lambda p: 3.0 * p[:, 0] ** 2 + 2.0 * p[:, 0] + 1.0)
    assert value == pytest.approx(4.0)


def test_integrate_a_transcendental_function_converges() -> None:
    """integral of exp(x) over [-1, 1] is e - 1/e; a Gauss rule converges fast.

    Not exact - exp is not a polynomial - which is exactly the point: exactness
    is a statement about polynomials, and everything else is approximation that
    improves with order.
    """
    exact = math.e - 1.0 / math.e
    errors = []
    for order in (1, 3, 5, 7, 9, 11, 13, 15):
        rule = GaussLegendreQuadrature(order)
        value = rule.integrate(lambda p: np.exp(p[:, 0]))
        errors.append(abs(float(value) - exact))
    assert errors == sorted(errors, reverse=True)  # monotone improvement
    # Exponential, not algebraic: eight points reach machine precision on a smooth
    # integrand, which is the practical reason Gaussian quadrature is worth its roots.
    assert errors[0] > 0.1
    assert errors[-1] < 1e-14
    assert errors[3] / errors[4] > 100.0  # each added point buys orders of magnitude


def test_integrate_over_the_triangle_by_hand() -> None:
    """integral of (1 - x - y) over the unit triangle is 1/6: the barycentric mean."""
    rule = DunavantQuadrature(2)
    value = rule.integrate(lambda p: 1.0 - p[:, 0] - p[:, 1])
    assert value == pytest.approx(1.0 / 6.0)


def test_integrate_over_the_square_by_hand() -> None:
    """integral of x^2 y^2 over [-1,1]^2 is 4/9."""
    rule = TensorProductQuadrature(CellType.QUADRILATERAL, 4)
    value = rule.integrate(lambda p: p[:, 0] ** 2 * p[:, 1] ** 2)
    assert value == pytest.approx(4.0 / 9.0)


def test_integrate_a_vector_field_componentwise() -> None:
    """A field with extra axes is integrated component by component."""
    rule = DunavantQuadrature(2)
    result = rule.integrate(lambda p: np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.allclose(result, [1 / 6, 1 / 6, 0.5])


def test_integrate_values_takes_a_pre_evaluated_field() -> None:
    """The form a caller with a cached tabulation uses."""
    rule = GaussLegendreQuadrature(3)
    values = rule.points[:, 0] ** 2
    assert rule.integrate_values(values) == pytest.approx(2.0 / 3.0)
    assert rule.integrate_values(values) == rule.integrate(lambda p: p[:, 0] ** 2)


def test_integrate_values_rejects_a_mismatched_length() -> None:
    """A field evaluated somewhere else is rejected rather than broadcast."""
    from nanofem.utils.exceptions import InputValidationError

    rule = GaussLegendreQuadrature(3)
    with pytest.raises(InputValidationError, match="points"):
        rule.integrate_values(np.zeros(7))


# ---- moments, measure, centroid ----------------------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        GaussLegendreQuadrature(5),
        GaussLobattoQuadrature(5),
        DunavantQuadrature(5),
        TensorProductQuadrature(CellType.QUADRILATERAL, 5),
    ],
    ids=["gauss", "lobatto", "dunavant", "tensor"],
)
def test_measure_and_centroid(rule: QuadratureRule) -> None:
    """Integrating 1 gives the measure; the first moment gives the centroid.

    The centroid computed by quadrature must agree with the one the reference
    element reports from its vertices - a cross-check between phase 2 and here.
    """
    assert rule.measure() == pytest.approx(rule.reference_element.reference_measure)
    assert np.allclose(rule.centroid(), rule.reference_element.centroid, atol=1e-14)


def test_triangle_centroid_is_one_third_one_third() -> None:
    """The value by hand, recovered by integration."""
    assert np.allclose(DunavantQuadrature(2).centroid(), [1 / 3, 1 / 3])


def test_moments_are_cached_per_rule() -> None:
    """A repeated moment is not recomputed."""
    rule = DunavantQuadrature(5)
    first = rule.moment((2, 1))
    assert (2, 1) in rule._moment_cache
    assert rule.moment((2, 1)) == first
