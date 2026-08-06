"""The verification suite, tensor-product correctness, caching, and regressions.

Each identity is broken on purpose and the corresponding check is required to
catch it. A verification that never fires is a decoration.
"""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest

from nanofem.numerics.quadrature import (
    AVAILABLE_QUADRATURES,
    AdaptiveQuadrature,
    DunavantQuadrature,
    ExactnessError,
    GaussJacobiQuadrature,
    GaussLegendreQuadrature,
    GaussLobattoQuadrature,
    QuadratureFactory,
    QuadratureFamily,
    QuadratureRule,
    SparseGridQuadrature,
    SymmetryError,
    TensorProductQuadrature,
    WeightError,
    available_quadratures,
    exact_monomial_integral,
    monomial_exponents_up_to,
    quadrature,
    symmetry_group,
)
from nanofem.numerics.quadrature.future import _FutureQuadrature
from nanofem.numerics.reference import ReferenceLine, reference_cell, reference_element
from nanofem.numerics.reference.enums import CellType
from nanofem.utils.exceptions import InputValidationError


def every_rule() -> list[QuadratureRule]:
    """A representative rule from every implemented family, at several orders."""
    rules: list[QuadratureRule] = []
    for order in (1, 2, 3, 4, 5):
        rules.append(GaussLegendreQuadrature(order))
        rules.append(GaussLobattoQuadrature(order))
        rules.append(TensorProductQuadrature(CellType.QUADRILATERAL, order))
        rules.append(DunavantQuadrature(order))
    return rules


ALL = every_rule()
IDS = [f"{r.family.value}-{r.cell_type.value}-{r.order}" for r in ALL]


# ---- the suite passes on every implemented rule -------------------------------


@pytest.mark.parametrize("rule", ALL, ids=IDS)
def test_every_rule_verifies(rule: QuadratureRule) -> None:
    """Every rule satisfies every identity."""
    rule.verify()
    assert rule.is_valid()


@pytest.mark.parametrize("rule", ALL, ids=IDS)
def test_exactness_against_the_closed_forms(rule: QuadratureRule) -> None:
    """Every monomial within the declared degree integrates to its analytic value."""
    dimension = rule.topological_dimension
    for exponents in monomial_exponents_up_to(rule.exactness_degree, dimension):
        assert rule.moment(exponents) == pytest.approx(
            exact_monomial_integral(rule.reference_element, exponents), abs=1e-13
        ), f"{IDS[ALL.index(rule)]} monomial {exponents}"


@pytest.mark.parametrize("rule", ALL, ids=IDS)
def test_exactness_is_maximal(rule: QuadratureRule) -> None:
    """Some monomial one degree higher must fail, or the rule understates itself."""
    dimension = rule.topological_dimension
    failures = [
        exponents
        for exponents in monomial_exponents_up_to(rule.exactness_degree + 1, dimension)
        if sum(exponents) == rule.exactness_degree + 1
        and not np.isclose(
            rule.moment(exponents),
            exact_monomial_integral(rule.reference_element, exponents),
            atol=1e-9,
        )
    ]
    assert failures, f"{IDS[ALL.index(rule)]} is stronger than it reports"


@pytest.mark.parametrize("rule", ALL, ids=IDS)
def test_weights_sum_to_the_reference_measure(rule: QuadratureRule) -> None:
    """Weight normalization: integrating 1 gives the domain's measure."""
    assert rule.total_weight == pytest.approx(rule.reference_element.reference_measure, abs=1e-14)
    assert float(rule.normalized_weights.sum()) == pytest.approx(1.0)


@pytest.mark.parametrize("rule", ALL, ids=IDS)
def test_weight_positivity_is_reported_accurately(rule: QuadratureRule) -> None:
    """The declared positivity matches the weights, whichever way it goes."""
    assert rule.has_positive_weights == bool(np.all(rule.weights > 0.0))
    negative = rule.family is QuadratureFamily.DUNAVANT and rule.order == 3
    assert rule.has_positive_weights is not negative


@pytest.mark.parametrize("rule", ALL, ids=IDS)
def test_points_lie_in_the_reference_domain(rule: QuadratureRule) -> None:
    """Every point is inside its domain, checked with the phase-2 containment test."""
    for point in rule.points:
        assert rule.reference_element.contains(point, tol=1e-12)


@pytest.mark.parametrize("rule", ALL, ids=IDS)
def test_classical_rules_are_symmetric(rule: QuadratureRule) -> None:
    """Every rule here claims and possesses the domain's full symmetry."""
    assert rule.declares_symmetry
    assert rule.is_symmetric
    assert len(rule.invariant_symmetries) == len(symmetry_group(rule.reference_element))


# ---- symmetry -----------------------------------------------------------------


def test_symmetry_groups_are_derived_not_declared() -> None:
    """Six for the triangle, eight for the square, two for the line.

    Nothing is hard-coded per cell: the group falls out of trying every vertex
    permutation and keeping the ones an exact, measure-preserving affine map
    realizes. Sixteen of the square's twenty-four permutations admit no such
    map, which is exactly why the dihedral group has eight elements.
    """
    assert len(symmetry_group(reference_element(CellType.LINE))) == 2
    assert len(symmetry_group(reference_element(CellType.TRIANGLE))) == 6
    assert len(symmetry_group(reference_element(CellType.QUADRILATERAL))) == 8


def test_symmetries_map_the_domain_onto_itself() -> None:
    """Each symmetry permutes the vertices and preserves measure."""
    for cell in (CellType.LINE, CellType.TRIANGLE, CellType.QUADRILATERAL):
        element = reference_element(cell)
        for symmetry in symmetry_group(element):
            images = symmetry.apply(element.vertex_coordinates)
            assert np.allclose(np.sort(images, axis=0), np.sort(element.vertex_coordinates, axis=0))
            assert abs(abs(float(np.linalg.det(symmetry.linear))) - 1.0) < 1e-12
            for point in element.vertex_coordinates:
                assert element.contains(symmetry.apply(point.reshape(1, -1))[0])


def test_symmetry_orbits_recover_the_published_structure() -> None:
    """Dunavant's tables are published as orbits; the orbits are recovered from points."""
    assert [len(o) for o in DunavantQuadrature(1).symmetry_orbits] == [1]  # centroid
    assert [len(o) for o in DunavantQuadrature(2).symmetry_orbits] == [3]  # one 3-fold
    assert sorted(len(o) for o in DunavantQuadrature(3).symmetry_orbits) == [1, 3]
    assert sorted(len(o) for o in DunavantQuadrature(4).symmetry_orbits) == [3, 3]
    assert sorted(len(o) for o in DunavantQuadrature(5).symmetry_orbits) == [1, 3, 3]


def test_orbit_members_share_a_weight() -> None:
    """The defining property of an orbit."""
    for rule in ALL:
        for orbit in rule.symmetry_orbits:
            weights = rule.weights[list(orbit)]
            assert np.allclose(weights, weights[0], atol=1e-14)


def test_gauss_grid_orbits_on_the_square() -> None:
    """A 3x3 Gauss grid falls into corners, edge midpoints, and the centre."""
    rule = TensorProductQuadrature(CellType.QUADRILATERAL, 4)
    assert rule.num_points == 9
    assert sorted(len(o) for o in rule.symmetry_orbits) == [1, 4, 4]


# ---- tensor-product correctness ------------------------------------------------


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 7])
def test_tensor_product_is_the_product_of_its_factors(order: int) -> None:
    """Points are the cartesian product; weights are the products."""
    rule = TensorProductQuadrature(CellType.QUADRILATERAL, order)
    line = GaussLegendreQuadrature(order)
    assert rule.num_points == line.num_points**2
    expected_points = np.array(
        list(itertools.product(line.points.ravel(), repeat=2)), dtype=np.float64
    )
    assert np.allclose(rule.points, expected_points)
    expected_weights = np.array(
        [float(np.prod(w)) for w in itertools.product(line.weights, repeat=2)]
    )
    assert np.allclose(rule.weights, expected_weights)
    rule.verify_tensor_product_construction()


def test_tensor_product_is_exact_on_the_full_q_space() -> None:
    """Stronger than its reported total degree, which is P not Q.

    A product of degree-p rules integrates every monomial with per-variable
    degree at most p - the whole space Q_p - while reporting total-degree
    exactness p, because xi^(p+1) fails. Both statements are true and the rule
    reports the conservative one, since that is what the rest of the library
    means by exactness.
    """
    rule = TensorProductQuadrature(CellType.QUADRILATERAL, 3)
    assert rule.exactness_degree == 3
    assert rule.per_variable_exactness == (3, 3)
    # Every monomial of Q_3, including xi^3*eta^3 of total degree 6, is exact:
    for p, q in itertools.product(range(4), repeat=2):
        assert rule.moment((p, q)) == pytest.approx(
            exact_monomial_integral(rule.reference_element, (p, q)), abs=1e-14
        )
    # But xi^4, of total degree 4, is not:
    assert not np.isclose(
        rule.moment((4, 0)), exact_monomial_integral(rule.reference_element, (4, 0))
    )


def test_anisotropic_composition() -> None:
    """Rule composition: different rules per axis, exactly what a field may need."""
    rule = TensorProductQuadrature.from_rules(
        CellType.QUADRILATERAL, [GaussLegendreQuadrature(5), GaussLegendreQuadrature(1)]
    )
    rule.verify()
    assert rule.num_points == 3 * 1
    assert rule.per_variable_exactness == (5, 1)
    assert rule.exactness_degree == 1  # the weakest axis governs total degree
    assert rule.moment((4, 0)) == pytest.approx(
        exact_monomial_integral(rule.reference_element, (4, 0)), abs=1e-14
    )  # exact in the strong direction


def test_anisotropic_products_are_honestly_not_symmetric() -> None:
    """Swapping axes swaps unequal factors, so the diagonals are lost - not a defect."""
    rule = TensorProductQuadrature.from_rules(
        CellType.QUADRILATERAL, [GaussLegendreQuadrature(5), GaussLegendreQuadrature(1)]
    )
    assert not rule.declares_symmetry
    assert not rule.is_symmetric
    assert len(rule.invariant_symmetries) == 4  # the axis reflections survive
    rule.verify()  # a legitimate rule: it is not failed for what it never claimed


def test_lobatto_tensor_products() -> None:
    """Any one-dimensional family lifts; the endpoints survive into the corners."""
    rule = TensorProductQuadrature(CellType.QUADRILATERAL, 3, QuadratureFamily.GAUSS_LOBATTO)
    rule.verify()
    assert rule.base_families == (QuadratureFamily.GAUSS_LOBATTO,) * 2
    corners = reference_element(CellType.QUADRILATERAL).vertex_coordinates
    for corner in corners:
        assert np.any(np.all(np.isclose(rule.points, corner), axis=1))


def test_tensor_product_on_the_line_is_the_one_dimensional_rule() -> None:
    """A one-fold product is a degenerate but consistent case."""
    rule = TensorProductQuadrature(CellType.LINE, 5)
    rule.verify()
    assert np.allclose(rule.points, GaussLegendreQuadrature(5).points)
    assert np.allclose(rule.weights, GaussLegendreQuadrature(5).weights)


def test_tensor_product_refuses_a_simplex() -> None:
    """The construction needs independent limits, which a simplex does not have."""
    with pytest.raises(InputValidationError) as excinfo:
        TensorProductQuadrature(CellType.TRIANGLE, 2)
    assert "simplex" in str(excinfo.value) and "coupled" in str(excinfo.value)


def test_tensor_product_validates_its_composition() -> None:
    """The factor count must match the dimension, and factors must be 1-D."""
    with pytest.raises(InputValidationError, match="needs 2"):
        TensorProductQuadrature.from_rules(CellType.QUADRILATERAL, [GaussLegendreQuadrature(3)])
    with pytest.raises(InputValidationError, match="one-dimensional"):
        TensorProductQuadrature.from_rules(
            CellType.QUADRILATERAL,
            [TensorProductQuadrature(CellType.QUADRILATERAL, 1), GaussLegendreQuadrature(1)],
        )
    with pytest.raises(InputValidationError, match="one-dimensional family"):
        TensorProductQuadrature(CellType.QUADRILATERAL, 2, QuadratureFamily.DUNAVANT)


# ---- trip tests: each check catches its own bug ---------------------------------


def test_verify_weight_normalization_catches_a_mis_scaled_rule() -> None:
    """A table transcribed without the area factor is caught."""

    class _Unscaled(DunavantQuadrature):
        @property
        def weights(self) -> np.ndarray:
            return np.asarray(super().weights) * 2.0

    with pytest.raises(WeightError, match="measure"):
        _Unscaled(2).verify_weight_normalization()


def test_verify_weight_positivity_catches_a_false_claim() -> None:
    """A rule reporting positive weights while carrying a negative one is caught."""

    class _LyingRule(DunavantQuadrature):
        @property
        def has_positive_weights(self) -> bool:
            return True

    with pytest.raises(WeightError, match="reports positive weights"):
        _LyingRule(3).verify_weight_positivity()  # degree 3 really is negative


def test_verify_points_in_domain_catches_a_stray_point() -> None:
    """A point outside the reference domain is caught before it integrates anything."""

    class _StrayPoint(DunavantQuadrature):
        @property
        def points(self) -> np.ndarray:
            moved = np.array(super().points)
            moved[0] = [5.0, 5.0]
            return moved

    with pytest.raises(WeightError, match="outside"):
        _StrayPoint(2).verify_points_in_domain()


def test_verify_exactness_catches_a_perturbed_weight() -> None:
    """A weight off by a part in a thousand breaks exactness against the closed forms."""

    class _Perturbed(GaussLegendreQuadrature):
        @property
        def weights(self) -> np.ndarray:
            values = np.array(super().weights)
            values[0] += 1e-3
            return values

    with pytest.raises(ExactnessError, match="exponents"):
        _Perturbed(3).verify_exactness()


def test_verify_exactness_catches_an_overstated_degree() -> None:
    """A rule claiming more than it delivers is caught."""

    class _Overstated(GaussLegendreQuadrature):
        @property
        def exactness_degree(self) -> int:
            return 9  # a two-point rule reaches 3

    with pytest.raises(ExactnessError, match="exponents"):
        _Overstated(3).verify_exactness()


def test_verify_exactness_is_maximal_catches_an_understated_degree() -> None:
    """A rule reporting less than it delivers is caught, so callers do not overpay."""

    class _Understated(GaussLegendreQuadrature):
        @property
        def exactness_degree(self) -> int:
            return 1  # a two-point rule really reaches 3

    with pytest.raises(ExactnessError, match="stronger than the degree"):
        _Understated(3).verify_exactness_is_maximal()


def test_verify_symmetry_catches_a_false_claim() -> None:
    """A rule claiming symmetry while having a lopsided weight is caught."""

    class _Lopsided(GaussLegendreQuadrature):
        @property
        def weights(self) -> np.ndarray:
            values = np.array(super().weights)
            values[0] *= 1.5
            values[-1] /= 1.5
            return values

    with pytest.raises(SymmetryError, match="claims symmetry"):
        _Lopsided(3).verify_symmetry()


def test_verify_moment_identities_catches_a_displaced_rule() -> None:
    """A rule whose first moment misses the centroid is caught."""

    class _Displaced(DunavantQuadrature):
        @property
        def points(self) -> np.ndarray:
            return np.array(super().points) * 0.5

    with pytest.raises(ExactnessError):
        _Displaced(2).verify_moment_identities()


def test_verify_weight_positivity_catches_a_useless_point() -> None:
    """A zero weight means a point that contributes nothing, which is a mistake."""

    class _ZeroWeight(GaussLegendreQuadrature):
        @property
        def weights(self) -> np.ndarray:
            values = np.array(super().weights)
            values[0] = 0.0
            return values

    with pytest.raises(WeightError, match="does nothing"):
        _ZeroWeight(5).verify_weight_positivity()


# ---- the factory and the SDS policy ---------------------------------------------


def test_factory_defaults_per_domain() -> None:
    """Gauss on the line, its product on the square, Dunavant on the triangle."""
    assert quadrature(CellType.LINE, 3).family is QuadratureFamily.GAUSS_LEGENDRE
    assert quadrature(CellType.QUADRILATERAL, 3).family is QuadratureFamily.TENSOR_PRODUCT
    assert quadrature(CellType.TRIANGLE, 3).family is QuadratureFamily.DUNAVANT


def test_factory_never_reduces_implicitly() -> None:
    """The SDS 2.5 policy: what is asked for is delivered, or more, never less."""
    for cell in (CellType.LINE, CellType.QUADRILATERAL, CellType.TRIANGLE):
        for order in (1, 2, 3, 4, 5):
            rule = quadrature(cell, order)
            assert rule.exactness_degree >= order, f"{cell.value} order {order}"


def test_factory_seam_accepts_a_mesh_cell_record() -> None:
    """The phase-0 QuadratureFactory declared a ReferenceCell argument; it is honored."""
    factory = QuadratureFactory()
    rule = factory.rule(reference_cell("tri6"), 4)
    assert rule.family is QuadratureFamily.DUNAVANT
    assert rule.exactness_degree >= 4
    assert factory.rule(reference_cell("quad4"), 2).cell_type is CellType.QUADRILATERAL
    pinned = QuadratureFactory(QuadratureFamily.GAUSS_LOBATTO)
    assert pinned.rule(reference_cell("line3"), 3).family is QuadratureFamily.GAUSS_LOBATTO


def test_factory_accepts_elements_types_and_names() -> None:
    """Four ways to name the domain, one result."""
    by_element = quadrature(ReferenceLine(), 3)
    assert by_element == quadrature(CellType.LINE, 3)
    assert by_element == quadrature("line", 3)
    assert by_element == quadrature("line3", 3)  # a mesh cell name


def test_factory_rejects_unknown_families_and_bad_pairings() -> None:
    """Unknown names and families undefined on a domain raise with context."""
    with pytest.raises(InputValidationError, match="unknown quadrature family"):
        quadrature(CellType.LINE, 3, "simpson")
    with pytest.raises(InputValidationError, match="not defined on the triangle"):
        quadrature(CellType.TRIANGLE, 3, QuadratureFamily.GAUSS_LEGENDRE)
    with pytest.raises(NotImplementedError, match="not implemented"):
        quadrature(CellType.LINE, 3, QuadratureFamily.GAUSS_JACOBI)


def test_available_quadratures_lists_what_exists() -> None:
    """Five (family, domain) combinations."""
    combinations = available_quadratures()
    assert len(combinations) == 5
    assert (QuadratureFamily.DUNAVANT, CellType.TRIANGLE) in combinations
    for family, cell in combinations:
        quadrature(cell, 2, family).verify()
    assert set(AVAILABLE_QUADRATURES) == {
        QuadratureFamily.GAUSS_LEGENDRE,
        QuadratureFamily.GAUSS_LOBATTO,
        QuadratureFamily.TENSOR_PRODUCT,
        QuadratureFamily.DUNAVANT,
    }


# ---- caching and immutability ---------------------------------------------------


def test_identical_requests_return_the_identical_rule() -> None:
    """Rules are immutable value objects, so the factory memoizes them."""
    assert quadrature(CellType.TRIANGLE, 5) is quadrature(CellType.TRIANGLE, 5)
    assert quadrature(CellType.TRIANGLE, 5) is not quadrature(CellType.TRIANGLE, 4)


@pytest.mark.parametrize("rule", ALL, ids=IDS)
def test_points_and_weights_are_read_only(rule: QuadratureRule) -> None:
    """A shared rule cannot be corrupted by one of its consumers."""
    for array in (rule.points, rule.weights, rule.normalized_weights):
        with pytest.raises(ValueError):
            array[0] = 99.0


def test_rules_are_frozen_and_hashable() -> None:
    """Value semantics: equal rules share a hash and work as dictionary keys."""
    from dataclasses import FrozenInstanceError

    a = GaussLegendreQuadrature(5)
    b = GaussLegendreQuadrature(5)
    assert a == b and hash(a) == hash(b)
    assert a != GaussLegendreQuadrature(3)
    assert a != GaussLobattoQuadrature(5)  # same points count, different family
    assert a != object()
    assert len({a, b, GaussLegendreQuadrature(3)}) == 2
    cache = {rule: rule.num_points for rule in ALL}
    assert cache[DunavantQuadrature(5)] == 7
    with pytest.raises(FrozenInstanceError):
        a.order_ = 7  # type: ignore[misc]


@pytest.mark.parametrize("rule", ALL, ids=IDS)
def test_serialization_round_trips_through_json(rule: QuadratureRule) -> None:
    """The record is complete and JSON-safe."""
    data = rule.to_dict()
    assert data["schema"] == "nanofem-quadrature-rule/1"
    assert data["num_points"] == rule.num_points
    assert len(data["points"]) == rule.num_points
    assert len(data["weights"]) == rule.num_points
    assert data["exactness_degree"] == rule.exactness_degree
    json.dumps(data)
    assert rule.to_json() == rule.to_json()
    assert json.loads(rule.to_json())["family"] == rule.family.value


@pytest.mark.parametrize("rule", ALL, ids=IDS)
def test_pretty_and_debug_summaries(rule: QuadratureRule) -> None:
    """Summaries report the rule's key facts."""
    text = rule.pretty()
    assert rule.family.value in text and "exactness" in text
    debug = rule.debug_summary()
    assert "quadrature points:" in debug
    assert debug.count("\n") > text.count("\n")
    assert f"points={rule.num_points}" in repr(rule)


# ---- placeholders ----------------------------------------------------------------


@pytest.mark.parametrize(
    "placeholder", [GaussJacobiQuadrature, AdaptiveQuadrature, SparseGridQuadrature]
)
def test_placeholders_refuse_and_name_their_blocker(
    placeholder: type[_FutureQuadrature],
) -> None:
    """Each declared family raises on construction and says what it needs."""
    with pytest.raises(NotImplementedError, match="declared placeholder"):
        placeholder(3)
    assert placeholder.BLOCKED_BY
    assert placeholder.PROVISIONAL_METADATA["needs"]
    assert not QuadratureFamily(placeholder.PROVISIONAL_METADATA["family"]).is_implemented


def test_adaptive_is_blocked_by_the_interface_not_the_maths() -> None:
    """Its points depend on the integrand, which breaks the shared-tabulation contract."""
    assert "integrand" in AdaptiveQuadrature.BLOCKED_BY
    assert AdaptiveQuadrature.PROVISIONAL_METADATA["points_depend_on_integrand"]


def test_gauss_jacobi_is_named_for_the_physics_that_will_want_it() -> None:
    """Eringen's integral kernel is exactly the non-polynomial weight it absorbs."""
    assert "(1-x)" in GaussJacobiQuadrature.PROVISIONAL_METADATA["weight_function"]
    assert "Jacobi polynomial roots" in GaussJacobiQuadrature.PROVISIONAL_METADATA["needs"]


def test_sparse_grid_earns_its_place_outside_element_integration() -> None:
    """Below four dimensions a full tensor product is already cheap."""
    assert not SparseGridQuadrature.PROVISIONAL_METADATA["has_positive_weights"]
    assert "parameter space" in SparseGridQuadrature.PROVISIONAL_METADATA["useful_dimension"]


# ---- regression: the frozen rules ------------------------------------------------


def test_regression_gauss_tables() -> None:
    """Points and weights pinned against the textbook values."""
    assert np.allclose(GaussLegendreQuadrature(1).points.ravel(), [0.0])
    assert np.allclose(
        GaussLegendreQuadrature(3).points.ravel(), [-0.5773502691896257, 0.5773502691896257]
    )
    assert np.allclose(
        GaussLegendreQuadrature(5).weights,
        [0.5555555555555556, 0.8888888888888888, 0.5555555555555556],
    )
    assert np.allclose(GaussLobattoQuadrature(3).weights, [1 / 3, 4 / 3, 1 / 3])


def test_regression_dunavant_degree_5_table() -> None:
    """The seven-point degree-5 rule, pinned against Dunavant (1985) Table 2."""
    rule = DunavantQuadrature(5)
    normalized = np.sort(rule.weights / rule.total_weight)
    assert np.allclose(
        normalized,
        np.sort(
            [
                0.225000000000000,
                0.132394152788506,
                0.132394152788506,
                0.132394152788506,
                0.125939180544827,
                0.125939180544827,
                0.125939180544827,
            ]
        ),
        atol=1e-14,
    )
    assert normalized.sum() == pytest.approx(1.0, abs=1e-14)


def test_regression_point_counts() -> None:
    """The point count of every default rule, degree 1 to 5."""
    counts = {
        (cell.value, order): quadrature(cell, order).num_points
        for cell in (CellType.LINE, CellType.TRIANGLE, CellType.QUADRILATERAL)
        for order in (1, 2, 3, 4, 5)
    }
    assert counts == {
        ("line", 1): 1,
        ("line", 2): 2,
        ("line", 3): 2,
        ("line", 4): 3,
        ("line", 5): 3,
        ("triangle", 1): 1,
        ("triangle", 2): 3,
        ("triangle", 3): 4,
        ("triangle", 4): 6,
        ("triangle", 5): 7,
        ("quadrilateral", 1): 1,
        ("quadrilateral", 2): 4,
        ("quadrilateral", 3): 4,
        ("quadrilateral", 4): 9,
        ("quadrilateral", 5): 9,
    }


def test_regression_dunavant_beats_the_tensor_product_on_a_triangle() -> None:
    """Why the triangle gets its own family: fewer points for the same degree.

    A degree-5 Dunavant rule needs seven points where a degree-5 product rule on
    the square needs nine - and the product does not even fit a triangle.
    """
    assert DunavantQuadrature(5).num_points == 7
    assert TensorProductQuadrature(CellType.QUADRILATERAL, 5).num_points == 9


# ---- independence -----------------------------------------------------------------


def test_quadrature_does_not_import_the_interpolation_layer() -> None:
    """The constraint that decides this package's shape.

    The spectral interpolation family's nodes *are* Gauss-Lobatto points, so
    ``numerics.interpolation`` will import ``numerics.quadrature`` when it
    lands. The dependency must not already run the other way, or neither
    package could be imported. Monomial evaluation and the closed-form integrals
    are therefore written here rather than borrowed.
    """
    import subprocess
    import sys

    script = (
        "import sys;"
        "from nanofem.numerics.quadrature.rules import QuadratureRule;"
        "from nanofem.numerics.quadrature.dunavant import DunavantQuadrature;"
        "r = DunavantQuadrature(5);"
        "r.verify();"
        "r.integrate(lambda p: p[:, 0]);"
        "loaded = set(sys.modules);"
        "assert not [m for m in loaded if 'interpolation' in m], 'interpolation imported';"
        "assert not [m for m in loaded if 'mapping' in m], 'mapping imported';"
        "print('leaf OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "leaf OK" in result.stdout


def test_quadrature_knows_nothing_about_finite_elements() -> None:
    """No shape function, element, or Jacobian appears in the package's source."""
    from pathlib import Path

    import nanofem.numerics.quadrature as package

    root = Path(package.__file__).parent
    forbidden = ("shape_function", "ShapeFunction", "stiffness", "assembl", "constitutive")
    for source in root.glob("*.py"):
        text = source.read_text()
        code = "\n".join(line for line in text.split("\n") if not line.strip().startswith("#"))
        # Strip docstrings crudely: they discuss these ideas deliberately.
        body = code.split('"""')
        executable = "".join(body[::2])
        for term in forbidden:
            assert term not in executable, f"{source.name} references {term!r}"


def test_math_import_is_used_for_factorials_only() -> None:
    """A guard on the closed forms: the triangle formula really is factorial based."""
    assert math.factorial(2) * math.factorial(2) / math.factorial(6) == pytest.approx(
        exact_monomial_integral(reference_element(CellType.TRIANGLE), (2, 2))
    )


# ---- coverage: the remaining reachable rules each get a test --------------------


def test_integrate_monomial_is_an_alias_of_moment() -> None:
    """The convenience name and the primitive agree."""
    rule = GaussLegendreQuadrature(5)
    assert rule.integrate_monomial((2,)) == rule.moment((2,))
    assert rule.integrate_monomial((4,)) == pytest.approx(rule.exact_moment((4,)))


def test_exact_moment_is_the_analytic_value() -> None:
    """The rule's exact_moment delegates to the closed form, not to itself."""
    rule = DunavantQuadrature(4)
    assert rule.exact_moment((2, 1)) == exact_monomial_integral(rule.reference_element, (2, 1))


def test_gauss_rules_reject_a_non_integer_or_negative_order() -> None:
    """A bad order is caught at construction, before any root finding."""
    for family in (GaussLegendreQuadrature, GaussLobattoQuadrature):
        with pytest.raises(InputValidationError, match="integer >= 0"):
            family(-1)
        with pytest.raises(InputValidationError, match="integer >= 0"):
            family(2.5)  # type: ignore[arg-type]
        with pytest.raises(InputValidationError, match="integer >= 0"):
            family(True)  # a bool is not an order


def test_dunavant_rejects_a_non_integer_order() -> None:
    """The triangle family checks its order type too."""
    with pytest.raises(InputValidationError, match="integer"):
        DunavantQuadrature(2.0)  # type: ignore[arg-type]


def test_moments_module_validates_its_arguments() -> None:
    """The closed-form helpers reject malformed requests."""
    from nanofem.numerics.quadrature import monomial_exponents, monomial_values

    with pytest.raises(InputValidationError, match="degree must be"):
        monomial_exponents(-1, 2)
    with pytest.raises(InputValidationError, match="n_variables"):
        monomial_exponents(2, 0)
    with pytest.raises(InputValidationError, match="shape"):
        monomial_values((1, 0), np.zeros((3, 3)))


def test_is_valid_returns_false_rather_than_raising() -> None:
    """The boolean wrapper swallows the verification errors."""

    class _Unscaled(DunavantQuadrature):
        @property
        def weights(self) -> np.ndarray:
            return np.asarray(super().weights) * 2.0

    assert _Unscaled(2).is_valid() is False


def test_normalized_weights_failure_is_reachable() -> None:
    """A rule whose weights are individually fine but mis-normalized is caught.

    Constructed by making the weight sum correct but the normalization path see
    a different total - the guard exists for a rule that overrides one without
    the other.
    """

    class _BadNormalization(GaussLegendreQuadrature):
        @property
        def normalized_weights(self) -> np.ndarray:
            values = np.array(super().weights)  # unnormalized on purpose
            values.setflags(write=False)
            return values

    with pytest.raises(WeightError, match="normalized weights"):
        _BadNormalization(5).verify_weight_normalization()
