"""Polynomial spaces and DOF functionals: the building blocks of the framework.

No shape function is formed anywhere in the module under test; these exercise
monomial exponent sets and the functionals that will later be dualized.
"""

from __future__ import annotations

import math

import pytest

from nanofem.numerics.interpolation import (
    DofFunctional,
    DofKind,
    InterpolationNode,
    PolynomialSpace,
    PolynomialSpaceType,
    monomial_label,
)
from nanofem.numerics.reference.enums import EntityType
from nanofem.utils.exceptions import InputValidationError

# ---- polynomial spaces ------------------------------------------------------


@pytest.mark.parametrize(
    ("order", "n_variables", "expected"),
    [(1, 1, 2), (2, 1, 3), (3, 1, 4), (1, 2, 3), (2, 2, 6), (3, 2, 10), (1, 3, 4), (2, 3, 10)],
)
def test_total_degree_dimension_matches_binomial(
    order: int, n_variables: int, expected: int
) -> None:
    """dim P_k = C(k + d, d)."""
    space = PolynomialSpace.total_degree(order, n_variables)
    assert space.dimension == expected == math.comb(order + n_variables, n_variables)
    assert space.dimension == space.expected_dimension


@pytest.mark.parametrize(
    ("order", "n_variables", "expected"),
    [(1, 1, 2), (2, 1, 3), (1, 2, 4), (2, 2, 9), (3, 2, 16), (1, 3, 8), (2, 3, 27)],
)
def test_tensor_product_dimension_matches_power(
    order: int, n_variables: int, expected: int
) -> None:
    """dim Q_k = (k + 1)^d."""
    space = PolynomialSpace.tensor_product(order, n_variables)
    assert space.dimension == expected == (order + 1) ** n_variables
    assert space.dimension == space.expected_dimension


def test_p_and_q_coincide_in_one_variable() -> None:
    """In 1-D the total-degree and tensor-product spaces are the same set."""
    for order in (1, 2, 3):
        assert (
            PolynomialSpace.total_degree(order, 1).exponents
            == PolynomialSpace.tensor_product(order, 1).exponents
        )


def test_completeness_degree_differs_from_max_total_degree_for_q_spaces() -> None:
    """Q_2 in 2-D reaches total degree 4 but is complete only to degree 2."""
    q2 = PolynomialSpace.tensor_product(2, 2)
    assert q2.max_total_degree == 4  # xi^2 * eta^2
    assert q2.completeness_degree == 2  # xi^3 is absent
    p2 = PolynomialSpace.total_degree(2, 2)
    assert p2.max_total_degree == p2.completeness_degree == 2


def test_p_k_is_contained_in_q_k() -> None:
    """Every total-degree monomial of degree <= k survives the per-variable cap."""
    for order in (1, 2, 3):
        assert set(PolynomialSpace.total_degree(order, 2).exponents) <= set(
            PolynomialSpace.tensor_product(order, 2).exponents
        )


def test_contains_constant_and_total_degree_queries() -> None:
    """Structural queries report what the monomial set actually holds."""
    p2 = PolynomialSpace.total_degree(2, 2)
    assert p2.contains_constant
    assert p2.contains_total_degree(2)
    assert not p2.contains_total_degree(3)
    q1 = PolynomialSpace.tensor_product(1, 2)
    assert q1.contains_total_degree(1)
    assert not q1.contains_total_degree(3)  # xi^2*eta missing (and degree 3 needs xi^3)


def test_monomial_ordering_is_graded_lexicographic() -> None:
    """Monomials are sorted by total degree then lexicographically."""
    space = PolynomialSpace.total_degree(2, 2)
    assert space.exponents == ((0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0))
    degrees = [sum(alpha) for alpha in space.exponents]
    assert degrees == sorted(degrees)


def test_monomial_labels() -> None:
    """Labels are readable and use the SDS reference-coordinate symbols."""
    assert monomial_label((0, 0)) == "1"
    assert monomial_label((1, 0)) == "xi"
    assert monomial_label((0, 1)) == "eta"
    assert monomial_label((2, 1)) == "xi^2*eta"
    assert monomial_label((0, 0, 3)) == "zeta^3"
    assert PolynomialSpace.total_degree(1, 2).monomial_labels == ("1", "eta", "xi")


def test_space_name_and_repr() -> None:
    """Spaces report their conventional name."""
    assert PolynomialSpace.total_degree(2, 2).name == "P2"
    assert PolynomialSpace.tensor_product(3, 2).name == "Q3"
    assert "Q3" in repr(PolynomialSpace.tensor_product(3, 2))


def test_space_is_frozen_hashable_and_serializable() -> None:
    """Spaces are value objects usable as cache keys."""
    a = PolynomialSpace.total_degree(2, 2)
    b = PolynomialSpace.total_degree(2, 2)
    assert a == b and hash(a) == hash(b)
    assert len({a, b}) == 1
    payload = a.to_dict()
    assert payload["name"] == "P2" and payload["dimension"] == 6
    with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError is an AttributeError
        a.order = 3  # type: ignore[misc]


def test_space_rejects_bad_input() -> None:
    """Negative orders, unsupported variable counts, and malformed sets raise."""
    with pytest.raises(InputValidationError):
        PolynomialSpace.total_degree(-1, 2)
    with pytest.raises(InputValidationError):
        PolynomialSpace.tensor_product(2, 4)
    with pytest.raises(InputValidationError):
        PolynomialSpace(PolynomialSpaceType.TOTAL_DEGREE, 1, 2, ())
    with pytest.raises(InputValidationError):
        PolynomialSpace(PolynomialSpaceType.TOTAL_DEGREE, 1, 2, ((0, 0), (0, 0)))
    with pytest.raises(InputValidationError):
        PolynomialSpace(PolynomialSpaceType.TOTAL_DEGREE, 1, 2, ((0, 0), (1,)))
    with pytest.raises(InputValidationError):
        PolynomialSpace(PolynomialSpaceType.TOTAL_DEGREE, 1, 2, ((0, 0), (-1, 0)))


def test_serendipity_dimension_formula_is_deferred() -> None:
    """S_k has no generating rule yet, and says so rather than guessing."""
    space = PolynomialSpace(PolynomialSpaceType.SERENDIPITY, 2, 2, ((0, 0), (1, 0), (0, 1)))
    with pytest.raises(NotImplementedError):
        _ = space.expected_dimension


# ---- interpolation nodes ----------------------------------------------------


def test_node_identifier_and_validation() -> None:
    """Nodes carry an entity-derived identifier and validate their inputs."""
    vertex = InterpolationNode(0, (0.0, 0.0), EntityType.VERTEX, 0)
    edge = InterpolationNode(3, (0.5, 0.5), EntityType.EDGE, 0, 1)
    interior = InterpolationNode(9, (0.25, 0.25), EntityType.CELL, 0, 0)
    assert vertex.identifier == "v0"
    assert edge.identifier == "e0.1"
    assert interior.identifier == "c0.0"
    assert vertex.dimension == 2
    with pytest.raises(InputValidationError):
        InterpolationNode(-1, (0.0,), EntityType.VERTEX, 0)
    with pytest.raises(InputValidationError):
        InterpolationNode(0, (), EntityType.VERTEX, 0)
    with pytest.raises(InputValidationError):
        InterpolationNode(0, (float("nan"),), EntityType.VERTEX, 0)
    with pytest.raises(InputValidationError):
        InterpolationNode(0, (0.0,), EntityType.VERTEX, -1)


# ---- dof functionals --------------------------------------------------------


def test_point_value_functional_evaluates_a_monomial() -> None:
    """A zero multi-index is the point value u -> u(x)."""
    dof = DofFunctional(0, 0, (2.0, 3.0), (0, 0), EntityType.VERTEX, 0)
    assert dof.kind is DofKind.POINT_VALUE
    assert dof.order == 0
    assert dof.identifier == "v0"
    assert dof.apply_to_monomial((0, 0)) == pytest.approx(1.0)  # the constant
    assert dof.apply_to_monomial((1, 0)) == pytest.approx(2.0)  # xi
    assert dof.apply_to_monomial((2, 1)) == pytest.approx(4.0 * 3.0)  # xi^2 * eta


def test_derivative_functional_differentiates_a_monomial() -> None:
    """A non-zero multi-index differentiates the monomial, not a shape function."""
    d_xi = DofFunctional(1, 0, (2.0, 3.0), (1, 0), EntityType.VERTEX, 0)
    assert d_xi.kind is DofKind.POINT_DERIVATIVE
    assert d_xi.order == 1
    assert d_xi.identifier == "v0_d10"
    assert d_xi.apply_to_monomial((0, 0)) == pytest.approx(0.0)  # d/dxi of a constant
    assert d_xi.apply_to_monomial((1, 0)) == pytest.approx(1.0)  # d/dxi of xi
    assert d_xi.apply_to_monomial((3, 1)) == pytest.approx(3.0 * 4.0 * 3.0)  # 3*xi^2*eta


def test_cross_derivative_functional() -> None:
    """The (1, 1) multi-index is the cross derivative the BFS quadrilateral needs."""
    cross = DofFunctional(3, 0, (2.0, 3.0), (1, 1), EntityType.VERTEX, 0)
    assert cross.order == 2
    assert cross.identifier == "v0_d11"
    assert cross.apply_to_monomial((1, 1)) == pytest.approx(1.0)  # d2/dxideta of xi*eta
    assert cross.apply_to_monomial((2, 0)) == pytest.approx(0.0)  # no eta to differentiate
    assert cross.apply_to_monomial((2, 2)) == pytest.approx(2 * 2.0 * 2 * 3.0)


def test_derivative_annihilates_lower_powers() -> None:
    """D^alpha x^beta vanishes whenever beta_i < alpha_i."""
    second = DofFunctional(0, 0, (5.0,), (2,), EntityType.VERTEX, 0)
    assert second.apply_to_monomial((0,)) == 0.0
    assert second.apply_to_monomial((1,)) == 0.0
    assert second.apply_to_monomial((2,)) == pytest.approx(2.0)
    assert second.apply_to_monomial((3,)) == pytest.approx(6.0 * 5.0)


def test_functional_validation() -> None:
    """Mismatched multi-indices, negative orders, and non-finite points raise."""
    with pytest.raises(InputValidationError):
        DofFunctional(0, 0, (1.0, 2.0), (1,), EntityType.VERTEX, 0)
    with pytest.raises(InputValidationError):
        DofFunctional(0, 0, (1.0,), (-1,), EntityType.VERTEX, 0)
    with pytest.raises(InputValidationError):
        DofFunctional(0, 0, (float("inf"),), (0,), EntityType.VERTEX, 0)
    with pytest.raises(InputValidationError):
        DofFunctional(-1, 0, (1.0,), (0,), EntityType.VERTEX, 0)
    dof = DofFunctional(0, 0, (1.0,), (0,), EntityType.VERTEX, 0)
    with pytest.raises(InputValidationError):
        dof.apply_to_monomial((1, 1))


def test_functional_serializes() -> None:
    """Functionals produce a JSON-compatible record."""
    payload = DofFunctional(1, 0, (0.0,), (1,), EntityType.VERTEX, 0).to_dict()
    assert payload["kind"] == "point_derivative"
    assert payload["derivative"] == [1]
    assert payload["identifier"] == "v0_d1"
