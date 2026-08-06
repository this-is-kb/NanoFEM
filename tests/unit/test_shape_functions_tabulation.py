"""Tabulation, caching, the factory, value semantics, and frozen regressions.

The regression tests pin coefficient matrices and nodal values so a refactor
cannot silently change the basis every future element will be built on.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from nanofem.numerics.interpolation import (
    HermiteInterpolation,
    HermiteShapeFunctions,
    LagrangeInterpolation,
    LagrangeShapeFunctions,
    ShapeFunctionFamily,
    ShapeFunctions,
    Tabulation,
    available_interpolations,
    interpolation,
    monomial_table,
    shape_functions,
)
from nanofem.numerics.interpolation.polynomial import PolynomialSpace
from nanofem.numerics.interpolation.tabulation import as_points
from nanofem.numerics.reference.cell import REFERENCE_CELLS
from nanofem.numerics.reference.enums import CellType
from nanofem.utils.exceptions import InputValidationError

ALL: list[ShapeFunctionFamily] = [
    shape_functions(interpolation(f, c, k)) for f, c, k in available_interpolations()
]
IDS = [f"{b.family.value}-{b.interpolation.cell_type.value}-{b.order}" for b in ALL]


# ---- the monomial tabulator --------------------------------------------------


def test_monomial_table_evaluates_and_differentiates() -> None:
    """Values, first derivatives, and the cross derivative of the P2/Q1 monomials."""
    space = PolynomialSpace.tensor_product(1, 2)  # 1, eta, xi, xi*eta
    points = np.array([[2.0, 3.0]])
    assert np.allclose(monomial_table(space, points), [[1.0, 3.0, 2.0, 6.0]])
    assert np.allclose(monomial_table(space, points, (1, 0)), [[0.0, 0.0, 1.0, 3.0]])
    assert np.allclose(monomial_table(space, points, (0, 1)), [[0.0, 1.0, 0.0, 2.0]])
    assert np.allclose(monomial_table(space, points, (1, 1)), [[0.0, 0.0, 0.0, 1.0]])


def test_monomial_table_annihilates_low_powers_without_dividing_by_zero() -> None:
    """D^2 of a linear monomial is 0, and the clamped power never sees 0 ** -1."""
    space = PolynomialSpace.total_degree(1, 1)  # 1, xi
    at_origin = monomial_table(space, np.array([[0.0]]), (2,))
    assert np.allclose(at_origin, [[0.0, 0.0]])
    assert np.all(np.isfinite(at_origin))


def test_monomial_table_agrees_with_the_phase_three_functionals() -> None:
    """The vectorized tabulator reproduces the unisolvence matrix exactly.

    Phase 3 fills M one functional at a time with a scalar routine; this layer
    tabulates whole batches. They must be the same arithmetic, and this pins it
    across the phase boundary.
    """
    for family, cell, order in available_interpolations():
        interp = interpolation(family, cell, order)
        expected = interp.unisolvence_matrix()
        rows = [
            monomial_table(
                interp.polynomial_space,
                np.asarray([dof.point], dtype=np.float64),
                dof.derivative,
            )[0]
            for dof in interp.dofs
        ]
        assert np.allclose(
            np.array(rows), expected, atol=1e-14
        ), f"{family.value} {cell.value} k={order}"


def test_monomial_table_validates_its_multi_index() -> None:
    """A multi-index of the wrong length or with a negative order raises."""
    space = PolynomialSpace.total_degree(1, 2)
    points = np.array([[0.0, 0.0]])
    with pytest.raises(InputValidationError, match="entries"):
        monomial_table(space, points, (1,))
    with pytest.raises(InputValidationError, match="negative"):
        monomial_table(space, points, (-1, 0))


def test_as_points_coerces_validates_and_promotes() -> None:
    """A flat point becomes one row; bad shapes and non-finite values raise."""
    assert as_points([0.5, 0.25], 2).shape == (1, 2)
    assert as_points(np.zeros((4, 2)), 2).shape == (4, 2)
    with pytest.raises(InputValidationError, match="shape"):
        as_points(np.zeros((4, 3)), 2)
    with pytest.raises(InputValidationError, match="non-finite"):
        as_points([[np.nan, 0.0]], 2)
    with pytest.raises(InputValidationError, match="at least one"):
        as_points(np.zeros((0, 2)), 2)


# ---- tabulate ----------------------------------------------------------------


@pytest.mark.parametrize("basis", ALL, ids=IDS)
def test_tabulate_shapes(basis: ShapeFunctionFamily) -> None:
    """A batch carries values, gradients, and Hessians with the documented shapes."""
    points = basis.verification_points()
    dimension = basis.interpolation.support_dimension
    batch = basis.tabulate(points, max_derivative=2)
    assert isinstance(batch, Tabulation)
    assert batch.values.shape == (len(points), basis.num_functions)
    assert batch.gradients is not None
    assert batch.gradients.shape == (len(points), basis.num_functions, dimension)
    assert batch.hessians is not None
    assert batch.hessians.shape == (len(points), basis.num_functions, dimension, dimension)
    assert batch.num_points == len(points)
    assert batch.num_functions == basis.num_functions
    assert batch.dimension == dimension
    assert batch.max_derivative == 2


def test_tabulate_respects_max_derivative() -> None:
    """Requesting fewer derivatives omits them rather than computing them."""
    basis = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 2))
    points = [[0.25, 0.25]]
    values_only = basis.tabulate(points, max_derivative=0)
    assert values_only.gradients is None and values_only.hessians is None
    assert values_only.max_derivative == 0
    with_gradient = basis.tabulate(points, max_derivative=1)
    assert with_gradient.gradients is not None and with_gradient.hessians is None
    assert with_gradient.max_derivative == 1
    with pytest.raises(InputValidationError, match="0, 1, or 2"):
        basis.tabulate(points, max_derivative=3)


def test_tabulation_repr_is_informative() -> None:
    """A batch reports its size at a glance."""
    basis = shape_functions(LagrangeInterpolation(CellType.LINE, 1))
    text = repr(basis.tabulate([[0.0]], max_derivative=2))
    assert "points=1" in text and "functions=2" in text and "max_derivative=2" in text


# ---- caching -----------------------------------------------------------------


def test_repeated_tabulation_hits_the_cache() -> None:
    """The same points are tabulated once; SDS C-8's sharing pattern."""
    basis = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 2))
    basis.clear_cache()
    assert basis.cache_info()["entries"] == 0
    points = np.array([[0.25, 0.25], [0.5, 0.25]])
    first = basis.evaluate(points)
    assert basis.cache_info()["entries"] == 1
    second = basis.evaluate(points)
    assert basis.cache_info()["entries"] == 1
    assert first is second  # the identical cached array, not a recomputation
    basis.evaluate(points + 0.1)
    assert basis.cache_info()["entries"] == 2
    basis.clear_cache()
    assert basis.cache_info()["entries"] == 0


def test_cache_distinguishes_derivative_orders() -> None:
    """Values and each derivative are cached separately at the same points."""
    basis = shape_functions(LagrangeInterpolation(CellType.LINE, 2))
    basis.clear_cache()
    points = np.array([[0.3]])
    basis.evaluate(points)
    basis.derivative(points, (1,))
    basis.derivative(points, (2,))
    assert basis.cache_info()["entries"] == 3


def test_cached_arrays_are_read_only() -> None:
    """A shared cached batch cannot be corrupted by one of its consumers."""
    basis = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 1))
    points = np.array([[0.25, 0.25]])
    for array in (basis.evaluate(points), basis.gradient(points), basis.hessian(points)):
        with pytest.raises(ValueError):
            array[0, 0] = 99.0
    with pytest.raises(ValueError):
        np.asarray(basis.coefficients)[0, 0] = 99.0


def test_flat_point_input_is_accepted() -> None:
    """A single point may be given flat; it is promoted to one row."""
    basis = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 1))
    assert basis.evaluate([1 / 3, 1 / 3]).shape == (1, 3)
    with pytest.raises(InputValidationError):
        basis.evaluate([0.5])  # wrong dimension for a triangle


# ---- the phase-0 SDS 2.4 contract --------------------------------------------


@pytest.mark.parametrize("basis", ALL, ids=IDS)
def test_the_sds_shape_function_contract_is_satisfied(basis: ShapeFunctionFamily) -> None:
    """The phase-0 ShapeFunctions seam is filled, not bypassed."""
    assert isinstance(basis, ShapeFunctions)
    assert basis.cell().name in REFERENCE_CELLS
    assert basis.cell().num_nodes == basis.interpolation.num_nodes
    assert basis.continuity() is basis.interpolation.continuity
    assert basis.completeness_degree() == basis.interpolation.order
    points = basis.verification_points()
    assert np.allclose(basis.derivatives(points), basis.gradient(points))


def test_the_cell_records_the_shape_functions_need_are_registered() -> None:
    """Every implemented element's mesh cell now resolves (closing dev note N-21).

    Phase 3 recorded that quad9, line4, tri10, and quad16 were missing from the
    registry, to be added by the family that consumes them. This is that family.
    """
    for name in ("line4", "tri10", "quad9", "quad16"):
        assert name in REFERENCE_CELLS
    for basis in ALL:
        assert basis.cell().name == basis.interpolation.mesh_cell_name


# ---- the factory and value semantics -----------------------------------------


def test_factory_dispatches_by_family() -> None:
    """shape_functions() returns the family-specific class."""
    assert isinstance(
        shape_functions(LagrangeInterpolation(CellType.LINE, 1)), LagrangeShapeFunctions
    )
    assert isinstance(
        shape_functions(HermiteInterpolation(CellType.LINE, 3)), HermiteShapeFunctions
    )


def test_hermite_shape_functions_rejects_a_lagrange_interpolation() -> None:
    """Each concrete family binds only its own interpolation family."""
    with pytest.raises(InputValidationError, match="hermite"):
        HermiteShapeFunctions(LagrangeInterpolation(CellType.LINE, 3))


def test_equality_and_hashing_follow_the_interpolation() -> None:
    """Two bases over the same triple are equal and share a hash."""
    a = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 2))
    b = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 2))
    c = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 1))
    assert a == b and hash(a) == hash(b)
    assert a != c and a != object()
    assert len({a, b, c}) == 2
    cache = {basis: basis.num_functions for basis in ALL}
    assert cache[shape_functions(HermiteInterpolation(CellType.LINE, 3))] == 4


@pytest.mark.parametrize("basis", ALL, ids=IDS)
def test_repr_and_pretty(basis: ShapeFunctionFamily) -> None:
    """Summaries report the basis, its space, and the construction."""
    assert type(basis).__name__ in repr(basis)
    assert f"functions={basis.num_functions}" in repr(basis)
    text = basis.pretty()
    assert "M^-T" in text
    assert basis.interpolation.polynomial_space.name in text


def test_families_are_frozen() -> None:
    """Shape function families are frozen dataclasses."""
    basis = LagrangeShapeFunctions(LagrangeInterpolation(CellType.LINE, 1))
    with pytest.raises(FrozenInstanceError):
        # mypy is right that this is illegal; that is exactly what is under test.
        basis.interpolation_ = LagrangeInterpolation(CellType.LINE, 2)  # type: ignore[misc]


# ---- regression: the frozen bases --------------------------------------------


def test_regression_line_p1_coefficients() -> None:
    """C = M^-T for the linear line: N_0 = (1 - xi)/2, N_1 = (1 + xi)/2."""
    basis = shape_functions(LagrangeInterpolation(CellType.LINE, 1))
    assert np.allclose(basis.coefficients, [[0.5, -0.5], [0.5, 0.5]])


def test_regression_hermite_line_coefficients_and_midpoint() -> None:
    """The classical cubic Hermite coefficients, and its values at xi = 0."""
    basis = shape_functions(HermiteInterpolation(CellType.LINE, 3))
    assert np.allclose(
        basis.coefficients,
        [
            [0.50, -0.75, 0.0, 0.25],  # H1 = (2 - 3xi + xi^3)/4
            [0.25, -0.25, -0.25, 0.25],  # H2 = (1 - xi - xi^2 + xi^3)/4
            [0.50, 0.75, 0.0, -0.25],  # H3 = (2 + 3xi - xi^3)/4
            [-0.25, -0.25, 0.25, 0.25],  # H4 = (-1 - xi + xi^2 + xi^3)/4
        ],
        atol=1e-12,
    )
    assert np.allclose(basis.evaluate([[0.0]]), [[0.5, 0.25, 0.5, -0.25]], atol=1e-12)


def test_regression_nodal_values_are_the_identity() -> None:
    """Every Lagrange basis evaluates to the identity at its own nodes."""
    for family, cell, order in available_interpolations():
        interp = interpolation(family, cell, order)
        if not interp.is_nodal:
            continue
        basis = shape_functions(interp)
        values = basis.evaluate(interp.node_locations)
        assert np.allclose(values, np.eye(interp.num_nodes), atol=1e-10), f"{cell.value} k={order}"


def test_regression_centroid_values() -> None:
    """Values at the centroid pin the low-order bases."""
    tri = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 1))
    assert np.allclose(tri.evaluate([[1 / 3, 1 / 3]]), [[1 / 3, 1 / 3, 1 / 3]])
    quad = shape_functions(LagrangeInterpolation(CellType.QUADRILATERAL, 1))
    assert np.allclose(quad.evaluate([[0.0, 0.0]]), [[0.25, 0.25, 0.25, 0.25]])
    tri2 = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 2))
    assert np.allclose(
        tri2.evaluate([[1 / 3, 1 / 3]]),
        [[-1 / 9, -1 / 9, -1 / 9, 4 / 9, 4 / 9, 4 / 9]],
        atol=1e-12,
    )


# ---- independence ------------------------------------------------------------


def test_module_needs_no_quadrature_or_mapping() -> None:
    """The success criterion: the basis stands on interpolation and geometry alone."""
    import subprocess
    import sys

    script = (
        "import sys;"
        "from nanofem.numerics.interpolation import LagrangeInterpolation, shape_functions;"
        "from nanofem.numerics.reference.enums import CellType;"
        "b = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 2));"
        "b.evaluate([[0.25, 0.25]]);"
        "b.gradient([[0.25, 0.25]]);"
        "b.verify();"
        "loaded = set(sys.modules);"
        "assert not [m for m in loaded if 'quadrature' in m], 'quadrature was imported';"
        "assert not [m for m in loaded if 'mapping' in m], 'mapping was imported';"
        "print('standalone OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "standalone OK" in result.stdout
