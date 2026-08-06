"""Shape functions against independently implemented closed forms.

Every reference here is written from the textbook definition - the Lagrange
product formula, the classical Hermite cubics, the barycentric triangle
polynomials, tensor products of the 1-D bases - and shares no code with the
``C = M^-T`` construction under test. Agreement between two independent routes
is the point; a check that routes both sides through the same machinery would
prove nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.numerics.interpolation import (
    HermiteInterpolation,
    LagrangeInterpolation,
    LagrangeShapeFunctions,
    shape_functions,
)
from nanofem.numerics.reference.enums import CellType


def sample_line() -> np.ndarray:
    """A spread of points along the reference line, including the vertices."""
    return np.linspace(-1.0, 1.0, 11).reshape(-1, 1)


def sample_square() -> np.ndarray:
    """A grid over the reference quadrilateral."""
    axis = np.linspace(-1.0, 1.0, 5)
    grid = np.array([[x, y] for y in axis for x in axis])
    return grid


def sample_triangle() -> np.ndarray:
    """A lattice over the reference triangle (barycentric, inside and on the edges)."""
    points = []
    steps = 5
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            points.append([i / steps, j / steps])
    return np.array(points)


def lagrange_product(nodes: np.ndarray, index: int, x: np.ndarray) -> np.ndarray:
    """The classical Lagrange product formula, written independently of the library.

    ``L_i(x) = prod_{j != i} (x - x_j) / (x_i - x_j)``.
    """
    result = np.ones_like(x)
    for j, node in enumerate(nodes):
        if j == index:
            continue
        result = result * (x - node) / (nodes[index] - node)
    return result


def hermite_cubics(x: np.ndarray) -> np.ndarray:
    """The classical cubic Hermite basis on ``[-1, 1]``, from the textbook.

    Dual to ``(u(-1), u'(-1), u(1), u'(1))``, the Euler-Bernoulli beam ordering.
    """
    return np.column_stack(
        [
            (2.0 - 3.0 * x + x**3) / 4.0,
            (1.0 - x - x**2 + x**3) / 4.0,
            (2.0 + 3.0 * x - x**3) / 4.0,
            (-1.0 - x + x**2 + x**3) / 4.0,
        ]
    )


# ---- Lagrange on the line ---------------------------------------------------


def test_line_p1_matches_the_closed_form() -> None:
    """N_0 = (1 - xi)/2, N_1 = (1 + xi)/2."""
    basis = shape_functions(LagrangeInterpolation(CellType.LINE, 1))
    xi = sample_line()
    values = basis.evaluate(xi)
    x = xi[:, 0]
    assert np.allclose(values[:, 0], (1.0 - x) / 2.0)
    assert np.allclose(values[:, 1], (1.0 + x) / 2.0)
    gradients = basis.gradient(xi)
    assert np.allclose(gradients[:, 0, 0], -0.5)
    assert np.allclose(gradients[:, 1, 0], 0.5)


def test_line_p2_matches_the_closed_form() -> None:
    """With nodes ordered (-1, +1, 0): N = xi(xi-1)/2, xi(xi+1)/2, 1 - xi^2."""
    basis = shape_functions(LagrangeInterpolation(CellType.LINE, 2))
    xi = sample_line()
    values = basis.evaluate(xi)
    x = xi[:, 0]
    assert np.allclose(values[:, 0], x * (x - 1.0) / 2.0)
    assert np.allclose(values[:, 1], x * (x + 1.0) / 2.0)
    assert np.allclose(values[:, 2], 1.0 - x**2)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_line_matches_the_lagrange_product_formula(order: int) -> None:
    """Vandermonde inversion agrees with the product formula at every order.

    Two independent constructions of the same basis: the library solves
    ``M^T C = I``, the reference multiplies out ``prod (x - x_j)/(x_i - x_j)``.
    """
    interpolation = LagrangeInterpolation(CellType.LINE, order)
    basis = shape_functions(interpolation)
    nodes = interpolation.node_locations[:, 0]
    xi = sample_line()
    values = basis.evaluate(xi)
    for index in range(interpolation.num_nodes):
        expected = lagrange_product(nodes, index, xi[:, 0])
        assert np.allclose(values[:, index], expected, atol=1e-12)


# ---- Lagrange on the triangle -----------------------------------------------


def test_triangle_p1_matches_the_barycentric_form() -> None:
    """N = (1 - xi - eta, xi, eta): the barycentric coordinates themselves."""
    basis = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 1))
    points = sample_triangle()
    values = basis.evaluate(points)
    xi, eta = points[:, 0], points[:, 1]
    assert np.allclose(values[:, 0], 1.0 - xi - eta)
    assert np.allclose(values[:, 1], xi)
    assert np.allclose(values[:, 2], eta)


def test_triangle_p2_matches_the_barycentric_form() -> None:
    """Vertices lam(2lam-1); edge nodes 4*lam_a*lam_b in reference edge order.

    Our edge 0 is (V1, V2), edge 1 is (V2, V0), edge 2 is (V0, V1), so the edge
    functions are 4*xi*eta, 4*eta*lam0, 4*lam0*xi in that order.
    """
    basis = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 2))
    points = sample_triangle()
    values = basis.evaluate(points)
    xi, eta = points[:, 0], points[:, 1]
    lam0 = 1.0 - xi - eta
    assert np.allclose(values[:, 0], lam0 * (2.0 * lam0 - 1.0))
    assert np.allclose(values[:, 1], xi * (2.0 * xi - 1.0))
    assert np.allclose(values[:, 2], eta * (2.0 * eta - 1.0))
    assert np.allclose(values[:, 3], 4.0 * xi * eta)
    assert np.allclose(values[:, 4], 4.0 * eta * lam0)
    assert np.allclose(values[:, 5], 4.0 * lam0 * xi)


def test_triangle_p1_gradients_are_the_constant_barycentric_gradients() -> None:
    """P1 gradients are constant: grad lam0 = (-1,-1), grad xi = (1,0), grad eta = (0,1)."""
    basis = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 1))
    gradients = basis.gradient(sample_triangle())
    assert np.allclose(gradients[:, 0, :], [-1.0, -1.0])
    assert np.allclose(gradients[:, 1, :], [1.0, 0.0])
    assert np.allclose(gradients[:, 2, :], [0.0, 1.0])


# ---- Lagrange on the quadrilateral ------------------------------------------


def test_quad_q1_matches_the_bilinear_closed_form() -> None:
    """N_i = (1 +- xi)(1 +- eta)/4 in reference vertex order."""
    basis = shape_functions(LagrangeInterpolation(CellType.QUADRILATERAL, 1))
    points = sample_square()
    values = basis.evaluate(points)
    xi, eta = points[:, 0], points[:, 1]
    assert np.allclose(values[:, 0], (1.0 - xi) * (1.0 - eta) / 4.0)
    assert np.allclose(values[:, 1], (1.0 + xi) * (1.0 - eta) / 4.0)
    assert np.allclose(values[:, 2], (1.0 + xi) * (1.0 + eta) / 4.0)
    assert np.allclose(values[:, 3], (1.0 - xi) * (1.0 + eta) / 4.0)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_quad_is_the_tensor_product_of_the_line_basis(order: int) -> None:
    """Every quad shape function factors as L_a(xi) * L_b(eta).

    A structural identity the construction never asserts: the library builds the
    quad basis by inverting a 2-D Vandermonde, with no notion that ``Q_k``
    separates. That it comes out separable is a real check of both.
    """
    quad = LagrangeInterpolation(CellType.QUADRILATERAL, order)
    line = LagrangeInterpolation(CellType.LINE, order)
    quad_basis = shape_functions(quad)
    line_nodes = line.node_locations[:, 0]

    points = sample_square()
    values = quad_basis.evaluate(points)
    for index, node in enumerate(quad.nodes):
        axis_x = int(np.argmin(np.abs(line_nodes - node.coordinates[0])))
        axis_y = int(np.argmin(np.abs(line_nodes - node.coordinates[1])))
        expected = lagrange_product(line_nodes, axis_x, points[:, 0]) * lagrange_product(
            line_nodes, axis_y, points[:, 1]
        )
        assert np.allclose(
            values[:, index], expected, atol=1e-10
        ), f"node {index} ({node.identifier}) does not factor"


# ---- Hermite ----------------------------------------------------------------


def test_hermite_line_matches_the_classical_cubics() -> None:
    """The four classical Hermite cubics, in the (w1, theta1, w2, theta2) order."""
    basis = shape_functions(HermiteInterpolation(CellType.LINE, 3))
    xi = sample_line()
    assert np.allclose(basis.evaluate(xi), hermite_cubics(xi[:, 0]), atol=1e-12)


def test_hermite_line_gradients_match_the_differentiated_cubics() -> None:
    """Derivatives of the classical cubics, differentiated by hand."""
    basis = shape_functions(HermiteInterpolation(CellType.LINE, 3))
    xi = sample_line()
    x = xi[:, 0]
    expected = np.column_stack(
        [
            (-3.0 + 3.0 * x**2) / 4.0,
            (-1.0 - 2.0 * x + 3.0 * x**2) / 4.0,
            (3.0 - 3.0 * x**2) / 4.0,
            (-1.0 + 2.0 * x + 3.0 * x**2) / 4.0,
        ]
    )
    assert np.allclose(basis.gradient(xi)[:, :, 0], expected, atol=1e-12)


def test_hermite_line_nodal_conditions() -> None:
    """H_val = 1 with zero slope at its node; H_slope = 0 with unit slope."""
    basis = shape_functions(HermiteInterpolation(CellType.LINE, 3))
    ends = np.array([[-1.0], [1.0]])
    values = basis.evaluate(ends)
    slopes = basis.gradient(ends)[:, :, 0]
    assert np.allclose(values, [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    assert np.allclose(slopes, [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def test_bfs_quad_is_the_tensor_product_of_the_hermite_line() -> None:
    """The Bogner-Fox-Schmit basis factors as H_a(xi) * H_b(eta).

    Including the cross-derivative functions, which is the structural reason the
    (1, 1) multi-index has to be in the DOF set at all.
    """
    quad = HermiteInterpolation(CellType.QUADRILATERAL, 3)
    basis = shape_functions(quad)
    points = sample_square()
    values = basis.evaluate(points)
    line_x = hermite_cubics(points[:, 0])
    line_y = hermite_cubics(points[:, 1])
    for index, dof in enumerate(quad.dofs):
        node = quad.nodes[dof.node_index]
        # 1-D dof index = 2 * (vertex is at +1) + derivative order on that axis
        column_x = 2 * int(node.coordinates[0] > 0.0) + dof.derivative[0]
        column_y = 2 * int(node.coordinates[1] > 0.0) + dof.derivative[1]
        expected = line_x[:, column_x] * line_y[:, column_y]
        assert np.allclose(
            values[:, index], expected, atol=1e-10
        ), f"dof {index} ({dof.identifier}) does not factor"


# ---- geometric symmetry of the bases ----------------------------------------


def test_line_basis_respects_the_reflection_symmetry() -> None:
    """Under xi -> -xi the line basis permutes: N_v0(-xi) = N_v1(xi)."""
    basis = shape_functions(LagrangeInterpolation(CellType.LINE, 2))
    xi = sample_line()
    forward = basis.evaluate(xi)
    mirrored = basis.evaluate(-xi)
    assert np.allclose(mirrored[:, 0], forward[:, 1])  # v0 <-> v1
    assert np.allclose(mirrored[:, 2], forward[:, 2])  # the midpoint is invariant


def test_hermite_slope_functions_are_odd_under_reflection() -> None:
    """Reflection maps a slope dof to minus its partner: the chain rule, visible.

    Under xi -> -xi the derivative functional picks up the Jacobian's sign, so
    the value functions swap while the slope functions swap *and* negate.
    """
    basis = shape_functions(HermiteInterpolation(CellType.LINE, 3))
    xi = sample_line()
    forward = basis.evaluate(xi)
    mirrored = basis.evaluate(-xi)
    assert np.allclose(mirrored[:, 0], forward[:, 2])  # value at v0 <-> value at v1
    assert np.allclose(mirrored[:, 1], -forward[:, 3])  # slope at v0 <-> -slope at v1


def test_triangle_basis_respects_the_vertex_swap_symmetry() -> None:
    """Swapping (xi, eta) exchanges V1 and V2, and the basis follows."""
    basis = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 1))
    points = sample_triangle()
    swapped = points[:, ::-1].copy()
    forward = basis.evaluate(points)
    reflected = basis.evaluate(swapped)
    assert np.allclose(reflected[:, 0], forward[:, 0])  # V0 is fixed
    assert np.allclose(reflected[:, 1], forward[:, 2])  # V1 <-> V2


# ---- order-1 Hessians -------------------------------------------------------


@pytest.mark.parametrize("cell", [CellType.LINE, CellType.TRIANGLE, CellType.QUADRILATERAL])
def test_first_order_hessians(cell: CellType) -> None:
    """A P1 basis has an identically zero Hessian; Q1 keeps only the cross term."""
    basis = shape_functions(LagrangeInterpolation(cell, 1))
    points = basis.verification_points()
    hessian = basis.hessian(points)
    if cell is CellType.QUADRILATERAL:
        assert np.allclose(np.diagonal(hessian, axis1=-2, axis2=-1), 0.0)
        assert not np.allclose(hessian[:, :, 0, 1], 0.0)  # the xi*eta term survives
    else:
        assert np.allclose(hessian, 0.0)


def test_lagrange_shape_functions_rejects_a_hermite_interpolation() -> None:
    """Each concrete family binds only its own interpolation family."""
    from nanofem.utils.exceptions import InputValidationError

    with pytest.raises(InputValidationError, match="lagrange"):
        LagrangeShapeFunctions(HermiteInterpolation(CellType.LINE, 3))
