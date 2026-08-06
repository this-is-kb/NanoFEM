"""Phase-6 success criterion, executable: ReferenceTriangle -> DunavantQuadrature(5)
-> points -> weights -> verify exactness -> integrate scalar functions, with no
finite element constructed.

Integration over reference domains only: no Jacobian, no mapping, no element
matrix, no assembly.
"""

from __future__ import annotations

import math

import numpy as np

from nanofem.numerics.quadrature import (
    DunavantQuadrature,
    GaussLegendreQuadrature,
    available_quadratures,
    exact_monomial_integral,
    quadrature,
)
from nanofem.numerics.reference import ReferenceTriangle
from nanofem.numerics.reference.enums import CellType


def main() -> None:
    """Walk the success-criterion chain, then survey the library."""
    triangle = ReferenceTriangle()
    rule = DunavantQuadrature(order=5)

    print("=== the success-criterion chain ===")
    print(rule.pretty())

    print("\npoints and weights:")
    for index, (point, weight) in enumerate(zip(rule.points, rule.weights, strict=True)):
        print(f"  [{index}] ({point[0]:+.9f}, {point[1]:+.9f})  w = {weight:+.9f}")

    print("\nverify exactness against the closed forms:")
    for exponents in [(0, 0), (1, 0), (2, 1), (3, 2), (4, 1)]:
        numerical = rule.moment(exponents)
        exact = exact_monomial_integral(triangle, exponents)
        mark = "ok" if math.isclose(numerical, exact, abs_tol=1e-13) else "FAIL"
        print(f"  x^{exponents[0]} y^{exponents[1]}: {numerical:+.12f} vs {exact:+.12f}  [{mark}]")

    rule.verify()
    print(f"\nverify(): {rule.is_valid()}")
    print("  weight normalization, positivity, points in domain, exactness,")
    print("  maximality, moment identities, symmetry, barycentric consistency")

    print("\nintegrate scalar functions over the triangle:")
    linear = rule.integrate(lambda p: 1.0 - p[:, 0] - p[:, 1])
    print(f"  integral (1 - x - y) = {linear:.12f}  (exact 1/6 = {1/6:.12f})")
    quadratic = rule.integrate(lambda p: p[:, 0] ** 2 + p[:, 1] ** 2)
    print(f"  integral (x^2 + y^2) = {quadratic:.12f}  (exact 1/6 = {1/6:.12f})")
    gaussian = rule.integrate(lambda p: np.exp(-(p[:, 0] ** 2 + p[:, 1] ** 2)))
    print(f"  integral exp(-(x^2+y^2)) = {gaussian:.12f}  (not a polynomial: approximate)")

    print("\nmeasure and centroid by quadrature:")
    print(f"  area     = {rule.measure():.12f}  (exact 1/2)")
    print(f"  centroid = {rule.centroid().round(12).tolist()}  (exact [1/3, 1/3])")

    print("\n=== the library of families ===")
    for family, cell in available_quadratures():
        item = quadrature(cell, 5, family)
        item.verify()
        print(
            f"  {family.value:<15} on {cell.value:<14} "
            f"n={item.num_points:2d} exact={item.exactness_degree} "
            f"positive={item.has_positive_weights}"
        )

    print("\n=== Gauss-Legendre reaches the 2n-1 ceiling ===")
    for order in (1, 3, 5, 7):
        g = GaussLegendreQuadrature(order)
        print(
            f"  order {order}: {g.num_points} points, exact to degree {g.exactness_degree} "
            f"= 2*{g.num_points}-1"
        )
    exact_e = math.e - 1.0 / math.e
    print("  integral exp(x) over [-1,1] converging to e - 1/e:")
    for order in (1, 5, 9, 13):
        value = GaussLegendreQuadrature(order).integrate(lambda p: np.exp(p[:, 0]))
        print(
            f"    {GaussLegendreQuadrature(order).num_points} points: error "
            f"{abs(float(value) - exact_e):.3e}"
        )

    print("\n=== the degree-3 Dunavant rule has a negative weight ===")
    d3 = DunavantQuadrature(3)
    print(
        f"  {d3.num_points} points, weights/area = "
        f"{(d3.weights / d3.total_weight).round(6).tolist()}"
    )
    print(
        f"  has_positive_weights = {d3.has_positive_weights}  "
        f"(reported, not hidden; degree 4 is the positive alternative)"
    )
    print(f"  it still verifies: {d3.is_valid()}")

    print("\n=== symmetry orbits are recovered from the points ===")
    for degree in (1, 2, 3, 4, 5):
        orbits = DunavantQuadrature(degree).symmetry_orbits
        print(f"  degree {degree}: orbits {sorted(len(o) for o in orbits)}")
    grid = quadrature(CellType.QUADRILATERAL, 4)
    print(
        f"  3x3 Gauss grid on the square: orbits "
        f"{sorted(len(o) for o in grid.symmetry_orbits)}  (corners, edges, centre)"
    )

    print("\nno mapping, no Jacobian, no assembly - phase 6 complete")


if __name__ == "__main__":
    main()
