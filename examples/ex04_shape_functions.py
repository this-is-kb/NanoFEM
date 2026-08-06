"""Phase-4 success criterion, executable: ReferenceTriangle -> Interpolation ->
ShapeFunctionFamily -> evaluate() -> gradient() -> verify(), with no assembly.

Reference coordinates only: no Jacobian, no mapping, no quadrature, no
integration, no element matrices.
"""

from __future__ import annotations

import numpy as np

from nanofem.numerics.interpolation import (
    HermiteInterpolation,
    LagrangeInterpolation,
    available_interpolations,
    interpolation,
    shape_functions,
)
from nanofem.numerics.reference.enums import CellType


def main() -> None:
    """Walk the success-criterion chain, then survey the library."""
    triple = LagrangeInterpolation(CellType.TRIANGLE, 2)
    basis = shape_functions(triple)

    print("=== the success-criterion chain ===")
    print(basis.pretty())

    print("\nthe construction, C = M^-T:")
    matrix = triple.unisolvence_matrix()
    print(
        f"  M (generalized Vandermonde) : {matrix.shape}, cond = "
        f"{triple.unisolvence_condition_number():.4g}"
    )
    print(f"  C = M^-T                    : {basis.coefficients.shape}")
    print(
        f"  C M^T == I                  : "
        f"{np.allclose(basis.coefficients @ matrix.T, np.eye(6))}"
    )

    centroid = [[1 / 3, 1 / 3]]
    print("\nevaluate() at the centroid:")
    for identifier, value in zip(
        basis.shape_function_ids, basis.evaluate(centroid)[0], strict=True
    ):
        print(f"  {identifier:<8} = {value:+.6f}")
    print(f"  sum      = {basis.evaluate(centroid)[0].sum():+.6f}  (partition of unity)")

    print("\ngradient() at the centroid (reference coordinates):")
    for identifier, row in zip(basis.shape_function_ids, basis.gradient(centroid)[0], strict=True):
        print(f"  grad {identifier:<8} = [{row[0]:+.6f}, {row[1]:+.6f}]")
    print(f"  sum      = {basis.gradient(centroid)[0].sum(axis=0).round(12).tolist()}")

    basis.verify()
    print(f"\nverify(): {basis.is_valid()}")
    print("  duality l_k(N_i) = delta_ki, partition of unity, polynomial reproduction,")
    print("  derivative consistency (finite differences), Hessian symmetry,")
    print("  boundary restriction, interpolation exactness")

    print("\n=== the library ===")
    for family, cell, order in available_interpolations():
        item = shape_functions(interpolation(family, cell, order))
        item.verify()
        print(
            f"  {family.value:<9} {cell.value:<14} k={order}  "
            f"functions={item.num_functions:>2}  {item.continuity().name}  "
            f"cell={item.cell().name:<7} verified"
        )

    print("\n=== the Hermite line is the classical beam basis ===")
    hermite = shape_functions(HermiteInterpolation(CellType.LINE, 3))
    x = np.array([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
    classical = np.column_stack(
        [
            (2 - 3 * x[:, 0] + x[:, 0] ** 3) / 4,
            (1 - x[:, 0] - x[:, 0] ** 2 + x[:, 0] ** 3) / 4,
            (2 + 3 * x[:, 0] - x[:, 0] ** 3) / 4,
            (-1 - x[:, 0] + x[:, 0] ** 2 + x[:, 0] ** 3) / 4,
        ]
    )
    deviation = float(np.abs(hermite.evaluate(x) - classical).max())
    print(f"  H(xi) at xi=0 : {hermite.evaluate([[0.0]])[0].round(6).tolist()}")
    print(f"  max deviation from the textbook cubics: {deviation:.3e}")
    print("  nodal pattern (value=1/slope=0 at own node):")
    print(f"    N at xi=-1 : {hermite.evaluate([[-1.0]])[0].round(9).tolist()}")
    print(f"    dN at xi=-1: {hermite.gradient([[-1.0]])[0, :, 0].round(9).tolist()}")

    print("\n=== boundary restriction: why conformity works ===")
    element = triple.reference_element
    start, end = element.vertex_coordinates[1], element.vertex_coordinates[2]
    t = np.linspace(0.0, 1.0, 5).reshape(-1, 1)
    on_edge0 = basis.evaluate(start + t * (end - start))
    print(
        f"  on edge 0 (V1->V2), N_v0 (a node off this edge) = "
        f"{on_edge0[:, 0].round(12).tolist()}"
    )
    print("  so the trace depends only on dofs the neighbour shares")

    print("\n=== caching ===")
    basis.clear_cache()
    points = np.array([[0.25, 0.25], [0.5, 0.25]])
    first = basis.evaluate(points)
    second = basis.evaluate(points)
    print(
        f"  same points tabulated once: {first is second}, "
        f"cache entries = {basis.cache_info()['entries']}"
    )
    basis.gradient(points)
    print(
        f"  after gradient(): {basis.cache_info()['entries']} entries " f"(values + one per axis)"
    )

    print("\nno Jacobian, no mapping, no quadrature, no assembly - phase 4 complete")


if __name__ == "__main__":
    main()
