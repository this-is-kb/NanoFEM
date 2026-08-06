"""Phase-3 success criterion, executable: ReferenceTriangle -> LagrangeInterpolation(order=2)
-> query interpolation nodes -> query polynomial degree -> validate metadata -> serialize,
without evaluating a single shape function.

Purely the interpolation framework: polynomial spaces as monomial sets, degrees
of freedom as functionals, and the verification suite that proves the (as yet
unbuilt) nodal basis exists.
"""

from __future__ import annotations

from nanofem.numerics.interpolation import (
    HermiteInterpolation,
    LagrangeInterpolation,
    SpectralInterpolation,
    available_interpolations,
    interpolation,
    interpolation_from_dict,
)
from nanofem.numerics.reference.enums import CellType


def main() -> None:
    """Walk the success-criterion chain, then survey the framework."""
    element = LagrangeInterpolation(CellType.TRIANGLE, 2)

    print("=== the success-criterion chain ===")
    print(element.pretty())

    print("\ninterpolation nodes:")
    for node in element.interpolation_nodes():
        coordinates = ", ".join(f"{x:+.4f}" for x in node.coordinates)
        print(
            f"  [{node.index}] {node.identifier:<6} ({coordinates})  "
            f"on {node.entity.value} {node.entity_index}"
        )

    space = element.polynomial_space
    print("\npolynomial degree:")
    print(f"  space               : {space.name}, spanned by {', '.join(element.basis_ids)}")
    print(f"  nominal order       : {element.order}")
    print(f"  completeness degree : {element.completeness_degree}  (governs the rate)")
    print(f"  max total degree    : {element.max_total_degree}")
    print(f"  space dimension     : {element.space_dimension} == {element.num_dofs} dofs")

    element.validate()
    print(f"\nmetadata validated: {element.is_valid()}")
    print(
        f"  unisolvence matrix : {element.unisolvence_matrix().shape} "
        f"cond={element.unisolvence_condition_number():.4g}  (built, never inverted)"
    )

    payload = element.to_dict()
    restored = interpolation_from_dict(payload)
    print(f"\nserialized to {len(element.to_json())} bytes of JSON")
    print(f"round trip reconstructs an equal element: {restored == element}")

    print("\n=== the implemented framework ===")
    for family, cell, order in available_interpolations():
        item = interpolation(family, cell, order)
        item.validate()
        print(
            f"  {family.value:<9} {cell.value:<14} k={order}  "
            f"{item.polynomial_space.name:<3} nodes={item.num_nodes:>2} dofs={item.num_dofs:>2} "
            f"{item.continuity.name}  {item.mesh_cell_name}"
        )

    print("\n=== Lagrange vs Hermite: where the extra order goes ===")
    lagrange = LagrangeInterpolation(CellType.LINE, 3)
    hermite = HermiteInterpolation(CellType.LINE, 3)
    print(
        f"  Lagrange P3 line : {lagrange.num_nodes} nodes, {lagrange.num_dofs} dofs "
        f"-> mesh cell {lagrange.mesh_cell_name} (order costs nodes)"
    )
    print(
        f"  Hermite  P3 line : {hermite.num_nodes} nodes, {hermite.num_dofs} dofs "
        f"-> mesh cell {hermite.mesh_cell_name} (order costs dofs at the same nodes)"
    )
    print(f"  Hermite dof order: {hermite.shape_function_ids}  = (w1, theta1, w2, theta2)")

    print("\n=== constant reproduction, from the functionals alone ===")
    constant = (0,) * hermite.support_dimension
    applied = [dof.apply_to_monomial(constant) for dof in hermite.dofs]
    print(f"  l_k(1) over the Hermite line dofs = {applied}")
    print("  value duals sum to 1; derivative functionals annihilate a constant")

    print("\n=== why the spectral family waits for quadrature ===")
    conditions = [
        LagrangeInterpolation(CellType.LINE, order).unisolvence_condition_number()
        for order in (1, 2, 3)
    ]
    print(
        f"  cond(M) on equispaced line nodes, k=1..3: "
        f"{', '.join(f'{c:.4g}' for c in conditions)}"
    )
    try:
        SpectralInterpolation(CellType.LINE, 3)
    except NotImplementedError as exc:
        print(f"  {exc}")

    print("\nno shape function was formed, evaluated, or differentiated - phase 3 complete")


if __name__ == "__main__":
    main()
