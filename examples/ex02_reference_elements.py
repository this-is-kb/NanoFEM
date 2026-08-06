"""Phase-2 success criterion, executable: ReferenceTriangle() -> query vertices ->
query edges -> query connectivity -> validate topology -> serialize.

Uses no NanoFEM module other than the reference library itself, demonstrating
that the geometric foundation stands alone. Purely geometric and topological:
no shape functions, no quadrature, no Jacobians, no mapping.
"""

from __future__ import annotations

import json

from nanofem.numerics.reference import (
    CellType,
    Orientation,
    ReferenceLine,
    ReferenceQuadrilateral,
    ReferenceTriangle,
    reference_element,
    reference_element_for_name,
    reference_element_from_dict,
)


def main() -> None:
    """Walk the success-criterion chain, then survey the whole library."""
    triangle = ReferenceTriangle()

    print("=== the success-criterion chain ===")
    print(triangle.pretty())

    print("\nvertices:")
    for i, coordinates in enumerate(triangle.vertex_coordinates):
        print(f"  V{i} = {coordinates.tolist()}")

    print("\nedges:")
    for i, (a, b) in enumerate(triangle.edge_vertex_indices):
        length = triangle.edge_lengths()[i]
        print(f"  E{i} = (V{a}, V{b})  length={length:.6f}")

    print("\nconnectivity (facet i is opposite vertex i):")
    normals = triangle.reference_normals()
    for i, facet in enumerate(triangle.facet_vertex_indices):
        print(f"  F{i} = {facet}  outward normal={normals[i].round(6).tolist()}")

    triangle.validate()
    print(f"\ntopology validated: {triangle.is_valid()}")

    payload = triangle.to_dict()
    restored = reference_element_from_dict(payload)
    print(f"serialized to {len(triangle.to_json())} bytes of JSON")
    print(f"round trip reconstructs an equal element: {restored == triangle}")
    print(f"json keys: {sorted(json.loads(triangle.to_json()))[:5]} ...")

    print("\n=== the implemented library ===")
    for element in (ReferenceLine(), ReferenceTriangle(), ReferenceQuadrilateral()):
        element.validate()
        print(
            f"  {element!r}"
            f"\n      measure={element.reference_measure:g} "
            f"centroid={element.centroid.round(4).tolist()} "
            f"diameter={element.diameter:.6f}"
        )

    print("\n=== orientation of a shared facet ===")
    print(f"  canonical : F0 = {triangle.permute_facet(0, Orientation.FORWARD)}")
    print(f"  neighbour : F0 = {triangle.permute_facet(0, Orientation.REVERSED)}")

    print("\n=== point queries (pure geometry) ===")
    for point in ((0.25, 0.25), (1.0 / 3.0, 1.0 / 3.0), (0.6, 0.6)):
        inside = triangle.contains(point)
        distance = triangle.signed_distance_to_boundary(point)
        where = "inside" if inside else "outside"
        print(f"  {point}: {where:<7} signed distance to boundary = {distance:+.6f}")

    print("\n=== registry ===")
    print(f"  by enum   : {reference_element(CellType.QUADRILATERAL)!r}")
    print(f"  by string : {reference_element('line')!r}")
    print(f"  by mesh name 'tri6' -> {reference_element_for_name('tri6')!r} (order stripped)")
    print(f"  equal shapes collapse in a set: {len({ReferenceTriangle(), ReferenceTriangle()})}")

    print("\nno mapping, no shape functions, no quadrature - phase 2 complete")


if __name__ == "__main__":
    main()
