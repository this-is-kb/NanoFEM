"""Phase-5 success criterion, executable: ReferenceTriangle -> physical nodes ->
AffineMapping -> map points -> recover reference coordinates -> Jacobian ->
transform gradients -> validate, with no numerical integration.

Purely geometric: no quadrature, no weak form, no element matrix, no assembly.
"""

from __future__ import annotations

import numpy as np

from nanofem.numerics.interpolation import LagrangeInterpolation, shape_functions
from nanofem.numerics.mapping import (
    AffineMapping,
    IdentityMapping,
    IsoparametricMapping,
    NonAffineError,
)
from nanofem.numerics.reference.enums import CellType


def main() -> None:
    """Walk the success-criterion chain, then survey the layer."""
    nodes = [[1.0, 1.0], [3.0, 1.0], [1.0, 4.0]]
    mapping = AffineMapping(CellType.TRIANGLE, nodes)

    print("=== the success-criterion chain ===")
    print(mapping.pretty())

    print("\nreference -> physical:")
    print(f"  A = {mapping.linear.round(6).tolist()}   b = {mapping.translation.round(6).tolist()}")
    for point in ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1 / 3, 1 / 3]):
        image = mapping.map([point])[0]
        print(f"  xi={np.array(point).round(4).tolist()} -> x={image.round(6).tolist()}")

    print("\nphysical -> reference (closed form, exact for an affine map):")
    physical = np.array([[2.0, 2.0], [1.5, 3.0]])
    recovered = mapping.inverse_map(physical)
    for target, back in zip(physical, recovered, strict=True):
        print(f"  x={target.tolist()} -> xi={back.round(9).tolist()}")
    probe = [[0.2, 0.3]]
    exact = np.allclose(mapping.inverse_map(mapping.map(probe)), probe)
    print(f"  round trip exact: {exact}")

    centre = [[1 / 3, 1 / 3]]
    print("\nJacobian and its family at the centroid:")
    print(f"  J          = {mapping.jacobian(centre)[0].round(6).tolist()}")
    print(f"  det J      = {mapping.jacobian_determinant(centre)[0]:.6f}")
    print(f"  J^-1       = {mapping.inverse_jacobian(centre)[0].round(6).tolist()}")
    print(f"  G = J^T J  = {mapping.metric_tensor(centre)[0].round(6).tolist()}")
    print(f"  g_a (covariant)     = {mapping.covariant_basis(centre)[0].round(6).tolist()}")
    print(f"  g^a (contravariant) = {mapping.contravariant_basis(centre)[0].round(6).tolist()}")
    print(f"  area = reference measure x volume scale = {mapping.physical_measure():.6f}")

    print("\ngradient transformation:")
    basis = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 1))
    reference_gradient = basis.gradient(centre)
    physical_gradient = mapping.physical_gradient(reference_gradient, centre)
    for identifier, row in zip(basis.shape_function_ids, physical_gradient[0], strict=True):
        print(f"  grad_x {identifier:<5} = {row.round(6).tolist()}")
    pulled_back = mapping.reference_gradient(physical_gradient, centre)
    print(
        f"  pull-back recovers the reference gradient: "
        f"{np.allclose(pulled_back, reference_gradient)}"
    )
    identity = np.einsum("ni,nj->ij", mapping.physical_vertices, physical_gradient[0])
    print(f"  sum_a x_a (x) grad_x N_a = {identity.round(9).tolist()}  (must be I)")

    mapping.validate()
    mapping.verify()
    print(f"\nvalidate() and verify(): {mapping.is_valid()}")

    print("\n=== affineness is a property of the geometry ===")
    trapezoid = [[0.0, 0.0], [2.0, 0.0], [3.0, 2.0], [0.0, 1.0]]
    parallelogram = [[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [1.0, 1.0]]
    for name, corners in (("trapezoid", trapezoid), ("parallelogram", parallelogram)):
        iso = IsoparametricMapping(LagrangeInterpolation(CellType.QUADRILATERAL, 1), corners)
        iso.validate()
        iso.verify()
        print(f"  {name:<14} is_affine={iso.is_affine}  quality={iso.quality:.4f}")
    try:
        AffineMapping(CellType.QUADRILATERAL, trapezoid)
    except NonAffineError as exc:
        print(f"  AffineMapping refuses the trapezoid: {str(exc)[:58]}...")

    print("\n=== the mapping Hessian correction is not optional ===")
    iso = IsoparametricMapping(LagrangeInterpolation(CellType.QUADRILATERAL, 1), trapezoid)
    quad_basis = iso.geometry_basis
    at = np.array([[0.2, -0.1]])
    correct = iso.physical_hessian(quad_basis.gradient(at), quad_basis.hessian(at), at)[0][0]
    inverse = iso.inverse_jacobian(at)[0]
    naive = np.einsum("ai,ab,bj->ij", inverse, quad_basis.hessian(at)[0][0], inverse)
    print(f"  corrected H_x(N_v0) = {correct.round(6).tolist()}")
    print(f"  naive     H_x(N_v0) = {naive.round(6).tolist()}")
    print(f"  the correction changes the answer: {not np.allclose(correct, naive)}")

    print("\n=== embedded elements: a bar in the plane ===")
    bar = AffineMapping(CellType.LINE, [[0.0, 0.0], [3.0, 4.0]])
    bar.validate()
    bar.verify()
    print(f"  J is {bar.jacobian([[0.0]])[0].shape} (tall, so no determinant)")
    print(
        f"  length = {bar.physical_measure():.6f}   metric G = "
        f"{bar.metric_tensor([[0.0]])[0].round(6).tolist()}"
    )
    projector = bar.jacobian([[0.0]])[0] @ bar.inverse_jacobian([[0.0]])[0]
    print(f"  J J^+ = {projector.round(6).tolist()}  (the tangent projector, not I)")

    print("\n=== degeneracy is scale invariant ===")
    for scale in (1.0, 1e-9):
        tiny = AffineMapping(CellType.TRIANGLE, [[0.0, 0.0], [scale, 0.0], [0.0, scale]])
        print(
            f"  {scale:g} m element: valid={tiny.is_valid()} quality={tiny.quality:.6f} "
            f"area={tiny.physical_measure():.3e}"
        )
    print("  an absolute tolerance on det J would call the nanometre element degenerate")

    print("\n=== the identity is the fixed point ===")
    reference_map = IdentityMapping(CellType.TRIANGLE)
    reference_map.verify()
    print(
        f"  J = {reference_map.jacobian(centre)[0].round(6).tolist()}  "
        f"volume scale = {reference_map.volume_scale(centre)[0]:.6f}"
    )

    print("\nno quadrature, no integration, no assembly - phase 5 complete")


if __name__ == "__main__":
    main()
