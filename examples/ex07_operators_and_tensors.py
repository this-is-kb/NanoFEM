"""v0.7.0 success criterion, executable: a triangle's shape-function gradients and
mapping push physical gradients forward -> every operator recipe runs on them ->
the resulting strain is converted to Voigt and Mandel form -> work conjugacy is
checked -> both this package's and tensors' verification suites run.

No element, no assembly, no constitutive model, no material.
"""

from __future__ import annotations

import numpy as np

from nanofem.numerics.interpolation import LagrangeInterpolation, shape_functions
from nanofem.numerics.mapping import AffineMapping
from nanofem.numerics.operators import (
    curl_matrix,
    divergence_matrix,
    gradient_matrix,
    helmholtz_matrix,
    identity_vector,
    is_operator_library_valid,
    laplacian_matrix,
    mass_term,
    surface_gradient_matrix,
    symmetric_gradient_matrix,
    tensor_transformation_matrix_voigt,
    vector_transformation_matrix,
    verify_operator_library,
)
from nanofem.numerics.quadrature import quadrature
from nanofem.numerics.reference.enums import CellType
from nanofem.numerics.tensors import (
    is_tensor_library_valid,
    stress_to_voigt,
    verify_tensor_library,
)
from nanofem.numerics.tensors.voigt import full_to_mandel, voigt_to_strain


def main() -> None:
    """Push a triangle's gradients through every operator recipe, then verify both layers."""
    print("=== build a real physical gradient batch, no mechanics involved yet ===")
    nodes = [[1.0, 1.0], [3.0, 1.0], [1.0, 4.0]]
    mapping = AffineMapping(CellType.TRIANGLE, nodes)
    basis = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 1))
    rule = quadrature(CellType.TRIANGLE, order=2)

    reference_gradient = basis.gradient(rule.points)
    physical_gradients = mapping.physical_gradient(reference_gradient, rule.points)
    volume_scale = mapping.volume_scale(rule.points)
    print(
        f"  {rule.points.shape[0]} quadrature points, physical_gradients shape "
        f"{physical_gradients.shape}"
    )

    print("\n=== gradient, divergence, curl ===")
    dofs_vector = np.array([[0.05, -0.02], [0.03, 0.04], [-0.01, 0.06]])
    grad = gradient_matrix(physical_gradients)
    div_row = divergence_matrix(physical_gradients)
    curl_row = curl_matrix(physical_gradients)
    divergence = np.einsum("qoai,ai->qo", div_row, dofs_vector)[:, 0]
    theta = np.einsum("qoai,ai->qo", curl_row, dofs_vector)[:, 0]
    print(f"  grad_matrix shape       = {grad.shape}")
    print(f"  divergence at each qp   = {divergence.round(6).tolist()}")
    print(f"  curl (2-D scalar) at qp = {theta.round(6).tolist()}")

    print("\n=== symmetric gradient (the B operator) -> Voigt -> Mandel ===")
    b_matrix = symmetric_gradient_matrix(physical_gradients)
    strain_voigt = np.einsum("qvai,ai->qv", b_matrix, dofs_vector)
    strain_full = voigt_to_strain(strain_voigt)
    print(f"  strain (Voigt) at qp[0] = {strain_voigt[0].round(6).tolist()}")
    print(f"  strain (full)  at qp[0] = {strain_full[0].round(6).tolist()}")

    print("\n=== work conjugacy: full, Voigt, and Mandel agree ===")
    stress_full_0 = 2.0 * strain_full[0]  # a toy "constitutive law", nothing physical
    work_full = float(np.einsum("ij,ij->", stress_full_0, strain_full[0]))
    work_voigt = float(stress_to_voigt(stress_full_0) @ strain_voigt[0])
    work_mandel = float(full_to_mandel(stress_full_0) @ full_to_mandel(strain_full[0]))
    print(f"  work (full tensor, qp0) = {work_full:.9f}")
    print(f"  work (Voigt, qp0)       = {work_voigt:.9f}")
    print(f"  work (Mandel, qp0)      = {work_mandel:.9f}")

    print("\n=== Laplacian and Helmholtz ===")
    laplacian = laplacian_matrix(physical_gradients, rule.weights, volume_scale)
    values = basis.evaluate(rule.points)
    mass = mass_term(values, rule.weights, volume_scale)
    helmholtz_0 = helmholtz_matrix(values, physical_gradients, rule.weights, volume_scale, 0.0)
    helmholtz_l = helmholtz_matrix(values, physical_gradients, rule.weights, volume_scale, 0.3)
    print(f"  laplacian_matrix =\n{laplacian.round(6)}")
    print(f"  helmholtz(l=0) == mass term: {np.allclose(helmholtz_0, mass)}")
    print(
        f"  helmholtz(l=0.3) - mass term == l^2 * laplacian: "
        f"{np.allclose(helmholtz_l - mass, 0.3**2 * laplacian)}"
    )

    print("\n=== surface gradient (a facet normal, chosen by hand here) ===")
    normals = np.tile(np.array([[1.0, 0.0]]), (rule.points.shape[0], 1))
    surface_row = surface_gradient_matrix(physical_gradients, normals)
    print(f"  surface_gradient_matrix shape = {surface_row.shape}")

    print("\n=== Voigt operators and transformation ===")
    m = identity_vector(2)
    print(
        f"  identity_vector(2) . strain_voigt[0] = {float(m @ strain_voigt[0]):.6f}  "
        f"(== trace of strain[0] = {np.trace(strain_full[0]):.6f})"
    )
    angle = np.pi / 6
    q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    t_vector = vector_transformation_matrix(q)
    m_sigma = tensor_transformation_matrix_voigt(q, kind="stress")
    print(f"  vector_transformation_matrix(30 deg) =\n{t_vector.round(6)}")
    print(f"  Bond stress matrix M_sigma =\n{m_sigma.round(6)}")

    print("\n=== verify both layers ===")
    verify_tensor_library()
    verify_operator_library()
    print(f"  tensors:   is_tensor_library_valid()   = {is_tensor_library_valid()}")
    print(f"  operators: is_operator_library_valid() = {is_operator_library_valid()}")

    print("\nno element, no assembly, no constitutive model - v0.7.0 complete")


if __name__ == "__main__":
    main()
