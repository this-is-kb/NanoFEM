"""Euler-Bernoulli beam equivalence: closed-form matrix vs. the fully composed path.

This is the test ADR-002/SDS E-5 requires ("declare equivalence to the
composed path in their theory-manual chapter - verification tests enforce
it"): the general finite-element pipeline (mapping + Hermite interpolation +
quadrature + second_gradient + constitutive response), built from scratch
here with no call into ``EulerBernoulliBeam`` itself, must reproduce
``EulerBernoulliBeam.local_stiffness()`` exactly.

A continuum ``Theory`` integrates its weak form over length only; the second
moment of area ``I`` is an *element-layer* multiplier, applied here by hand
after the length integral - not hidden inside the composed path, exactly as
``test_bar_verification.py`` applies ``area`` by hand.

The one genuinely non-obvious step: ``HermiteInterpolation``'s "derivative"
DOFs are values of ``dw/dxi`` (the *reference*-coordinate derivative), not
``dw/dx`` (the physical rotation this element's global DOF actually is).
Composing the raw ``physical_hessian``/``second_gradient_tensor`` output
directly against physically-parametrized rotation DOFs reproduces a
stiffness matrix off by exactly the Jacobian ``J`` on every rotation
row/column and ``J^2`` on the rotation-rotation term - confirmed by direct
computation while building this test.
``physics.elasticity.euler_bernoulli._reference_derivative_scale`` corrects
it.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.elements.structural.beam_eb import EulerBernoulliBeam
from nanofem.numerics.interpolation import HermiteInterpolation, shape_functions
from nanofem.numerics.mapping import AffineMapping
from nanofem.numerics.operators.second_gradient import second_gradient_tensor
from nanofem.numerics.quadrature import quadrature
from nanofem.numerics.reference.enums import CellType
from nanofem.physics.elasticity.euler_bernoulli import (
    EulerBernoulliBendingConstitutive,
    _curvature_from_hessian,
    _reference_derivative_scale,
)


def _composed_stiffness(
    x0: float, x1: float, young_modulus: float, second_moment: float
) -> np.ndarray:
    """Build K_e from the general pipeline; returns the (4, 4) bending stiffness."""
    mapping = AffineMapping(CellType.LINE, [[x0], [x1]])
    interpolation = HermiteInterpolation(CellType.LINE, order=3)
    basis = shape_functions(interpolation)
    rule = quadrature(CellType.LINE, order=2)  # curvature is linear in x; the B^T D B
    # integrand is quadratic - a 2-point Gauss rule (exact to degree 3) covers it

    reference_gradients = basis.derivatives(rule.points)
    reference_hessians = basis.hessian(rule.points)
    physical_hessians = mapping.physical_hessian(
        reference_gradients, reference_hessians, rule.points
    )
    hessians = second_gradient_tensor(physical_hessians)
    curvature_shapes = _curvature_from_hessian(hessians)  # (n_qp, 4), reference-derivative DOFs

    jacobian = mapping.jacobian_determinant(rule.points)
    dof_orders = tuple(dof.derivative[0] for dof in interpolation.dofs)  # (0, 1, 0, 1)
    scale = _reference_derivative_scale(dof_orders, float(jacobian[0]))
    curvature_shapes = curvature_shapes * scale  # now in physical-rotation DOFs

    n_qp = rule.points.shape[0]
    model = EulerBernoulliBendingConstitutive()
    strains = np.zeros((1, n_qp, 1))
    properties = {"E": np.full((1, n_qp), young_modulus)}
    _, tangent = model.respond_batch(strains, properties)

    k_continuum = np.zeros((4, 4))
    for q in range(n_qp):
        b_row = curvature_shapes[q, :]
        d_scalar = float(tangent[0, q, 0, 0])
        k_continuum += rule.weights[q] * jacobian[q] * d_scalar * np.outer(b_row, b_row)
    return second_moment * k_continuum


@pytest.mark.parametrize(
    ("x0", "x1", "e", "i"), [(0.0, 2.5, 71.7e9, 8.0e-8), (1.0, 4.0, 200e9, 3.2e-6)]
)
def test_composed_path_matches_closed_form(x0: float, x1: float, e: float, i: float) -> None:
    beam = EulerBernoulliBeam(
        0, (0, 1), np.array([[x0], [x1]]), (0, 1, 2, 3), young_modulus=e, second_moment=i
    )
    composed = _composed_stiffness(x0, x1, e, i)
    np.testing.assert_allclose(composed, beam.local_stiffness(), rtol=1e-9)
