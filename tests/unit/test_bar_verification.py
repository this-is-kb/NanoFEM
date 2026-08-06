"""Bar equivalence: closed-form matrix vs. the fully composed operators+tensors path.

This is the test ADR-002/SDS E-5 requires ("declare equivalence to the
composed path in their theory-manual chapter - verification tests enforce
it"): the general finite-element pipeline (mapping + interpolation +
quadrature + symmetric_gradient + constitutive response), built from scratch
here with no call into ``Bar`` itself, must reproduce ``Bar.local_stiffness()``
exactly.

A continuum ``Theory`` integrates its weak form over length only (SDS
Section 1's pipeline is dimensionless in the cross-section); the
cross-sectional area is an *element-layer* multiplier, applied here by hand
after the length integral - not hidden inside the composed path.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.elements.structural.bar import Bar
from nanofem.numerics.interpolation import LagrangeInterpolation, shape_functions
from nanofem.numerics.mapping import AffineMapping
from nanofem.numerics.operators.symmetric_gradient import symmetric_gradient_matrix
from nanofem.numerics.quadrature import quadrature
from nanofem.numerics.reference.enums import CellType
from nanofem.numerics.tensors.fourth_order import isotropic_stiffness
from nanofem.physics.elasticity.isotropic import IsotropicElasticConstitutive


def _composed_stiffness(x0: float, x1: float, young_modulus: float, area: float) -> np.ndarray:
    """Build K_e from the general pipeline; returns the (2, 2) axial stiffness."""
    mapping = AffineMapping(CellType.LINE, [[x0], [x1]])
    interpolation = LagrangeInterpolation(CellType.LINE, order=1)
    basis = shape_functions(interpolation)
    rule = quadrature(CellType.LINE, order=1)  # exact: the integrand is constant

    reference_gradients = basis.derivatives(rule.points)
    physical_gradients = mapping.physical_gradient(reference_gradients, rule.points)
    b_matrix = symmetric_gradient_matrix(physical_gradients)  # (n_qp, 1, 2, 1)
    jacobian = mapping.jacobian_determinant(rule.points)

    n_qp = rule.points.shape[0]
    model = IsotropicElasticConstitutive()
    strains = np.zeros((1, n_qp, 1))
    properties = {"E": np.full((1, n_qp), young_modulus)}
    _, tangent = model.respond_batch(strains, properties)  # (1, n_qp, 1, 1)

    k_continuum = np.zeros((2, 2))
    for q in range(n_qp):
        b_row = b_matrix[q, 0, :, 0]
        d_scalar = float(tangent[0, q, 0, 0])
        k_continuum += rule.weights[q] * jacobian[q] * d_scalar * np.outer(b_row, b_row)
    return area * k_continuum


@pytest.mark.parametrize(
    ("x0", "x1", "e", "a"), [(0.0, 2.5, 71.7e9, 0.0025), (1.0, 4.0, 200e9, 1e-4)]
)
def test_composed_path_matches_closed_form(x0: float, x1: float, e: float, a: float) -> None:
    bar = Bar(0, (0, 1), np.array([[x0], [x1]]), (0, 1), young_modulus=e, area=a)
    composed = _composed_stiffness(x0, x1, e, a)
    np.testing.assert_allclose(composed, bar.local_stiffness(), rtol=1e-9)


def test_constitutive_tangent_matches_isotropic_oracle_at_dim_one() -> None:
    """Independent cross-check: ``isotropic_stiffness`` collapses to E exactly at dim=1.

    ``isotropic_stiffness`` is a verification-oracle-only 3-D-generalized
    tensor (not a materials-layer law); its deviatoric term vanishes
    identically at dim=1, which is a genuine coincidence of the general
    formula, not a hidden dependency of the constitutive model under test.
    """
    rng = np.random.default_rng(0)
    model = IsotropicElasticConstitutive()
    for _ in range(5):
        e = float(rng.uniform(1.0e9, 300.0e9))
        strain = np.array([[[float(rng.uniform(-1e-3, 1e-3))]]])
        _, tangent = model.respond_batch(strain, {"E": np.array([[e]])})
        oracle = isotropic_stiffness(
            bulk_modulus=e, shear_modulus=float(rng.uniform(1e9, 1e11)), dim=1
        )
        np.testing.assert_allclose(tangent[0, 0], oracle[0, 0], rtol=1e-9)
