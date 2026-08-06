"""Timoshenko beam equivalence: closed-form matrix vs. the fully composed SRI path.

This is the test ADR-002/SDS E-5 requires: the general finite-element
pipeline (mapping + linear Lagrange interpolation + selective-reduced-
integration quadrature + gradient + constitutive response), built from
scratch here with no call into ``TimoshenkoBeam`` itself, must reproduce
``TimoshenkoBeam.local_stiffness()`` exactly.

Per SDS clause E-3 ("Timoshenko declares selective-reduced integration of
the shear term"): the bending term (``kappa = d(theta)/dx``) is integrated
with a full 2-point Gauss rule; the shear term (``gamma = dw/dx - theta``)
is integrated with a reduced 1-point Gauss rule, evaluated at the element
midpoint. ``I``/``A_s`` are element-layer multipliers, applied by hand after
each term's length integral - not hidden inside the composed path, exactly
as ``test_bar_verification.py`` applies ``area`` by hand.

A single element is **not** exact for a cantilever under this formulation
(unlike ``Bar``/``EulerBernoulliBeam``, both exact for one element) - this
was verified directly this session and is a genuine, textbook-correct
property of the selective-reduced-integration element, not a defect: the
"exact" Phi-parametrized Timoshenko element found in most textbooks uses a
different shape-function family (derived from the governing ODEs) that this
codebase's ``Interpolation`` framework does not have. What SRI *does*
guarantee - and what this file's mesh-convergence test checks - is
monotonic convergence to the exact cantilever solution
``delta = PL^3/(3EI) + PL/(GA_s)`` as the mesh refines, with no shear
locking.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.elements.structural.beam_timoshenko import TimoshenkoBeam
from nanofem.numerics.interpolation import LagrangeInterpolation, shape_functions
from nanofem.numerics.mapping import AffineMapping
from nanofem.numerics.operators.gradient import gradient_matrix
from nanofem.numerics.quadrature import quadrature
from nanofem.numerics.reference.enums import CellType
from nanofem.physics.elasticity.timoshenko import TimoshenkoBeamConstitutive


def _element_stiffness(
    x0: float, x1: float, young_modulus: float, shear_modulus: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build K_e from the general pipeline (I, A_s = 1; applied by the caller)."""
    mapping = AffineMapping(CellType.LINE, [[x0], [x1]])
    interpolation = LagrangeInterpolation(CellType.LINE, order=1)
    basis = shape_functions(interpolation)
    model = TimoshenkoBeamConstitutive()

    # Bending: full (2-point) integration of kappa = d(theta)/dx.
    bending_rule = quadrature(CellType.LINE, order=2)
    reference_gradients = basis.derivatives(bending_rule.points)
    physical_gradients = gradient_matrix(
        mapping.physical_gradient(reference_gradients, bending_rule.points)
    )
    jacobian_b = mapping.jacobian_determinant(bending_rule.points)
    n_qp_b = bending_rule.points.shape[0]
    strains_b = np.zeros((1, n_qp_b, 2))
    properties = {
        "E": np.full((1, n_qp_b), young_modulus),
        "G": np.full((1, n_qp_b), shear_modulus),
    }
    _, tangent_b = model.respond_batch(strains_b, properties)

    k_bend = np.zeros((4, 4))
    for q in range(n_qp_b):
        row = np.zeros(4)
        row[1] = physical_gradients[q, 0, 0]
        row[3] = physical_gradients[q, 0, 1]
        d_kappa = float(tangent_b[0, q, 0, 0])
        k_bend += bending_rule.weights[q] * jacobian_b[q] * d_kappa * np.outer(row, row)

    # Shear: reduced (1-point) integration of gamma = dw/dx - theta.
    shear_rule = quadrature(CellType.LINE, order=0)
    reference_gradients_s = basis.derivatives(shear_rule.points)
    physical_gradients_s = gradient_matrix(
        mapping.physical_gradient(reference_gradients_s, shear_rule.points)
    )
    values_s = basis.evaluate(shear_rule.points)
    jacobian_s = mapping.jacobian_determinant(shear_rule.points)
    n_qp_s = shear_rule.points.shape[0]
    strains_s = np.zeros((1, n_qp_s, 2))
    properties_s = {
        "E": np.full((1, n_qp_s), young_modulus),
        "G": np.full((1, n_qp_s), shear_modulus),
    }
    _, tangent_s = model.respond_batch(strains_s, properties_s)

    k_shear = np.zeros((4, 4))
    for q in range(n_qp_s):
        row = np.zeros(4)
        row[0] = physical_gradients_s[q, 0, 0]
        row[2] = physical_gradients_s[q, 0, 1]
        row[1] -= values_s[q, 0]
        row[3] -= values_s[q, 1]
        d_gamma = float(tangent_s[0, q, 1, 1])
        k_shear += shear_rule.weights[q] * jacobian_s[q] * d_gamma * np.outer(row, row)

    return k_bend, k_shear


def _composed_stiffness(
    x0: float,
    x1: float,
    young_modulus: float,
    shear_modulus: float,
    second_moment: float,
    shear_area: float,
) -> np.ndarray:
    k_bend, k_shear = _element_stiffness(x0, x1, young_modulus, shear_modulus)
    return second_moment * k_bend + shear_area * k_shear


@pytest.mark.parametrize(
    ("x0", "x1", "e", "g", "i", "a_s"),
    [(0.0, 2.5, 71.7e9, 26.0e9, 8.0e-8, 0.0025), (1.0, 4.0, 200e9, 80e9, 3.2e-6, 0.001)],
)
def test_composed_path_matches_closed_form(
    x0: float, x1: float, e: float, g: float, i: float, a_s: float
) -> None:
    beam = TimoshenkoBeam(
        0,
        (0, 1),
        np.array([[x0], [x1]]),
        (0, 1, 2, 3),
        young_modulus=e,
        shear_modulus=g,
        second_moment=i,
        shear_area=a_s,
    )
    composed = _composed_stiffness(x0, x1, e, g, i, a_s)
    np.testing.assert_allclose(composed, beam.local_stiffness(), rtol=1e-9)


def _cantilever_tip_deflection(
    n_elements: int, total_length: float, e: float, i: float, g: float, a_s: float, tip_force: float
) -> float:
    """Chain n_elements TimoshenkoBeams into a cantilever; solve for the tip deflection."""
    n_nodes = n_elements + 1
    n_dof = 2 * n_nodes
    stiffness = np.zeros((n_dof, n_dof))
    element_length = total_length / n_elements
    for element_index in range(n_elements):
        x0 = element_index * element_length
        x1 = (element_index + 1) * element_length
        dofs = (
            2 * element_index,
            2 * element_index + 1,
            2 * element_index + 2,
            2 * element_index + 3,
        )
        beam = TimoshenkoBeam(
            element_index,
            (element_index, element_index + 1),
            np.array([[x0], [x1]]),
            dofs,
            young_modulus=e,
            shear_modulus=g,
            second_moment=i,
            shear_area=a_s,
        )
        local = beam.local_stiffness()
        for a in range(4):
            for b in range(4):
                stiffness[dofs[a], dofs[b]] += local[a, b]

    force = np.zeros(n_dof)
    force[-2] = tip_force
    free_dofs = list(range(2, n_dof))
    reduced = stiffness[np.ix_(free_dofs, free_dofs)]
    solution = np.linalg.solve(reduced, force[free_dofs])
    return float(solution[-2])


def test_mesh_convergence_toward_exact_cantilever_solution() -> None:
    """SRI Timoshenko does not lock: refinement converges monotonically to the exact solution.

    A single element is not exact here (unlike Bar/EulerBernoulliBeam) - this
    is the correct, textbook property of this formulation. Confirmed this
    session: ratios 0.751 -> 0.938 -> 0.984 -> 0.996 -> 0.999 -> 0.9998 for
    n_elements = 1, 2, 4, 8, 16, 32.
    """
    length, young_modulus, second_moment = 3.0, 200e9, 4.0e-6
    shear_modulus, shear_area, tip_force = 80e9, 0.001, 1_000.0
    exact = tip_force * length**3 / (3.0 * young_modulus * second_moment) + tip_force * length / (
        shear_modulus * shear_area
    )

    ratios = []
    for n_elements in (1, 2, 4, 8, 16, 32):
        deflection = _cantilever_tip_deflection(
            n_elements, length, young_modulus, second_moment, shear_modulus, shear_area, tip_force
        )
        ratios.append(deflection / exact)

    assert ratios[0] < 0.8  # a single element is deliberately not exact
    # pairwise comparison; ratios[1:] is deliberately one element shorter
    for earlier, later in zip(ratios, ratios[1:], strict=False):
        assert later > earlier  # monotonic convergence, no locking
    assert ratios[-1] == pytest.approx(1.0, rel=1e-3)
