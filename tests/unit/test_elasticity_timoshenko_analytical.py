"""TimoshenkoBeamTheory/Constitutive: declarations + SDS Section 5 checks."""

from __future__ import annotations

import numpy as np

from nanofem.numerics.assembly.contributions import ContributionKind, OperatorRole
from nanofem.numerics.operators.base import Continuity
from nanofem.physics.base import Locality
from nanofem.physics.elasticity.timoshenko import (
    TimoshenkoBeamConstitutive,
    TimoshenkoBeamTheory,
)
from nanofem.state.layout import StateLayout


def test_theory_declarations() -> None:
    theory = TimoshenkoBeamTheory()
    assert theory.fields() == (("u", 1), ("r", 1))
    # C0, not C1: theta is an independent field, curvature = d(theta)/dx is a first
    # derivative - the precise kinematic reason Timoshenko needs no Hermite/C1 basis.
    assert theory.continuity_requirements() == {"u": Continuity.C0, "r": Continuity.C0}
    assert theory.required_properties() == ("E", "G")
    assert theory.required_state() == StateLayout(())
    assert theory.operators_used() == ("gradient",)
    assert theory.operator_roles() == (OperatorRole.STIFFNESS, OperatorRole.FORCE)
    assert theory.contribution_kinds() == (ContributionKind.CELL,)
    assert theory.locality() is Locality.LOCAL


def test_constitutive_declarations() -> None:
    model = TimoshenkoBeamConstitutive()
    assert model.required_properties() == ("E", "G")
    assert model.state_layout() == StateLayout(())
    assert model.response_components() == 2


def test_stress_is_diagonal_hookes_law() -> None:
    """[M_per_I, V_per_As] = [E*kappa, G*gamma]; the two responses do not couple."""
    model = TimoshenkoBeamConstitutive()
    strains = np.array([[[0.001, -0.0005], [-0.002, 0.0007]]])  # (1, 2, 2): [kappa, gamma]
    properties = {"E": np.array([[71.7e9, 71.7e9]]), "G": np.array([[26.0e9, 26.0e9]])}
    stress, tangent = model.respond_batch(strains, properties)
    expected_stress = np.stack(
        [properties["E"] * strains[..., 0], properties["G"] * strains[..., 1]], axis=-1
    )
    np.testing.assert_allclose(stress, expected_stress)
    for qp in range(2):
        np.testing.assert_allclose(tangent[0, qp], [[71.7e9, 0.0], [0.0, 26.0e9]])


def test_tangent_matches_finite_difference_per_component() -> None:
    """SDS Section 5 consistency condition 2, checked independently for kappa and gamma."""
    model = TimoshenkoBeamConstitutive()
    e_modulus, g_modulus = 200e9, 80e9
    properties = {"E": np.array([[e_modulus]]), "G": np.array([[g_modulus]])}
    strain0 = np.array([[[0.0021, -0.0013]]])
    h = 1e-9

    for component in (0, 1):
        perturbation = np.zeros_like(strain0)
        perturbation[..., component] = h
        stress_plus, _ = model.respond_batch(strain0 + perturbation, properties)
        stress_minus, _ = model.respond_batch(strain0 - perturbation, properties)
        fd_column = (stress_plus - stress_minus) / (2.0 * h)
        _, tangent = model.respond_batch(strain0, properties)
        np.testing.assert_allclose(tangent[0, 0, :, component], fd_column[0, 0], rtol=1e-6)


def test_null_response() -> None:
    """SDS Section 5 consistency condition 3: zero strain yields zero stress."""
    model = TimoshenkoBeamConstitutive()
    strains = np.zeros((1, 1, 2))
    stress, _ = model.respond_batch(strains, {"E": np.array([[200e9]]), "G": np.array([[80e9]])})
    np.testing.assert_allclose(stress, 0.0)
