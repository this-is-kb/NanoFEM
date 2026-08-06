"""IsotropicElasticity/IsotropicElasticConstitutive: declarations + SDS Section 5 checks."""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.numerics.assembly.contributions import ContributionKind, OperatorRole
from nanofem.numerics.operators.base import Continuity
from nanofem.physics.base import Locality
from nanofem.physics.elasticity.isotropic import (
    IsotropicElasticConstitutive,
    IsotropicElasticity,
)
from nanofem.state.layout import StateLayout
from nanofem.utils.exceptions import PhysicsError


def test_theory_declarations() -> None:
    theory = IsotropicElasticity()
    assert theory.fields() == (("u", 1),)
    assert theory.continuity_requirements() == {"u": Continuity.C0}
    assert theory.required_properties() == ("E",)
    assert theory.required_state() == StateLayout(())
    assert theory.operators_used() == ("symmetric_gradient", "voigt_map")
    assert theory.operator_roles() == (
        OperatorRole.STIFFNESS,
        OperatorRole.MASS,
        OperatorRole.GEOMETRIC_STIFFNESS,
        OperatorRole.FORCE,
    )
    assert theory.contribution_kinds() == (ContributionKind.CELL,)
    assert theory.locality() is Locality.LOCAL


def test_theory_declarations_dim_two() -> None:
    """dim=2 kinematics: a 2-component displacement field, E and nu both required."""
    theory = IsotropicElasticity(dim=2)
    assert theory.fields() == (("u", 2),)
    assert theory.continuity_requirements() == {"u": Continuity.C0}
    assert theory.required_properties() == ("E", "nu")
    assert theory.operators_used() == ("symmetric_gradient", "voigt_map")


def test_theory_accepts_dim_two_and_rejects_dim_three() -> None:
    IsotropicElasticity(dim=2)  # no longer raises - plane stress/strain kinematics
    with pytest.raises(PhysicsError, match="dim=1|dim=2"):
        IsotropicElasticity(dim=3)


def test_constitutive_rejects_dim_other_than_one() -> None:
    """dim=1 axial law only; dim=2 uses Plane{Stress,Strain}Constitutive instead."""
    with pytest.raises(PhysicsError):
        IsotropicElasticConstitutive(dim=2)


def test_constitutive_declarations() -> None:
    model = IsotropicElasticConstitutive()
    assert model.required_properties() == ("E",)
    assert model.state_layout() == StateLayout(())
    assert model.response_components() == 1


def test_stress_is_hookes_law() -> None:
    model = IsotropicElasticConstitutive()
    strains = np.array([[[0.001], [-0.002]]])
    properties = {"E": np.array([[210e9, 210e9]])}
    stress, tangent = model.respond_batch(strains, properties)
    expected_stress = properties["E"][..., np.newaxis] * strains
    np.testing.assert_allclose(stress, expected_stress)
    np.testing.assert_allclose(tangent[0, 0], [[210e9]])
    np.testing.assert_allclose(tangent[0, 1], [[210e9]])


def test_tangent_matches_finite_difference() -> None:
    """SDS Section 5 consistency condition 2: D matches a finite-difference derivative."""
    model = IsotropicElasticConstitutive()
    e_modulus = 71.7e9
    properties = {"E": np.array([[e_modulus]])}
    strain0 = np.array([[[0.0037]]])
    h = 1e-9
    stress_plus, _ = model.respond_batch(strain0 + h, properties)
    stress_minus, _ = model.respond_batch(strain0 - h, properties)
    fd_tangent = (stress_plus - stress_minus) / (2.0 * h)
    _, tangent = model.respond_batch(strain0, properties)
    np.testing.assert_allclose(tangent[0, 0, 0], fd_tangent[0, 0], rtol=1e-6)


def test_null_response() -> None:
    """SDS Section 5 consistency condition 3: zero strain yields zero stress."""
    model = IsotropicElasticConstitutive()
    strains = np.zeros((1, 1, 1))
    stress, _ = model.respond_batch(strains, {"E": np.array([[200e9]])})
    np.testing.assert_allclose(stress, 0.0)
