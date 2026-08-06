"""EulerBernoulliBendingTheory/Constitutive: declarations + SDS Section 5 checks."""

from __future__ import annotations

import numpy as np

from nanofem.numerics.assembly.contributions import ContributionKind, OperatorRole
from nanofem.numerics.operators.base import Continuity
from nanofem.physics.base import Locality
from nanofem.physics.elasticity.euler_bernoulli import (
    EulerBernoulliBendingConstitutive,
    EulerBernoulliBendingTheory,
)
from nanofem.state.layout import StateLayout


def test_theory_declarations() -> None:
    theory = EulerBernoulliBendingTheory()
    assert theory.fields() == (("u", 1), ("r", 1))
    assert theory.continuity_requirements() == {"u": Continuity.C1, "r": Continuity.C1}
    assert theory.required_properties() == ("E",)
    assert theory.required_state() == StateLayout(())
    assert theory.operators_used() == ("second_gradient",)
    assert theory.operator_roles() == (OperatorRole.STIFFNESS, OperatorRole.FORCE)
    assert theory.contribution_kinds() == (ContributionKind.CELL,)
    assert theory.locality() is Locality.LOCAL


def test_constitutive_declarations() -> None:
    model = EulerBernoulliBendingConstitutive()
    assert model.required_properties() == ("E",)
    assert model.state_layout() == StateLayout(())
    assert model.response_components() == 1


def test_moment_is_curvature_times_e() -> None:
    model = EulerBernoulliBendingConstitutive()
    curvatures = np.array([[[0.001], [-0.002]]])
    properties = {"E": np.array([[71.7e9, 71.7e9]])}
    moment, tangent = model.respond_batch(curvatures, properties)
    expected_moment = properties["E"][..., np.newaxis] * curvatures
    np.testing.assert_allclose(moment, expected_moment)
    np.testing.assert_allclose(tangent[0, 0], [[71.7e9]])
    np.testing.assert_allclose(tangent[0, 1], [[71.7e9]])


def test_tangent_matches_finite_difference() -> None:
    """SDS Section 5 consistency condition 2: D matches a finite-difference derivative."""
    model = EulerBernoulliBendingConstitutive()
    e_modulus = 200e9
    properties = {"E": np.array([[e_modulus]])}
    curvature0 = np.array([[[0.0021]]])
    h = 1e-9
    moment_plus, _ = model.respond_batch(curvature0 + h, properties)
    moment_minus, _ = model.respond_batch(curvature0 - h, properties)
    fd_tangent = (moment_plus - moment_minus) / (2.0 * h)
    _, tangent = model.respond_batch(curvature0, properties)
    np.testing.assert_allclose(tangent[0, 0, 0], fd_tangent[0, 0], rtol=1e-6)


def test_null_response() -> None:
    """SDS Section 5 consistency condition 3: zero curvature yields zero moment."""
    model = EulerBernoulliBendingConstitutive()
    curvatures = np.zeros((1, 1, 1))
    moment, _ = model.respond_batch(curvatures, {"E": np.array([[200e9]])})
    np.testing.assert_allclose(moment, 0.0)
