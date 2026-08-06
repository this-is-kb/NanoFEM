"""EringenDifferentialTheory / EringenDifferentialMaterial: declarations and delegation.

Companion to ``test_nonlocal_continuum_element.py`` (the element-level weak-form checks) - this
file covers only the ``Theory``/``ConstitutiveModel`` layer: field declarations, dimension
rejection, Voigt component naming, and that ``EringenDifferentialMaterial`` genuinely delegates
to its wrapped classical law rather than duplicating the formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.numerics.operators.base import Continuity
from nanofem.physics.base import Locality
from nanofem.physics.elasticity.eringen_differential import (
    DISPLACEMENT_FIELD,
    NONLOCAL_STRAIN_FIELD,
    EringenDifferentialMaterial,
    EringenDifferentialTheory,
    voigt_component_names,
)
from nanofem.physics.elasticity.plane import PlaneStrainConstitutive, PlaneStressConstitutive
from nanofem.utils.exceptions import PhysicsError


def test_voigt_component_names_dim_2() -> None:
    assert voigt_component_names(2) == ("xx", "yy", "xy")


def test_theory_declares_two_fields() -> None:
    theory = EringenDifferentialTheory(dim=2)
    assert theory.fields() == ((DISPLACEMENT_FIELD, 2), (NONLOCAL_STRAIN_FIELD, 3))
    assert theory.dim == 2


def test_theory_names_e_star_components_by_voigt_pair_not_positionally() -> None:
    theory = EringenDifferentialTheory(dim=2)
    overrides = theory.field_component_names()
    assert overrides == {NONLOCAL_STRAIN_FIELD: ("xx", "yy", "xy")}


def test_theory_continuity_is_c0_for_both_fields() -> None:
    theory = EringenDifferentialTheory(dim=2)
    continuity = theory.continuity_requirements()
    assert continuity == {DISPLACEMENT_FIELD: Continuity.C0, NONLOCAL_STRAIN_FIELD: Continuity.C0}


def test_theory_required_properties() -> None:
    theory = EringenDifferentialTheory(dim=2)
    assert theory.required_properties() == ("E", "nu", "e0a")


def test_theory_locality_is_local_not_pairwise() -> None:
    """Architecturally local (a multi-field theory), despite the physics being 'nonlocal' -
    PAIRWISE is reserved for the integral model (kernel + neighbor search), out of scope."""
    theory = EringenDifferentialTheory(dim=2)
    assert theory.locality() is Locality.LOCAL


def test_theory_rejects_dim_other_than_two() -> None:
    EringenDifferentialTheory(dim=2)  # does not raise
    with pytest.raises(PhysicsError, match="dim=2"):
        EringenDifferentialTheory(dim=1)
    with pytest.raises(PhysicsError, match="dim=2"):
        EringenDifferentialTheory(dim=3)


@pytest.mark.parametrize(
    "local_law_cls", [PlaneStressConstitutive, PlaneStrainConstitutive], ids=["stress", "strain"]
)
def test_material_delegates_respond_batch_to_the_wrapped_law(local_law_cls: type) -> None:
    """sigma* = D @ e_star, with D identical to the wrapped classical law's own D."""
    local_law = local_law_cls()
    material = EringenDifferentialMaterial(local_law)
    strains = np.array([[[0.001, -0.0007, 0.0004]]])
    properties = {
        "E": np.array([[200e9]]),
        "nu": np.array([[0.3]]),
        "e0a": np.array([[0.5]]),  # must be tolerated (stripped), not passed to the local law
    }
    stress, tangent = material.respond_batch(strains, properties)
    expected_stress, expected_tangent = local_law.respond_batch(
        strains, {"E": properties["E"], "nu": properties["nu"]}
    )
    np.testing.assert_allclose(stress, expected_stress)
    np.testing.assert_allclose(tangent, expected_tangent)


def test_material_required_properties_adds_e0a_to_the_wrapped_laws_own() -> None:
    material = EringenDifferentialMaterial(PlaneStressConstitutive())
    assert material.required_properties() == ("E", "nu", "e0a")


def test_material_response_components_and_state_layout_are_delegated() -> None:
    local_law = PlaneStrainConstitutive()
    material = EringenDifferentialMaterial(local_law)
    assert material.response_components() == local_law.response_components()
    assert material.state_layout() == local_law.state_layout()


def test_material_local_law_property_returns_the_wrapped_instance() -> None:
    local_law = PlaneStressConstitutive()
    material = EringenDifferentialMaterial(local_law)
    assert material.local_law is local_law
