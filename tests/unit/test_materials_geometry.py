"""Unit tests for Material and section geometry (requirements 8-9)."""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.geometry.custom import CustomSection
from nanofem.geometry.standard import (
    HollowCircularSection,
    HollowRectangularSection,
    ISection,
    RectangularSection,
)
from nanofem.materials.grading import ExponentialGrading, PowerLawGrading
from nanofem.materials.material import Material
from nanofem.materials.properties import CANONICAL_KEYS, SpatialProperty
from nanofem.utils.exceptions import InputValidationError


def test_material_bounds_welcome_auxetics() -> None:
    """nu in (-1, 0.5) open: -0.3 legal (auxetic by design), 0.6 rejected."""
    m = Material("aux", E=1.0e9, nu=-0.3, rho=1200.0)
    assert m.value("nu") == -0.3 and m.defines("E") and not m.defines("G")
    with pytest.raises(InputValidationError):
        Material("bad", E=1.0, nu=0.6)
    with pytest.raises(InputValidationError):
        Material("bad", E=-1.0, nu=0.3)


def test_material_g_consistency_check() -> None:
    """E, nu, G together must satisfy G = E / (2(1+nu)) (SDS Section 6)."""
    Material("ok", E=200.0e9, nu=0.3, rho=7850.0, G=200.0e9 / 2.6)
    with pytest.raises(InputValidationError):
        Material("bad", E=200.0e9, nu=0.3, G=1.0e9)


def test_material_key_policy() -> None:
    """Unknown bare keys raise listing canonical keys; dotted user keys pass."""
    with pytest.raises(InputValidationError, match="canonical"):
        Material("m", zeta=1.0)
    m = Material("m", **{"E": 1.0, "user.xi2": 0.3})
    assert m.value("user.xi2") == 0.3
    with pytest.raises(InputValidationError, match="does not define"):
        m.value("nu")


def test_material_round_trip_and_negative_surface_modulus() -> None:
    """Serialization inverts exactly; mu_s may be negative (Miller-Shenoy)."""
    m = Material("surf", E=1.0, nu=0.25, mu_s=-2.5, tau0=0.1)
    m2 = Material.from_dict(m.to_dict())
    assert m2.name == "surf" and m2.value("mu_s") == -2.5
    assert sorted(m2.keys) == sorted(m.keys)


def test_sections_store_and_defer_computation() -> None:
    """Non-circular sections validate and store dimensions; computation stays phase 2.

    ``CircularSection`` is excluded here: its properties are real closed forms
    (walking skeleton), covered separately in test_geometry_circular_analytical.py.
    """
    r = RectangularSection(width=0.02, height=0.04)
    with pytest.raises(InputValidationError):
        RectangularSection(width=-1.0, height=1.0)
    with pytest.raises(NotImplementedError):
        r.area()


def test_hollow_and_i_sections_store_and_defer_computation() -> None:
    """HollowCircular/HollowRectangular/ISection validate dimensions; computation stays phase 2."""
    hollow_circle = HollowCircularSection(outer_radius=0.02, inner_radius=0.015)
    with pytest.raises(InputValidationError, match="smaller than outer_radius"):
        HollowCircularSection(outer_radius=0.01, inner_radius=0.02)

    hollow_rect = HollowRectangularSection(width=0.1, height=0.05, thickness=0.005)
    with pytest.raises(InputValidationError, match="below half the smaller side"):
        HollowRectangularSection(width=0.1, height=0.05, thickness=0.03)

    i_section = ISection(
        flange_width=0.1, flange_thickness=0.01, web_height=0.2, web_thickness=0.008
    )
    with pytest.raises(InputValidationError):
        ISection(flange_width=-0.1, flange_thickness=0.01, web_height=0.2, web_thickness=0.008)

    for section in (hollow_circle, hollow_rect, i_section):
        with pytest.raises(NotImplementedError):
            section.area()


def test_custom_section_lookup() -> None:
    """CustomSection serves stored values and raises on missing keys."""
    s = CustomSection("measured", area=3.0e-4, second_moment_z=2.0e-8)
    assert s.area() == 3.0e-4
    with pytest.raises(Exception, match="warping|measured"):
        s.warping_constant()


def test_grading_laws_validate_but_defer_evaluation() -> None:
    """PowerLawGrading/ExponentialGrading store validated parameters; evaluate() is phase-2."""
    power = PowerLawGrading(p_top=200e9, p_bottom=70e9, thickness=0.01, exponent=1.5)
    with pytest.raises(InputValidationError):
        PowerLawGrading(p_top=200e9, p_bottom=70e9, thickness=-0.01, exponent=1.5)
    with pytest.raises(NotImplementedError):
        power.evaluate(np.zeros(3))

    exponential = ExponentialGrading(p_reference=100e9, rate=2.0)
    with pytest.raises(NotImplementedError):
        exponential.evaluate(np.zeros(3))


def test_canonical_keys_matches_material_bounds() -> None:
    """CANONICAL_KEYS is derived from Material's own bounds table, not a second copy."""
    assert set(CANONICAL_KEYS) == {
        "E",
        "nu",
        "G",
        "rho",
        "alpha_thermal",
        "eta_damping",
        "e0a",
        "l_sg",
        "l_cs",
        "mu_s",
        "lambda_s",
        "tau0",
        "rho_s",
    }
    Material("m", **{key: 1.0 for key in ("rho", "alpha_thermal")})


def test_spatial_property_binds_key_to_grading_law() -> None:
    """SpatialProperty stores a validated (key, GradingLaw) binding; evaluation is phase-2."""
    law = ExponentialGrading(p_reference=100e9, rate=2.0)
    bound = SpatialProperty("E", law)
    assert bound.key == "E"
    assert bound.law is law
