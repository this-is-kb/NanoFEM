"""CircularSection: every property against an independently hand-computed closed form (SDS 2.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nanofem.geometry.standard import CircularSection
from nanofem.utils.exceptions import InputValidationError


def test_area() -> None:
    section = CircularSection(radius=2.0)
    assert section.area() == pytest.approx(math.pi * 4.0)


def test_second_moments_equal_and_correct() -> None:
    section = CircularSection(radius=2.0)
    expected = math.pi * 2.0**4 / 4.0
    assert section.second_moment_y() == pytest.approx(expected)
    assert section.second_moment_z() == pytest.approx(expected)


def test_polar_moment_is_sum_of_second_moments() -> None:
    section = CircularSection(radius=2.0)
    assert section.polar_moment() == pytest.approx(math.pi * 2.0**4 / 2.0)
    assert section.polar_moment() == pytest.approx(
        section.second_moment_y() + section.second_moment_z()
    )


def test_torsion_constant_equals_polar_moment() -> None:
    """The circular-only identity J_t = I_p (SDS 2.2)."""
    section = CircularSection(radius=1.5)
    assert section.torsion_constant() == pytest.approx(section.polar_moment())


def test_shear_correction_cowper_factor() -> None:
    section = CircularSection(radius=1.0)
    nu = 0.3
    expected = 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)
    assert section.shear_correction(nu) == pytest.approx(expected)
    assert section.shear_correction(nu) == pytest.approx(0.8863636363636364)


def test_warping_constant_is_zero() -> None:
    assert CircularSection(radius=3.0).warping_constant() == 0.0


def test_centroid_and_shear_center_at_origin() -> None:
    section = CircularSection(radius=1.0)
    np.testing.assert_array_equal(section.centroid(), np.zeros(2))
    np.testing.assert_array_equal(section.shear_center(), np.zeros(2))
    np.testing.assert_array_equal(section.shear_center(), section.centroid())


def test_rejects_nonpositive_radius() -> None:
    with pytest.raises(InputValidationError):
        CircularSection(radius=0.0)
    with pytest.raises(InputValidationError):
        CircularSection(radius=-1.0)
