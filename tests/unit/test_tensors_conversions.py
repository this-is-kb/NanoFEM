"""Voigt/Mandel/full-tensor conversions: round trips, the engineering-shear and
``sqrt(2)`` factors, and the work-conjugacy identity those factors exist to protect.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.numerics.tensors import (
    RepresentationError,
    double_contraction,
    fourth_order_to_mandel,
    full_to_mandel,
    isotropic_stiffness,
    mandel_to_fourth_order,
    mandel_to_full,
    mandel_to_voigt,
    strain_to_voigt,
    stress_to_voigt,
    voigt_to_mandel,
    voigt_to_strain,
    voigt_to_stress,
)

DIMS = (1, 2, 3)


def _random_symmetric(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(dim, dim))
    return 0.5 * (raw + raw.T)


# ---- round trips --------------------------------------------------------------


@pytest.mark.parametrize("dim", DIMS)
def test_strain_voigt_round_trip(dim: int) -> None:
    tensor = _random_symmetric(dim, seed=dim)
    assert np.allclose(voigt_to_strain(strain_to_voigt(tensor)), tensor)


@pytest.mark.parametrize("dim", DIMS)
def test_stress_voigt_round_trip(dim: int) -> None:
    tensor = _random_symmetric(dim, seed=dim + 10)
    assert np.allclose(voigt_to_stress(stress_to_voigt(tensor)), tensor)


@pytest.mark.parametrize("dim", DIMS)
def test_mandel_round_trip(dim: int) -> None:
    tensor = _random_symmetric(dim, seed=dim + 20)
    assert np.allclose(mandel_to_full(full_to_mandel(tensor)), tensor)


@pytest.mark.parametrize("dim", DIMS)
def test_voigt_mandel_round_trip_for_both_kinds(dim: int) -> None:
    tensor = _random_symmetric(dim, seed=dim + 30)
    strain_voigt = strain_to_voigt(tensor)
    assert np.allclose(
        mandel_to_voigt(voigt_to_mandel(strain_voigt, kind="strain"), kind="strain"),
        strain_voigt,
    )
    stress_voigt = stress_to_voigt(tensor)
    assert np.allclose(
        mandel_to_voigt(voigt_to_mandel(stress_voigt, kind="stress"), kind="stress"),
        stress_voigt,
    )


def test_fourth_order_mandel_round_trip() -> None:
    stiffness = isotropic_stiffness(5.0, 2.0, 3)
    matrix = fourth_order_to_mandel(stiffness)
    assert np.allclose(mandel_to_fourth_order(matrix), stiffness)


def test_unsupported_dimension_raises_representation_error() -> None:
    tensor = np.eye(4)
    with pytest.raises(RepresentationError):
        strain_to_voigt(tensor)


# ---- the engineering-shear and sqrt(2) factors ---------------------------------


def test_strain_voigt_doubles_the_off_diagonal_shear() -> None:
    tensor = np.array([[1.0, 0.5], [0.5, 2.0]])
    voigt = strain_to_voigt(tensor)
    assert voigt[0] == pytest.approx(1.0)
    assert voigt[1] == pytest.approx(2.0)
    assert voigt[2] == pytest.approx(1.0)  # gamma_xy = 2 * 0.5


def test_stress_voigt_keeps_the_off_diagonal_plain() -> None:
    tensor = np.array([[1.0, 0.7], [0.7, 2.0]])
    voigt = stress_to_voigt(tensor)
    assert voigt[2] == pytest.approx(0.7)


def test_mandel_uses_sqrt_two_for_both_kinds() -> None:
    tensor = np.array([[1.0, 0.5], [0.5, 2.0]])
    mandel = full_to_mandel(tensor)
    assert mandel[2] == pytest.approx(np.sqrt(2.0) * 0.5)


# ---- work conjugacy -------------------------------------------------------------


@pytest.mark.parametrize("dim", DIMS)
def test_work_conjugacy_agrees_across_representations(dim: int) -> None:
    sigma = _random_symmetric(dim, seed=dim + 40)
    eps = _random_symmetric(dim, seed=dim + 50)

    work_full = float(double_contraction(sigma[np.newaxis], eps[np.newaxis])[0])
    work_voigt = float(stress_to_voigt(sigma) @ strain_to_voigt(eps))
    work_mandel = float(full_to_mandel(sigma) @ full_to_mandel(eps))

    assert work_voigt == pytest.approx(work_full)
    assert work_mandel == pytest.approx(work_full)


def test_work_conjugacy_breaks_without_the_factor_of_two() -> None:
    """The trip test: a strain Voigt vector missing the engineering-shear factor
    no longer reproduces the full-tensor work, proving the factor is load-bearing.
    """
    sigma = np.array([[1.0, 0.3], [0.3, 2.0]])
    eps = np.array([[0.1, 0.05], [0.05, -0.2]])

    work_full = float(double_contraction(sigma[np.newaxis], eps[np.newaxis])[0])
    correct_voigt_work = float(stress_to_voigt(sigma) @ strain_to_voigt(eps))
    assert correct_voigt_work == pytest.approx(work_full)

    # Deliberately broken: plain shear component instead of engineering shear.
    broken_strain_voigt = np.array([eps[0, 0], eps[1, 1], eps[0, 1]])
    broken_work = float(stress_to_voigt(sigma) @ broken_strain_voigt)
    assert broken_work != pytest.approx(work_full)
