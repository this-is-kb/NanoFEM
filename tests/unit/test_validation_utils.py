"""Unit tests for the reconstructed validation preconditions (data guards, no math)."""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.utils.exceptions import InputValidationError
from nanofem.utils.validation import (
    require_finite,
    require_finite_array,
    require_identifier,
    require_in_open_interval,
    require_non_negative,
    require_positive,
    require_positive_int,
    require_shape,
)


def test_positive_accepts_and_rejects() -> None:
    """Positive check: accepts > 0, rejects zero, negatives, non-finite."""
    require_positive(1.5, "x")
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(InputValidationError):
            require_positive(bad, "x")


def test_non_negative_boundary() -> None:
    """Zero is legal (rho = 0, e0a = 0 local limit; SDS Section 6)."""
    require_non_negative(0.0, "rho")
    with pytest.raises(InputValidationError):
        require_non_negative(-1e-9, "rho")


def test_positive_int_rejects_bool_and_floats() -> None:
    """Integer check: bools and floats are not counts."""
    require_positive_int(3, "n")
    for bad in (0, -1, True, 2.0):
        with pytest.raises(InputValidationError):
            require_positive_int(bad, "n")  # type: ignore[arg-type]


def test_open_interval_is_open() -> None:
    """Endpoints excluded: nu = 0.5 must fail, nu = -0.3 (auxetic) must pass."""
    require_in_open_interval(-0.3, -1.0, 0.5, "nu")
    with pytest.raises(InputValidationError):
        require_in_open_interval(0.5, -1.0, 0.5, "nu")


def test_finite_scalar_and_array() -> None:
    """Scalar and array finiteness are separate helpers with distinct semantics."""
    require_finite(-2.0, "v")
    require_finite_array(np.array([1.0, 2.0]), "a")
    with pytest.raises(InputValidationError):
        require_finite(float("nan"), "v")
    with pytest.raises(InputValidationError):
        require_finite_array(np.array([1.0, np.inf]), "a")


def test_identifier_rejects_padding_and_empty() -> None:
    """Identifiers are non-empty and unpadded."""
    require_identifier("left_edge", "name")
    for bad in ("", " pad", "pad "):
        with pytest.raises(InputValidationError):
            require_identifier(bad, "name")


def test_shape_with_wildcard() -> None:
    """-1 means any extent along that axis."""
    require_shape(np.zeros((4, 2)), (-1, 2), "coords")
    with pytest.raises(InputValidationError):
        require_shape(np.zeros((4, 3)), (-1, 2), "coords")
