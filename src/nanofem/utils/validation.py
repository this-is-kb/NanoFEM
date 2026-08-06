"""Reusable precondition checks raising ``InputValidationError`` with context.

These are data-integrity guards (SDS fail-fast policy), not numerical
algorithms: every public constructor validates immediately, with the
offending value in the message. Scalar checks take floats; array checks are
suffixed ``_array``.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from nanofem.utils.exceptions import InputValidationError

_EnumT = TypeVar("_EnumT", bound=Enum)


def resolve_enum_member(value: _EnumT | str, enum_cls: type[_EnumT], label: str) -> _EnumT:
    """Coerce ``value`` to a member of ``enum_cls``; an unknown string lists valid members.

    The shared form of the "accept an enum instance or its string value" coercion
    every registry/factory module needs (``numerics/reference/registry.py``,
    ``numerics/interpolation/registry.py``, ...) - previously reimplemented
    identically in each.
    """
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError:
        known = ", ".join(sorted(member.value for member in enum_cls))
        raise InputValidationError(f"unknown {label} {value!r}; valid: {known}") from None


def require_finite(value: float, name: str) -> None:
    """Raise unless ``value`` is a finite number."""
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise InputValidationError(f"{name} must be a finite number, got {value!r}")


def require_positive(value: float, name: str) -> None:
    """Raise unless ``value`` is a strictly positive finite number."""
    require_finite(value, name)
    if value <= 0.0:
        raise InputValidationError(f"{name} must be > 0, got {value!r}")


def require_non_negative(value: float, name: str) -> None:
    """Raise unless ``value`` is a finite number >= 0."""
    require_finite(value, name)
    if value < 0.0:
        raise InputValidationError(f"{name} must be >= 0, got {value!r}")


def require_positive_int(value: int, name: str) -> None:
    """Raise unless ``value`` is an integer >= 1."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise InputValidationError(f"{name} must be an integer >= 1, got {value!r}")


def require_in_open_interval(value: float, low: float, high: float, name: str) -> None:
    """Raise unless ``low < value < high`` (e.g. Poisson ratio in (-1, 0.5))."""
    require_finite(value, name)
    if not (low < value < high):
        raise InputValidationError(
            f"{name} must lie in the open interval ({low}, {high}), got {value!r}"
        )


def require_identifier(value: str, name: str) -> None:
    """Raise unless ``value`` is a non-empty, unpadded string."""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise InputValidationError(f"{name} must be a non-empty, unpadded string, got {value!r}")


def require_finite_array(array: NDArray[np.float64], name: str) -> None:
    """Raise unless every entry of ``array`` is finite."""
    if not np.all(np.isfinite(array)):
        raise InputValidationError(f"{name} contains non-finite entries")


def require_shape(
    array: NDArray[np.float64] | NDArray[np.int64], expected: tuple[int, ...], name: str
) -> None:
    """Raise unless ``array`` has the expected shape (-1 = any extent)."""
    shape = tuple(array.shape)
    if len(shape) != len(expected) or any(
        e not in (-1, s) for s, e in zip(shape, expected, strict=True)
    ):
        raise InputValidationError(f"{name} must have shape {expected}, got {shape}")
