"""Unit tests for small utils/ helpers: config, logging, scaling, serialize."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from nanofem.utils.config import Configuration
from nanofem.utils.exceptions import InputValidationError
from nanofem.utils.logging import get_logger
from nanofem.utils.scaling import Nondimensionalizer
from nanofem.utils.serialize import decode_array, encode_array


def test_configuration_defaults_and_override() -> None:
    default = Configuration()
    assert default.strict is False and default.debug_checks is False
    strict = Configuration(strict=True, debug_checks=True)
    assert strict.strict is True and strict.debug_checks is True


def test_get_logger_namespaces_under_nanofem() -> None:
    assert get_logger("core.model").name == "nanofem.core.model"
    assert get_logger("nanofem.core.model").name == "nanofem.core.model"
    assert isinstance(get_logger("x"), logging.Logger)


def test_nondimensionalizer_stores_scales_and_defers_conversion() -> None:
    scaler = Nondimensionalizer(length=0.001, force=1.0e-6)
    assert scaler.length == 0.001 and scaler.force == 1.0e-6
    with pytest.raises(InputValidationError):
        Nondimensionalizer(length=-1.0, force=1.0)
    with pytest.raises(NotImplementedError):
        scaler.to_model(1.0, "m")
    with pytest.raises(NotImplementedError):
        scaler.from_model(1.0, "m")


def test_encode_decode_array_round_trip() -> None:
    array = np.array([[1.0, 2.5], [3.0, 4.25]], dtype=np.float64)
    payload = encode_array(array)
    assert payload["dtype"] == "float64" and payload["shape"] == [2, 2]
    restored = decode_array(payload)
    np.testing.assert_array_equal(restored, array)
    assert restored.dtype == array.dtype

    ints = np.array([1, 2, 3], dtype=np.int64)
    restored_ints = decode_array(encode_array(ints))
    np.testing.assert_array_equal(restored_ints, ints)


def test_decode_array_rejects_malformed_payload() -> None:
    with pytest.raises(InputValidationError, match="malformed array payload"):
        decode_array({"dtype": "float64"})  # missing "shape"/"data"
