"""Array <-> JSON-compatible payload helpers for dtype/shape-preserving round trips.

Serialization strategy (phase 1): each data object implements ``to_dict()``
returning a JSON-compatible dict and a ``from_dict()`` classmethod inverting
it exactly. Arrays are encoded with dtype and shape so round trips preserve
SDS C-4 types. File I/O lives in ``nanofem.io`` (rule R3); objects only
produce and consume payloads.

Available for any ``to_dict``/``from_dict`` pair that needs this exact
{dtype, shape, data} array encoding; existing implementations (``Mesh``,
``Material``, ``FieldSpec``, ``DofHandler``, ...) currently each encode their
own arrays inline rather than calling these, since their array fields are
simple enough (plain float64/int64, no dtype ambiguity) not to need the
general form. Kept here as the one place SDS C-4's general contract is
implemented, for the first data object whose array fields do need it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from nanofem.utils.exceptions import InputValidationError


def encode_array(array: NDArray[np.float64] | NDArray[np.int64]) -> dict[str, Any]:
    """Encode an array as {dtype, shape, data} with row-major flat data."""
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data": array.ravel(order="C").tolist(),
    }


def decode_array(payload: dict[str, Any]) -> NDArray[Any]:
    """Invert :func:`encode_array`, restoring dtype and shape exactly."""
    try:
        dtype = np.dtype(payload["dtype"])
        shape = tuple(int(s) for s in payload["shape"])
        data = payload["data"]
    except (KeyError, TypeError) as exc:
        raise InputValidationError(f"malformed array payload: {exc!r}") from None
    return np.asarray(data, dtype=dtype).reshape(shape)
