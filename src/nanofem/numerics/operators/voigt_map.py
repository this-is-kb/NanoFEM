"""The Voigt operators: identity vector, deviatoric projector, trace row (SDS Section 8).

Constant matrices per spatial dimension, derivative order 0 - these compose
with the tensor layer's ``VOIGT_ORDER``/``VOIGT_LENGTH`` tables rather than
duplicating them, so a strain-gradient or thermoelastic theory can build
``ε_th = α ΔT m`` (SDS Section 8's own example) from a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.operators.base import OPERATOR_DERIVATIVE_ORDER, DiscreteOperator
from nanofem.numerics.operators.errors import OperatorError, UnsupportedDimensionError
from nanofem.numerics.tensors.conventions import VOIGT_ORDER
from nanofem.utils.exceptions import InputValidationError


def identity_vector(dim: int) -> NDArray[np.float64]:
    """``m``: 1 on diagonal Voigt positions, 0 off-diagonal.

    ``m . strain_voigt(eps) = tr(eps)``.
    """
    if dim not in VOIGT_ORDER:
        raise UnsupportedDimensionError(f"no Voigt convention is defined for dimension {dim}")
    return np.array([1.0 if i == j else 0.0 for i, j in VOIGT_ORDER[dim]], dtype=np.float64)


def deviatoric_projector_voigt(dim: int) -> NDArray[np.float64]:
    """``P_dev`` in Voigt form: ``P_dev @ strain_voigt(eps) == strain_voigt(dev(eps))``."""
    m = identity_vector(dim)
    return np.asarray(np.eye(m.size, dtype=np.float64) - np.outer(m, m) / dim, dtype=np.float64)


def trace_row_voigt(dim: int) -> NDArray[np.float64]:
    """``m`` as a ``(1, n_voigt)`` row.

    ``trace_row_voigt(dim) @ strain_voigt(eps) == [tr(eps)]``.
    """
    return identity_vector(dim)[np.newaxis, :]


@dataclass(frozen=True)
class VoigtMapOperator(DiscreteOperator):
    """``DiscreteOperator`` wrapper around the identity vector, deviatoric projector, trace row."""

    def name(self) -> str:
        return "voigt_map"

    def required_derivative_order(self) -> int:
        return OPERATOR_DERIVATIVE_ORDER[self.name()]

    def verify(self) -> None:
        """The identity vector recovers trace; the projector matches ``tensors.deviator``."""
        from nanofem.numerics.tensors.second_order import deviator, trace
        from nanofem.numerics.tensors.voigt import strain_to_voigt

        eps = np.array([[1.0, 0.3], [0.3, -0.5]])
        eps_voigt = strain_to_voigt(eps)

        m = identity_vector(2)
        if not np.isclose(float(m @ eps_voigt), float(trace(eps[np.newaxis])[0])):
            raise OperatorError("identity_vector did not recover the trace")
        if not np.allclose(trace_row_voigt(2)[0], m):
            raise OperatorError("trace_row_voigt did not match identity_vector")

        p_dev = deviatoric_projector_voigt(2)
        expected = strain_to_voigt(deviator(eps[np.newaxis])[0])
        if not np.allclose(p_dev @ eps_voigt, expected):
            raise OperatorError("deviatoric_projector_voigt did not match tensors.deviator")

    def is_valid(self) -> bool:
        try:
            self.verify()
        except (OperatorError, InputValidationError):
            return False
        return True
