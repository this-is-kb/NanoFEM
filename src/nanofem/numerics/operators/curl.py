"""The curl operator ``u -> curl(u)`` (SDS Section 8).

Two genuinely different operators share this name: the 2-D scalar rotation
``theta = du_y/dx - du_x/dy`` (the un-halved quantity - couple-stress theory's
``theta = 1/2 curl(u)`` is a theory-level scaling applied downstream, not this
recipe's business) and the 3-D vector curl. There is no 1-D curl, and asking
for one is a caller error, not a silent zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.operators.base import OPERATOR_DERIVATIVE_ORDER, DiscreteOperator
from nanofem.numerics.operators.errors import (
    OperatorError,
    OperatorShapeError,
    UnsupportedDimensionError,
)
from nanofem.utils.exceptions import InputValidationError

#: eps[k, i, j]: the Levi-Civita symbol in 3-D, used to build the vector curl row.
_LEVI_CIVITA_3D: NDArray[np.float64] = np.zeros((3, 3, 3), dtype=np.float64)
_LEVI_CIVITA_3D[0, 1, 2] = _LEVI_CIVITA_3D[1, 2, 0] = _LEVI_CIVITA_3D[2, 0, 1] = 1.0
_LEVI_CIVITA_3D[0, 2, 1] = _LEVI_CIVITA_3D[2, 1, 0] = _LEVI_CIVITA_3D[1, 0, 2] = -1.0


def curl_matrix(physical_gradients: NDArray[np.float64]) -> NDArray[np.float64]:
    """The curl row(s): shape ``(n_qp, 1, n_fun, 2)`` in 2-D, ``(n_qp, 3, n_fun, 3)`` in 3-D.

    2-D: ``row[q, 0, a, 0] = -dN_a/dy``, ``row[q, 0, a, 1] = dN_a/dx``, so
    ``theta = sum_{a,i} row[q,0,a,i] u_{a,i} = du_y/dx - du_x/dy``.
    3-D: ``row[q, k, a, j] = sum_i eps_kij dN_a/dx_i`` - the standard
    ``(curl u)_k = eps_kij du_j/dx_i`` written as a linear map on nodal DOFs.
    """
    if physical_gradients.ndim != 3:
        raise OperatorShapeError(
            f"physical_gradients must have shape (n_qp, n_fun, dim), got {physical_gradients.shape}"
        )
    dim = physical_gradients.shape[-1]
    if dim == 2:
        n_qp, n_fun, _ = physical_gradients.shape
        row = np.zeros((n_qp, 1, n_fun, 2), dtype=np.float64)
        row[:, 0, :, 0] = -physical_gradients[:, :, 1]
        row[:, 0, :, 1] = physical_gradients[:, :, 0]
        return row
    if dim == 3:
        return np.asarray(
            np.einsum("kij,qai->qkaj", _LEVI_CIVITA_3D, physical_gradients), dtype=np.float64
        )
    raise UnsupportedDimensionError(f"curl is defined for dim in {{2, 3}}, got dim={dim}")


@dataclass(frozen=True)
class CurlOperator(DiscreteOperator):
    """``DiscreteOperator`` wrapper around :func:`curl_matrix`."""

    def name(self) -> str:
        return "curl"

    def required_derivative_order(self) -> int:
        return OPERATOR_DERIVATIVE_ORDER[self.name()]

    def evaluate(self, physical_gradients: NDArray[np.float64]) -> NDArray[np.float64]:
        return curl_matrix(physical_gradients)

    def verify(self) -> None:
        """The rigid rotation field ``u = (-y, x)`` has curl == 2 everywhere in 2-D."""
        # N1 = x (grad [1, 0]), N2 = y (grad [0, 1]); u_x = -N2, u_y = N1.
        gradients = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        dofs = np.array([[0.0, 1.0], [-1.0, 0.0]])
        row = self.evaluate(gradients)
        theta = np.einsum("qoai,ai->qo", row, dofs)
        if not np.isclose(float(theta[0, 0]), 2.0, atol=1e-12):
            raise OperatorError(
                f"curl of the rigid rotation field was {float(theta[0, 0]):.6g}, expected 2"
            )
        try:
            self.evaluate(np.zeros((1, 2, 1)))
        except UnsupportedDimensionError:
            pass
        else:
            raise OperatorError("curl_matrix accepted dim=1 without raising")

    def is_valid(self) -> bool:
        try:
            self.verify()
        except (OperatorError, InputValidationError):
            return False
        return True
