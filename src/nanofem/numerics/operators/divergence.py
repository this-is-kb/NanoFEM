"""The divergence operator ``u -> div(u)`` (SDS Section 8).

``divergence_matrix`` is literally the physical gradients with an inserted
row axis: ``div(u) = sum_{a,i} dN_a/dx_i * u_{a,i}`` is a single scalar row
per quadrature point, contracted against the unflattened nodal DOF array
``u[a, i]`` the same way every operator in this package leaves its DOF axis
(SDS Section 8; the flattening convention itself is deferred to `elements/`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.operators.base import OPERATOR_DERIVATIVE_ORDER, DiscreteOperator
from nanofem.numerics.operators.errors import OperatorError, OperatorShapeError
from nanofem.utils.exceptions import InputValidationError


def divergence_matrix(physical_gradients: NDArray[np.float64]) -> NDArray[np.float64]:
    """``row[q, 0, a, i] = dN_a/dx_i``, shape ``(n_qp, 1, n_fun, dim)``."""
    if physical_gradients.ndim != 3:
        raise OperatorShapeError(
            f"physical_gradients must have shape (n_qp, n_fun, dim), got {physical_gradients.shape}"
        )
    return np.asarray(physical_gradients[:, np.newaxis, :, :], dtype=np.float64)


@dataclass(frozen=True)
class DivergenceOperator(DiscreteOperator):
    """``DiscreteOperator`` wrapper around :func:`divergence_matrix`."""

    def name(self) -> str:
        return "divergence"

    def required_derivative_order(self) -> int:
        return OPERATOR_DERIVATIVE_ORDER[self.name()]

    def evaluate(self, physical_gradients: NDArray[np.float64]) -> NDArray[np.float64]:
        return divergence_matrix(physical_gradients)

    def verify(self) -> None:
        """The field ``u = (x, -y)`` has divergence 0 everywhere - the trace-free check."""
        gradients = np.array(
            [[[1.0, 0.0], [0.0, 1.0]]]
        )  # grad N1 = [1, 0] (N1=x), grad N2 = [0, 1] (N2=y)
        dofs = np.array([[1.0, 0.0], [0.0, -1.0]])  # u_x = +N1 = x, u_y = -N2 = -y
        row = self.evaluate(gradients)
        if row.shape != (1, 1, 2, 2):
            raise OperatorError(
                f"divergence_matrix returned shape {row.shape}, expected (1, 1, 2, 2)"
            )
        divergence = np.einsum("qoai,ai->qo", row, dofs)
        if not np.isclose(float(divergence[0, 0]), 0.0, atol=1e-12):
            raise OperatorError("divergence of the trace-free field u=(x,-y) was not zero")

    def is_valid(self) -> bool:
        try:
            self.verify()
        except (OperatorError, InputValidationError):
            return False
        return True
