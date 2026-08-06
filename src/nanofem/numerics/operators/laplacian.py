"""The weak Laplacian pairing ``integral grad(N_a) . grad(N_b)`` (SDS Section 8).

Unlike the other recipes in this package, a weak-form pairing already sums
over quadrature points - it produces the element matrix directly, not a
per-point row, because it has nothing further to compose with (SDS Section 8
notes this blurs the operator/integrand line deliberately, keeping it here
because it is still a physics-free recipe).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.operators.base import OPERATOR_DERIVATIVE_ORDER, DiscreteOperator
from nanofem.numerics.operators.errors import OperatorError, OperatorShapeError
from nanofem.utils.exceptions import InputValidationError


def laplacian_matrix(
    physical_gradients: NDArray[np.float64],
    weights: NDArray[np.float64],
    volume_scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``K[a, b] = sum_q w_q J_q grad(N_a)(x_q) . grad(N_b)(x_q)``, shape ``(n_fun, n_fun)``."""
    if physical_gradients.ndim != 3:
        raise OperatorShapeError(
            f"physical_gradients must have shape (n_qp, n_fun, dim), got {physical_gradients.shape}"
        )
    n_qp = physical_gradients.shape[0]
    if weights.shape != (n_qp,) or volume_scale.shape != (n_qp,):
        raise OperatorShapeError(
            f"weights and volume_scale must have shape ({n_qp},), got {weights.shape} and "
            f"{volume_scale.shape}"
        )
    return np.asarray(
        np.einsum("q,q,qai,qbi->ab", weights, volume_scale, physical_gradients, physical_gradients),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class LaplacianOperator(DiscreteOperator):
    """``DiscreteOperator`` wrapper around :func:`laplacian_matrix`."""

    def name(self) -> str:
        return "laplacian"

    def required_derivative_order(self) -> int:
        return OPERATOR_DERIVATIVE_ORDER[self.name()]

    def evaluate(
        self,
        physical_gradients: NDArray[np.float64],
        weights: NDArray[np.float64],
        volume_scale: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return laplacian_matrix(physical_gradients, weights, volume_scale)

    def verify(self) -> None:
        """The unit reference triangle's constant-strain-triangle stiffness pattern.

        ``N1 = 1 - xi - eta``, ``N2 = xi``, ``N3 = eta`` have constant gradients
        ``[-1,-1]``, ``[1,0]``, ``[0,1]``; one quadrature point at the centroid
        with weight equal to the triangle's area (1/2) reproduces the classical
        textbook pattern exactly, since the integrand is constant.
        """
        gradients = np.array([[[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]])
        weights = np.array([1.0])
        volume_scale = np.array([0.5])
        matrix = self.evaluate(gradients, weights, volume_scale)
        expected = 0.5 * np.array([[2.0, -1.0, -1.0], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
        if not np.allclose(matrix, expected):
            raise OperatorError("laplacian_matrix did not match the CST closed form")
        if not np.allclose(matrix, matrix.T):
            raise OperatorError("laplacian_matrix was not symmetric")

    def is_valid(self) -> bool:
        try:
            self.verify()
        except (OperatorError, InputValidationError):
            return False
        return True
