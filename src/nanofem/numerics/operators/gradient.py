"""The gradient operator ``s -> grad(s)`` (SDS Section 8).

A stateless recipe: given physical shape-function gradients already tabulated
at quadrature points (the ``GeometricMapping.physical_gradient`` output), form
the matrix mapping nodal DOFs to the pointwise gradient. Consumes only plain
``NDArray`` batches - never a ``ShapeFunctionFamily`` or ``GeometricMapping``
instance - so this module stays a leaf of ``numerics`` alongside
``interpolation``/``mapping``/``quadrature`` rather than importing them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.operators.base import OPERATOR_DERIVATIVE_ORDER, DiscreteOperator
from nanofem.numerics.operators.errors import OperatorError, OperatorShapeError
from nanofem.utils.exceptions import InputValidationError


def gradient_matrix(physical_gradients: NDArray[np.float64]) -> NDArray[np.float64]:
    """``G_q[d, a] = dN_a/dx_d``, shape ``(n_qp, dim, n_fun)``.

    ``physical_gradients`` has shape ``(n_qp, n_fun, dim)`` - the
    :meth:`GeometricMapping.physical_gradient` convention - and this recipe is
    exactly its transpose of the last two axes.
    """
    if physical_gradients.ndim != 3:
        raise OperatorShapeError(
            f"physical_gradients must have shape (n_qp, n_fun, dim), got {physical_gradients.shape}"
        )
    return np.asarray(np.swapaxes(physical_gradients, -1, -2), dtype=np.float64)


@dataclass(frozen=True)
class GradientOperator(DiscreteOperator):
    """``DiscreteOperator`` wrapper around :func:`gradient_matrix`."""

    def name(self) -> str:
        return "gradient"

    def required_derivative_order(self) -> int:
        return OPERATOR_DERIVATIVE_ORDER[self.name()]

    def evaluate(self, physical_gradients: NDArray[np.float64]) -> NDArray[np.float64]:
        return gradient_matrix(physical_gradients)

    def verify(self) -> None:
        """Self-check against a hand-built two-function, two-quadrature-point batch."""
        # Two linear functions on a 2-D domain with constant gradients
        # N1 = x  (grad = [1, 0]); N2 = y  (grad = [0, 1]); two identical points.
        gradients = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
        result = self.evaluate(gradients)
        expected = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
        if result.shape != (2, 2, 2):
            raise OperatorError(
                f"gradient_matrix returned shape {result.shape}, expected (2, 2, 2)"
            )
        if not np.allclose(result, expected):
            raise OperatorError("gradient_matrix did not reproduce the hand-built gradients")
        try:
            gradient_matrix(np.zeros((2, 2)))
        except OperatorShapeError:
            pass
        else:
            raise OperatorError("gradient_matrix accepted a rank-2 array without raising")

    def is_valid(self) -> bool:
        try:
            self.verify()
        except (OperatorError, InputValidationError):
            return False
        return True
