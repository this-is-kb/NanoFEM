"""The symmetric gradient (B) operator ``u -> eps = grad_s(u)`` (SDS Section 8).

The strain-displacement operator every elasticity-family theory composes.
Voigt-ordered (kinematic convention, SDS C-1: engineering shear on the
off-diagonal rows) via :mod:`nanofem.numerics.tensors.conventions` - the same
``VOIGT_ORDER`` table the tensor layer uses, so the two never drift apart.
The DOF axis is left unflattened (``(n_qp, n_voigt, n_fun, dim)`` rather than
``(n_voigt, n_dof)``), deferring the node-major-vs-other flattening choice to
the not-yet-built ``elements/`` layer.
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
from nanofem.numerics.tensors.conventions import VOIGT_ORDER
from nanofem.numerics.tensors.voigt import voigt_to_strain
from nanofem.utils.exceptions import InputValidationError


def symmetric_gradient_matrix(physical_gradients: NDArray[np.float64]) -> NDArray[np.float64]:
    """``B[q, v, a, c]``: dof (node ``a``, component ``c``)'s contribution to strain row ``v``.

    For a diagonal Voigt pair ``(i, i)``: ``B[q, v, a, i] = dN_a/dx_i``.
    For an off-diagonal pair ``(i, j)``: ``B[q, v, a, i] = dN_a/dx_j`` and
    ``B[q, v, a, j] = dN_a/dx_i``, so contracting against nodal DOFs gives the
    engineering shear ``gamma_ij = du_i/dx_j + du_j/dx_i``.
    """
    if physical_gradients.ndim != 3:
        raise OperatorShapeError(
            f"physical_gradients must have shape (n_qp, n_fun, dim), got {physical_gradients.shape}"
        )
    dim = physical_gradients.shape[-1]
    if dim not in VOIGT_ORDER:
        raise UnsupportedDimensionError(f"no Voigt convention is defined for dimension {dim}")
    n_qp, n_fun, _ = physical_gradients.shape
    order = VOIGT_ORDER[dim]
    matrix = np.zeros((n_qp, len(order), n_fun, dim), dtype=np.float64)
    for v, (i, j) in enumerate(order):
        if i == j:
            matrix[:, v, :, i] = physical_gradients[:, :, i]
        else:
            matrix[:, v, :, i] += physical_gradients[:, :, j]
            matrix[:, v, :, j] += physical_gradients[:, :, i]
    return matrix


@dataclass(frozen=True)
class SymmetricGradientOperator(DiscreteOperator):
    """``DiscreteOperator`` wrapper around :func:`symmetric_gradient_matrix`."""

    def name(self) -> str:
        return "symmetric_gradient"

    def required_derivative_order(self) -> int:
        return OPERATOR_DERIVATIVE_ORDER[self.name()]

    def evaluate(self, physical_gradients: NDArray[np.float64]) -> NDArray[np.float64]:
        return symmetric_gradient_matrix(physical_gradients)

    def verify(self) -> None:
        """A prescribed uniform strain field is recovered exactly (the patch test)."""
        gradients = np.array([[[1.0, 0.0], [0.0, 1.0]]])  # N1 = x, N2 = y
        eps0 = np.array([[0.1, 0.02], [0.02, -0.05]])
        # u_i(x, y) = N1(x,y) u_{1,i} + N2(x,y) u_{2,i} = eps0_i0 x + eps0_i1 y
        dofs = np.stack([eps0[:, 0], eps0[:, 1]], axis=0)

        matrix = self.evaluate(gradients)
        strain_voigt = np.einsum("qvai,ai->qv", matrix, dofs)
        reconstructed = voigt_to_strain(strain_voigt)[0]
        if not np.allclose(reconstructed, eps0, atol=1e-12):
            raise OperatorError("symmetric_gradient_matrix did not recover the prescribed strain")

        zero_strain_voigt = np.einsum("qvai,ai->qv", matrix, np.zeros_like(dofs))
        if not np.allclose(zero_strain_voigt, 0.0):
            raise OperatorError("a zero displacement field did not produce zero strain")

    def is_valid(self) -> bool:
        try:
            self.verify()
        except (OperatorError, InputValidationError):
            return False
        return True
