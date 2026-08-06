"""The surface gradient operator ``u -> grad_s(u) = P grad(u)`` (SDS Section 8).

``P = I - n (x) n`` projects the differentiation *direction* onto the tangent
plane at a facet quadrature point; it does not mix vector components, so the
DOF-component axis is broadcast rather than contracted - the surface gradient
of component ``i`` uses the same projected direction row as every other
component. This is the building block Gurtin-Murdoch surface elasticity
(SDS Section 4.6) composes into a symmetrized surface strain; that
symmetrization is theory-specific and lands with ``physics/surface``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.operators.base import OPERATOR_DERIVATIVE_ORDER, DiscreteOperator
from nanofem.numerics.operators.errors import OperatorError, OperatorShapeError
from nanofem.numerics.operators.gradient import gradient_matrix
from nanofem.utils.exceptions import InputValidationError


def surface_projector(normals: NDArray[np.float64]) -> NDArray[np.float64]:
    """``P_q = I - n_q (x) n_q``, shape ``(n_qp, dim, dim)``, from unit outward normals."""
    if normals.ndim != 2:
        raise OperatorShapeError(f"normals must have shape (n_qp, dim), got {normals.shape}")
    dim = normals.shape[-1]
    identity = np.eye(dim, dtype=np.float64)
    return np.asarray(identity - np.einsum("qi,qj->qij", normals, normals), dtype=np.float64)


def surface_gradient_matrix(
    physical_gradients: NDArray[np.float64], normals: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``row[q, k, a, c] = sum_j P_q[k, j] dN_a/dx_j``, broadcast over the component axis ``c``.

    Contracting ``row`` against nodal DOFs ``u[a, c]`` (summing only over
    ``a``, for each fixed ``c``) reproduces ``(grad_s u)_{c,k}`` - the surface
    gradient does not mix components, so every component sees the same
    projected direction row.
    """
    if physical_gradients.ndim != 3:
        raise OperatorShapeError(
            f"physical_gradients must have shape (n_qp, n_fun, dim), got {physical_gradients.shape}"
        )
    n_qp, n_fun, dim = physical_gradients.shape
    projector = surface_projector(normals)
    if projector.shape[0] != n_qp:
        raise OperatorShapeError(
            f"normals must have {n_qp} rows to match physical_gradients, got {projector.shape[0]}"
        )
    gradient_rows = gradient_matrix(physical_gradients)  # (n_qp, dim, n_fun)
    projected = np.einsum("qkj,qja->qka", projector, gradient_rows)  # (n_qp, dim, n_fun)
    return np.asarray(np.repeat(projected[..., np.newaxis], dim, axis=-1), dtype=np.float64)


@dataclass(frozen=True)
class SurfaceGradientOperator(DiscreteOperator):
    """``DiscreteOperator`` wrapper around :func:`surface_gradient_matrix`."""

    def name(self) -> str:
        return "surface_gradient"

    def required_derivative_order(self) -> int:
        return OPERATOR_DERIVATIVE_ORDER[self.name()]

    def evaluate(
        self, physical_gradients: NDArray[np.float64], normals: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return surface_gradient_matrix(physical_gradients, normals)

    def verify(self) -> None:
        """``P`` is idempotent, symmetric, annihilates ``n``; the recipe reduces to ``P @ grad``."""
        normals = np.array([[1.0, 0.0]])
        projector = surface_projector(normals)[0]
        if not np.allclose(projector, projector.T):
            raise OperatorError("surface projector was not symmetric")
        if not np.allclose(projector @ projector, projector):
            raise OperatorError("surface projector was not idempotent")
        if not np.allclose(projector @ normals[0], 0.0, atol=1e-12):
            raise OperatorError("surface projector did not annihilate the normal")

        gradients = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        row = self.evaluate(gradients, normals)
        expected = np.einsum("kj,qja->qka", projector, gradient_matrix(gradients))
        if not np.allclose(row[:, :, :, 0], expected):
            raise OperatorError("surface_gradient_matrix did not reduce to P @ gradient_matrix")

    def is_valid(self) -> bool:
        try:
            self.verify()
        except (OperatorError, InputValidationError):
            return False
        return True
