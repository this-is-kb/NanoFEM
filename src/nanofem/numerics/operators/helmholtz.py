"""The Helmholtz weak pairing ``integral (N_a N_b + l^2 grad(N_a).grad(N_b))`` (SDS Section 8).

The building block of Eringen's differential (implicit-gradient) nonlocal
elasticity: the length scale ``l`` (``e0a`` in the SDS's notation) is passed
**per call**, never baked into the operator's constructor state - a
``DiscreteOperator`` recipe is contractually stateless and materials-blind,
and a length scale is a material/nonlocal parameter. That boundary is exactly
why ``kernels.Kernel`` objects, which *do* carry state, live in a separate,
higher layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.operators.base import OPERATOR_DERIVATIVE_ORDER, DiscreteOperator
from nanofem.numerics.operators.errors import OperatorError, OperatorShapeError
from nanofem.numerics.operators.laplacian import laplacian_matrix
from nanofem.utils.exceptions import InputValidationError


def mass_term(
    values: NDArray[np.float64], weights: NDArray[np.float64], volume_scale: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``M[a, b] = sum_q w_q J_q N_a(x_q) N_b(x_q)``, shape ``(n_fun, n_fun)``.

    Unweighted by density - a physical mass matrix multiplies this by rho.
    """
    if values.ndim != 2:
        raise OperatorShapeError(f"values must have shape (n_qp, n_fun), got {values.shape}")
    n_qp = values.shape[0]
    if weights.shape != (n_qp,) or volume_scale.shape != (n_qp,):
        raise OperatorShapeError(
            f"weights and volume_scale must have shape ({n_qp},), got {weights.shape} and "
            f"{volume_scale.shape}"
        )
    return np.asarray(
        np.einsum("q,q,qa,qb->ab", weights, volume_scale, values, values), dtype=np.float64
    )


def helmholtz_matrix(
    values: NDArray[np.float64],
    physical_gradients: NDArray[np.float64],
    weights: NDArray[np.float64],
    volume_scale: NDArray[np.float64],
    length_scale: float,
) -> NDArray[np.float64]:
    """``(I - l^2 laplacian)`` weak form: ``mass_term + l^2 * laplacian_matrix``."""
    if length_scale < 0.0:
        raise InputValidationError(f"length_scale must be >= 0, got {length_scale}")
    return np.asarray(
        mass_term(values, weights, volume_scale)
        + length_scale**2 * laplacian_matrix(physical_gradients, weights, volume_scale),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class HelmholtzOperator(DiscreteOperator):
    """``DiscreteOperator`` wrapper around :func:`helmholtz_matrix`."""

    def name(self) -> str:
        return "helmholtz"

    def required_derivative_order(self) -> int:
        return OPERATOR_DERIVATIVE_ORDER[self.name()]

    def evaluate(
        self,
        values: NDArray[np.float64],
        physical_gradients: NDArray[np.float64],
        weights: NDArray[np.float64],
        volume_scale: NDArray[np.float64],
        length_scale: float,
    ) -> NDArray[np.float64]:
        return helmholtz_matrix(values, physical_gradients, weights, volume_scale, length_scale)

    def verify(self) -> None:
        """``l=0`` collapses to the mass term; the remainder is ``l^2`` times the Laplacian."""
        rng = np.random.default_rng(11)
        values = rng.normal(size=(3, 2))
        gradients = rng.normal(size=(3, 2, 2))
        weights = np.array([0.3, 0.3, 0.4])
        volume_scale = np.array([1.0, 1.0, 1.0])

        zero_scale = self.evaluate(values, gradients, weights, volume_scale, 0.0)
        mass = mass_term(values, weights, volume_scale)
        if not np.allclose(zero_scale, mass):
            raise OperatorError("helmholtz_matrix at length_scale=0 did not equal the mass term")

        length_scale = 0.7
        full = self.evaluate(values, gradients, weights, volume_scale, length_scale)
        laplacian = laplacian_matrix(gradients, weights, volume_scale)
        if not np.allclose(full - mass, length_scale**2 * laplacian):
            raise OperatorError(
                "helmholtz_matrix's remainder above the mass term was not l^2 * laplacian"
            )

        try:
            self.evaluate(values, gradients, weights, volume_scale, -1.0)
        except InputValidationError:
            pass
        else:
            raise OperatorError("helmholtz_matrix accepted a negative length scale without raising")

    def is_valid(self) -> bool:
        try:
            self.verify()
        except (OperatorError, InputValidationError):
            return False
        return True
