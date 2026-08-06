"""Tensor invariants and spectra over batched arrays (SDS Section 9, "Invariants and spectra").

``principal_values``/``principal_directions`` assume a symmetric tensor batch
(as stress and strain are) and use ``eigvalsh``/``eigh``, which are exact and
stable for symmetric input rather than the general (and slower, complex-valued)
``eig``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.tensors.second_order import (
    determinant,
    deviator,
    double_contraction,
    trace,
)
from nanofem.utils.exceptions import InputValidationError


def _check_square(tensor: NDArray[np.float64]) -> None:
    if tensor.ndim < 2 or tensor.shape[-1] != tensor.shape[-2]:
        raise InputValidationError(f"tensor must have shape (..., d, d), got {tensor.shape}")


def first_invariant(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """``I1 = tr(A)``, shape ``(...,)``."""
    return trace(tensor)


def second_invariant(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """``I2 = 1/2 [(tr A)^2 - tr(A^2)]``, shape ``(...,)``."""
    _check_square(tensor)
    trace_a = trace(tensor)
    trace_a_squared = trace(np.einsum("...ij,...jk->...ik", tensor, tensor))
    return np.asarray(0.5 * (trace_a**2 - trace_a_squared), dtype=np.float64)


def third_invariant(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """``I3 = det(A)``, shape ``(...,)``."""
    return determinant(tensor)


def deviatoric_second_invariant(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """``J2 = 1/2 dev(A):dev(A)``, shape ``(...,)``."""
    deviatoric = deviator(tensor)
    return np.asarray(0.5 * double_contraction(deviatoric, deviatoric), dtype=np.float64)


def deviatoric_third_invariant(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """``J3 = det(dev(A))``, shape ``(...,)``."""
    return determinant(deviator(tensor))


def von_mises(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """``sqrt(3 J2)``, shape ``(...,)``."""
    j2 = deviatoric_second_invariant(tensor)
    return np.asarray(np.sqrt(3.0 * j2), dtype=np.float64)


def principal_values(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """Eigenvalues of a symmetric tensor batch, ascending, shape ``(..., d)``."""
    _check_square(tensor)
    return np.asarray(np.linalg.eigvalsh(tensor), dtype=np.float64)


def principal_directions(
    tensor: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eigenvalues (ascending) and eigenvectors (columns) of a symmetric tensor batch."""
    _check_square(tensor)
    values, vectors = np.linalg.eigh(tensor)
    return np.asarray(values, dtype=np.float64), np.asarray(vectors, dtype=np.float64)
