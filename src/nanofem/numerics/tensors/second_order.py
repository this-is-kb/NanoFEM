"""Second-order tensor algebra over batched arrays (SDS Section 9, "Second order").

Every function accepts arbitrary leading batch axes and operates on the
trailing ``(d, d)`` (or ``(d,)`` for vectors) axes - a single tensor is a
batch of shape ``()``. No function here is specific to strain, stress, or any
other mechanics quantity; this module is pure linear algebra on square
matrices and is mechanics-free, matching the ``numerics`` leaf contract.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.tensors.errors import TensorError
from nanofem.utils.exceptions import InputValidationError


def _check_square(tensor: NDArray[np.float64], name: str = "tensor") -> int:
    if tensor.ndim < 2 or tensor.shape[-1] != tensor.shape[-2]:
        raise InputValidationError(f"{name} must have shape (..., d, d), got {tensor.shape}")
    return int(tensor.shape[-1])


def symmetric_part(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """The symmetric part ``1/2 (A + A^T)`` of a batch of square tensors."""
    _check_square(tensor)
    return np.asarray(0.5 * (tensor + np.swapaxes(tensor, -1, -2)), dtype=np.float64)


def skew_part(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """The skew-symmetric part ``1/2 (A - A^T)`` of a batch of square tensors."""
    _check_square(tensor)
    return np.asarray(0.5 * (tensor - np.swapaxes(tensor, -1, -2)), dtype=np.float64)


def trace(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """``tr(A)``, shape ``(...,)``."""
    _check_square(tensor)
    return np.asarray(np.trace(tensor, axis1=-2, axis2=-1), dtype=np.float64)


def determinant(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """``det(A)``, shape ``(...,)``."""
    _check_square(tensor)
    return np.asarray(np.linalg.det(tensor), dtype=np.float64)


def inverse(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """``A^-1``, shape ``(..., d, d)``.

    Raises :class:`~nanofem.numerics.tensors.errors.TensorError` naming the
    failure rather than letting a bare ``LinAlgError`` escape this layer, per
    the exception-tree policy that every NanoFEM module raises from its own
    tree.
    """
    _check_square(tensor)
    try:
        return np.asarray(np.linalg.inv(tensor), dtype=np.float64)
    except np.linalg.LinAlgError as exc:
        raise TensorError(f"tensor batch of shape {tensor.shape} is singular") from exc


def deviator(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """``dev(A) = A - (tr(A)/d) I``, shape ``(..., d, d)``."""
    dim = _check_square(tensor)
    identity = np.eye(dim, dtype=np.float64)
    volumetric = (trace(tensor) / dim)[..., np.newaxis, np.newaxis] * identity
    return np.asarray(tensor - volumetric, dtype=np.float64)


def frobenius_norm(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """``sqrt(A:A)``, shape ``(...,)``."""
    _check_square(tensor)
    return np.asarray(np.sqrt(np.einsum("...ij,...ij->...", tensor, tensor)), dtype=np.float64)


def outer(u: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
    """``u (x) v``, shape ``(..., d, d)`` from two vector batches of shape ``(..., d)``."""
    if u.shape != v.shape:
        raise InputValidationError(f"u and v must share a shape, got {u.shape} and {v.shape}")
    return np.asarray(np.einsum("...i,...j->...ij", u, v), dtype=np.float64)


def symmetric_outer(u: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
    """``1/2 (u (x) v + v (x) u)``, shape ``(..., d, d)``.

    Not ``symmetric_part(outer(u, v) + outer(v, u))``: that sum is already
    symmetric, so ``symmetric_part`` (which averages a tensor with its own
    transpose) would return it unchanged rather than halved.
    """
    return np.asarray(0.5 * (outer(u, v) + outer(v, u)), dtype=np.float64)


def single_contraction(
    tensor: NDArray[np.float64], vector: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``A.v``, shape ``(..., d)`` from a tensor batch and a vector batch."""
    dim = _check_square(tensor)
    if vector.shape[-1] != dim:
        raise InputValidationError(
            f"vector's last axis must equal tensor's dimension {dim}, got {vector.shape}"
        )
    return np.asarray(np.einsum("...ij,...j->...i", tensor, vector), dtype=np.float64)


def double_contraction(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """``A:B``, shape ``(...,)`` - the work-conjugate pairing of two tensor batches."""
    _check_square(a)
    _check_square(b)
    if a.shape != b.shape:
        raise InputValidationError(f"a and b must share a shape, got {a.shape} and {b.shape}")
    return np.asarray(np.einsum("...ij,...ij->...", a, b), dtype=np.float64)


def is_symmetric(tensor: NDArray[np.float64], *, atol: float = 1e-9) -> NDArray[np.bool_]:
    """Whether each tensor in the batch satisfies ``A == A^T`` within ``atol``."""
    _check_square(tensor)
    difference = tensor - np.swapaxes(tensor, -1, -2)
    return np.asarray(np.all(np.abs(difference) <= atol, axis=(-2, -1)), dtype=np.bool_)
