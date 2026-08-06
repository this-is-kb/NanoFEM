"""Voigt/Mandel/full-tensor converters: the only representation bridge (SDS Section 9).

Strain vectors are *kinematic* Voigt (engineering shear, ``gamma_ij = 2 eps_ij``
for ``i != j``); stress vectors are *kinetic* Voigt (plain components) - SDS
C-1. Work conjugacy ``sigma^T eps_voigt = sigma:eps`` then holds without extra
factors. Mandel form uses a single, symmetric ``sqrt(2)`` weighting for both
stress and strain, which is exactly why spectral and norm computations are
done there rather than in Voigt form (SDS Section 9, "Voigt and Mandel").

Every Voigt<->Mandel conversion routes through the full tensor form
(``voigt_to_strain``/``voigt_to_stress`` then ``full_to_mandel``, and the
reverse), so there is exactly one place the kinematic/kinetic factor-of-2 is
applied and exactly one place the Mandel ``sqrt(2)`` is applied - the
"hand-inserted factors of two" bug this module exists to make structurally
impossible.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.tensors.conventions import VOIGT_LENGTH, VOIGT_ORDER
from nanofem.numerics.tensors.errors import RepresentationError
from nanofem.utils.exceptions import InputValidationError

_LENGTH_TO_DIM: dict[int, int] = {length: dim for dim, length in VOIGT_LENGTH.items()}

VoigtKind = Literal["strain", "stress"]


def _dim_of(tensor: NDArray[np.float64]) -> int:
    if tensor.ndim < 2 or tensor.shape[-1] != tensor.shape[-2]:
        raise InputValidationError(f"tensor must have shape (..., d, d), got {tensor.shape}")
    dim = int(tensor.shape[-1])
    if dim not in VOIGT_ORDER:
        raise RepresentationError(f"no Voigt convention is defined for dimension {dim}")
    return dim


def _dim_from_length(length: int) -> int:
    if length not in _LENGTH_TO_DIM:
        raise RepresentationError(f"{length} is not a valid Voigt vector length (1, 3, or 6)")
    return _LENGTH_TO_DIM[length]


def strain_to_voigt(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """Symmetric strain tensor batch -> kinematic Voigt (engineering shear)."""
    dim = _dim_of(tensor)
    order = VOIGT_ORDER[dim]
    components = [(tensor[..., i, j] if i == j else 2.0 * tensor[..., i, j]) for i, j in order]
    return np.asarray(np.stack(components, axis=-1), dtype=np.float64)


def stress_to_voigt(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """Symmetric stress tensor batch -> kinetic Voigt (plain components)."""
    dim = _dim_of(tensor)
    order = VOIGT_ORDER[dim]
    components = [tensor[..., i, j] for i, j in order]
    return np.asarray(np.stack(components, axis=-1), dtype=np.float64)


def voigt_to_strain(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Kinematic Voigt vector batch -> the full symmetric strain tensor."""
    dim = _dim_from_length(vector.shape[-1])
    order = VOIGT_ORDER[dim]
    tensor = np.zeros(vector.shape[:-1] + (dim, dim), dtype=np.float64)
    for idx, (i, j) in enumerate(order):
        value = vector[..., idx]
        if i == j:
            tensor[..., i, j] = value
        else:
            tensor[..., i, j] = 0.5 * value
            tensor[..., j, i] = 0.5 * value
    return tensor


def voigt_to_stress(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Kinetic Voigt vector batch -> the full symmetric stress tensor."""
    dim = _dim_from_length(vector.shape[-1])
    order = VOIGT_ORDER[dim]
    tensor = np.zeros(vector.shape[:-1] + (dim, dim), dtype=np.float64)
    for idx, (i, j) in enumerate(order):
        value = vector[..., idx]
        tensor[..., i, j] = value
        tensor[..., j, i] = value
    return tensor


def full_to_mandel(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """Symmetric tensor batch -> Mandel vector (orthonormal, ``sqrt(2)`` off-diagonal).

    Identical weighting for strain and stress - Mandel form has no
    kinematic/kinetic distinction, which is the entire point of using it for
    spectra and norms (SDS Section 9).
    """
    dim = _dim_of(tensor)
    order = VOIGT_ORDER[dim]
    root_two = np.sqrt(2.0)
    components = [(tensor[..., i, j] if i == j else root_two * tensor[..., i, j]) for i, j in order]
    return np.asarray(np.stack(components, axis=-1), dtype=np.float64)


def mandel_to_full(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mandel vector batch -> the full symmetric tensor."""
    dim = _dim_from_length(vector.shape[-1])
    order = VOIGT_ORDER[dim]
    root_two = np.sqrt(2.0)
    tensor = np.zeros(vector.shape[:-1] + (dim, dim), dtype=np.float64)
    for idx, (i, j) in enumerate(order):
        value = vector[..., idx]
        if i == j:
            tensor[..., i, j] = value
        else:
            tensor[..., i, j] = value / root_two
            tensor[..., j, i] = value / root_two
    return tensor


def voigt_to_mandel(vector: NDArray[np.float64], *, kind: VoigtKind) -> NDArray[np.float64]:
    """Voigt vector batch -> Mandel vector batch, via the full tensor (the only bridge)."""
    if kind == "strain":
        full = voigt_to_strain(vector)
    elif kind == "stress":
        full = voigt_to_stress(vector)
    else:
        raise InputValidationError(f"kind must be 'strain' or 'stress', got {kind!r}")
    return full_to_mandel(full)


def mandel_to_voigt(vector: NDArray[np.float64], *, kind: VoigtKind) -> NDArray[np.float64]:
    """Mandel vector batch -> Voigt vector batch, via the full tensor (the only bridge)."""
    full = mandel_to_full(vector)
    if kind == "strain":
        return strain_to_voigt(full)
    if kind == "stress":
        return stress_to_voigt(full)
    raise InputValidationError(f"kind must be 'strain' or 'stress', got {kind!r}")


def fourth_order_to_mandel(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """Major- and minor-symmetric fourth-order tensor batch -> Mandel matrix.

    ``M[a, b] = w_a w_b C[i(a), j(a), k(b), l(b)]``, with ``w = 1`` for a
    diagonal Voigt pair and ``w = sqrt(2)`` off-diagonal - the fourth-order
    analogue of :func:`full_to_mandel`, and the representation in which
    ``C`` is a plain symmetric matrix (SDS Section 9).
    """
    if (
        tensor.ndim < 4
        or len({tensor.shape[-4], tensor.shape[-3], tensor.shape[-2], tensor.shape[-1]}) != 1
    ):
        raise InputValidationError(
            f"tensor must have shape (..., d, d, d, d) with equal trailing axes, got {tensor.shape}"
        )
    dim = tensor.shape[-1]
    if dim not in VOIGT_ORDER:
        raise RepresentationError(f"no Voigt convention is defined for dimension {dim}")
    order = VOIGT_ORDER[dim]
    n_voigt = len(order)
    root_two = np.sqrt(2.0)
    weights = np.array([1.0 if i == j else root_two for i, j in order], dtype=np.float64)
    matrix = np.zeros(tensor.shape[:-4] + (n_voigt, n_voigt), dtype=np.float64)
    for a, (i, j) in enumerate(order):
        for b, (p, q) in enumerate(order):
            matrix[..., a, b] = weights[a] * weights[b] * tensor[..., i, j, p, q]
    return matrix


def mandel_to_fourth_order(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mandel matrix batch -> the full, minor-symmetric fourth-order tensor."""
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise InputValidationError(f"matrix must have shape (..., n, n), got {matrix.shape}")
    dim = _dim_from_length(matrix.shape[-1])
    order = VOIGT_ORDER[dim]
    root_two = np.sqrt(2.0)
    weights = np.array([1.0 if i == j else root_two for i, j in order], dtype=np.float64)
    tensor = np.zeros(matrix.shape[:-2] + (dim, dim, dim, dim), dtype=np.float64)
    for a, (i, j) in enumerate(order):
        for b, (p, q) in enumerate(order):
            value = matrix[..., a, b] / (weights[a] * weights[b])
            tensor[..., i, j, p, q] = value
            tensor[..., j, i, p, q] = value
            tensor[..., i, j, q, p] = value
            tensor[..., j, i, q, p] = value
    return tensor
