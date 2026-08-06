"""``SO(d)`` rotation of vectors, tensors, and Voigt-form stiffnesses (SDS Section 9).

The Bond transformation matrices ``M_sigma``/``M_epsilon`` are not hand-derived
per dimension - a classic source of convention bugs - but built column by
column from the rotation's action on the Voigt basis tensors, reusing
:func:`~nanofem.numerics.tensors.voigt.voigt_to_stress` and
:func:`~nanofem.numerics.tensors.voigt.stress_to_voigt` as the single source of
truth for the Voigt convention. ``M_epsilon`` is then computed as
``inv(M_sigma).T`` rather than derived independently, so ``M_epsilon =
M_sigma^-T`` holds by construction rather than by two formulas that happen to
agree (or don't).

This module is distinct from ``numerics.math.rotations``, which holds a bare
2-D rotation-matrix constructor shared across all of ``numerics``; this module
is the tensor-algebra consumer of an already-built ``Q``, dimension-agnostic
throughout.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.tensors.conventions import VOIGT_ORDER
from nanofem.numerics.tensors.errors import NotRotationError
from nanofem.numerics.tensors.voigt import stress_to_voigt, voigt_to_stress
from nanofem.utils.exceptions import InputValidationError


def is_rotation(q: NDArray[np.float64], *, atol: float = 1e-9) -> bool:
    """Whether ``q`` is a proper rotation: ``Q^T Q = I`` and ``det(Q) = +1``."""
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise InputValidationError(f"q must have shape (d, d), got {q.shape}")
    dim = q.shape[0]
    orthogonal = np.allclose(q.T @ q, np.eye(dim, dtype=np.float64), atol=atol)
    proper = bool(abs(np.linalg.det(q) - 1.0) <= atol)
    return bool(orthogonal and proper)


def _require_rotation(q: NDArray[np.float64], *, atol: float = 1e-9) -> None:
    if not is_rotation(q, atol=atol):
        raise NotRotationError(f"q of shape {q.shape} is not a member of SO(d)")


def rotate_vector(q: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
    """``v' = Q v``, batched over ``v``'s leading axes."""
    _require_rotation(q)
    if v.shape[-1] != q.shape[0]:
        raise InputValidationError(
            f"v's last axis {v.shape[-1]} must match q's dimension {q.shape[0]}"
        )
    return np.asarray(np.einsum("ij,...j->...i", q, v), dtype=np.float64)


def rotate_second_order(q: NDArray[np.float64], tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """``A' = Q A Q^T``, batched over ``tensor``'s leading axes."""
    _require_rotation(q)
    if tensor.ndim < 2 or tensor.shape[-1] != tensor.shape[-2] or tensor.shape[-1] != q.shape[0]:
        raise InputValidationError(
            f"tensor must have shape (..., d, d) with d = {q.shape[0]}, got {tensor.shape}"
        )
    return np.asarray(np.einsum("ik,...kl,jl->...ij", q, tensor, q), dtype=np.float64)


def bond_matrix_stress(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """The stress Bond matrix ``M_sigma``: ``sigma'_voigt = M_sigma @ sigma_voigt``.

    Built column by column: column ``a`` is the kinetic-Voigt encoding of the
    rotated unit stress tensor for Voigt component ``a``.
    """
    _require_rotation(q)
    dim = q.shape[0]
    if dim not in VOIGT_ORDER:
        raise InputValidationError(f"no Voigt convention is defined for dimension {dim}")
    n_voigt = len(VOIGT_ORDER[dim])
    matrix = np.zeros((n_voigt, n_voigt), dtype=np.float64)
    for a in range(n_voigt):
        basis_voigt = np.zeros(n_voigt, dtype=np.float64)
        basis_voigt[a] = 1.0
        basis_full = voigt_to_stress(basis_voigt)
        rotated_full = rotate_second_order(q, basis_full)
        matrix[:, a] = stress_to_voigt(rotated_full)
    return matrix


def bond_matrix_strain(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """The strain Bond matrix ``M_epsilon = M_sigma^-T``, computed by inversion.

    Never re-derived independently from the strain (kinematic-Voigt) basis, so
    the identity ``M_epsilon = M_sigma^-T`` holds by construction rather than
    by two formulas that must be kept in sync.
    """
    m_sigma = bond_matrix_stress(q)
    return np.asarray(np.linalg.inv(m_sigma).T, dtype=np.float64)


def rotate_stiffness_voigt(
    d_voigt: NDArray[np.float64], q: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``D'_voigt = M_sigma @ D_voigt @ M_sigma^T`` (SDS Section 9, "Rotation")."""
    m_sigma = bond_matrix_stress(q)
    if d_voigt.shape != m_sigma.shape:
        raise InputValidationError(
            f"d_voigt shape {d_voigt.shape} does not match the Bond matrix shape {m_sigma.shape}"
        )
    return np.asarray(m_sigma @ d_voigt @ m_sigma.T, dtype=np.float64)
