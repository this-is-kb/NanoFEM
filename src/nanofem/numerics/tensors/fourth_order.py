"""Fourth-order tensor algebra: identity, projectors, the isotropic oracle (SDS Section 9).

Fourth-order tensors are represented as full ``(..., d, d, d, d)`` arrays, with
``d`` the spatial dimension - no Voigt compaction here, that conversion is
``voigt.py``'s job. :func:`isotropic_stiffness` is a **verification oracle
only**: it is the textbook closed form ``C = d kappa J + 2 mu K`` (SDS Section 9
states this for ``d = 3`` as ``C = 3 kappa J + 2 mu K``; see the function's own
docstring for why the coefficient is ``d`` here) used to check
constitutive code elsewhere in the framework against an independent formula,
not a materials-layer constitutive law - it carries no knowledge of plane
stress/strain reductions or any other physical modeling choice, which is
`materials`/`physics` territory (SDS Section 9's own "future-facing" clause on
anisotropy is the reason this module stops here).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nanofem.utils.exceptions import InputValidationError


def _check_fourth_order(tensor: NDArray[np.float64], name: str = "tensor") -> int:
    if (
        tensor.ndim < 4
        or len({tensor.shape[-4], tensor.shape[-3], tensor.shape[-2], tensor.shape[-1]}) != 1
    ):
        raise InputValidationError(
            f"{name} must have shape (..., d, d, d, d) with equal trailing axes, "
            f"got {tensor.shape}"
        )
    return int(tensor.shape[-1])


def identity_fourth(dim: int) -> NDArray[np.float64]:
    """The fourth-order identity ``I_ijkl = delta_ik delta_jl``, shape ``(d, d, d, d)``."""
    if dim < 1:
        raise InputValidationError(f"dim must be >= 1, got {dim}")
    eye = np.eye(dim, dtype=np.float64)
    return np.asarray(np.einsum("ik,jl->ijkl", eye, eye), dtype=np.float64)


def symmetrizer(dim: int) -> NDArray[np.float64]:
    """The symmetrizer ``Ibar_ijkl = 1/2 (delta_ik delta_jl + delta_il delta_jk)``."""
    if dim < 1:
        raise InputValidationError(f"dim must be >= 1, got {dim}")
    eye = np.eye(dim, dtype=np.float64)
    first = np.einsum("ik,jl->ijkl", eye, eye)
    second = np.einsum("il,jk->ijkl", eye, eye)
    return np.asarray(0.5 * (first + second), dtype=np.float64)


def volumetric_projector(dim: int) -> NDArray[np.float64]:
    """The volumetric projector ``J = 1/d I2 (x) I2``, ``J_ijkl = delta_ij delta_kl / d``."""
    if dim < 1:
        raise InputValidationError(f"dim must be >= 1, got {dim}")
    eye = np.eye(dim, dtype=np.float64)
    return np.asarray(np.einsum("ij,kl->ijkl", eye, eye) / dim, dtype=np.float64)


def deviatoric_projector(dim: int) -> NDArray[np.float64]:
    """The deviatoric projector ``K = Ibar - J``."""
    return np.asarray(symmetrizer(dim) - volumetric_projector(dim), dtype=np.float64)


def isotropic_stiffness(bulk_modulus: float, shear_modulus: float, dim: int) -> NDArray[np.float64]:
    """The isotropic stiffness ``C = d kappa J + 2 mu K`` - a verification oracle only.

    SDS Section 9 states this for ``d = 3`` as ``C = 3 kappa J + 2 mu K``; the
    coefficient here is ``dim`` rather than the literal ``3`` so the formula
    stays consistent, for every ``d``, with the Lame-form equivalence the same
    SDS section states in the same breath: ``C:eps = lambda tr(eps) I + 2 mu
    eps`` with ``lambda = kappa - 2 mu / d``. Since ``volumetric_projector``
    already carries a ``1/d`` normalization, ``d kappa J`` is what makes the
    volumetric part of ``C:eps`` come out to exactly ``kappa tr(eps) I`` for
    any dimension, not only three.
    """
    if dim < 1:
        raise InputValidationError(f"dim must be >= 1, got {dim}")
    projector_j = volumetric_projector(dim)
    projector_k = deviatoric_projector(dim)
    return np.asarray(
        float(dim) * bulk_modulus * projector_j + 2.0 * shear_modulus * projector_k,
        dtype=np.float64,
    )


def has_major_symmetry(c: NDArray[np.float64], *, atol: float = 1e-9) -> NDArray[np.bool_]:
    """Whether ``C_ijkl == C_klij`` holds within ``atol``, for each tensor in the batch."""
    _check_fourth_order(c)
    transposed = np.swapaxes(np.swapaxes(c, -4, -2), -3, -1)
    difference = np.abs(c - transposed)
    return np.asarray(np.all(difference <= atol, axis=(-4, -3, -2, -1)), dtype=np.bool_)


def has_minor_symmetry(c: NDArray[np.float64], *, atol: float = 1e-9) -> NDArray[np.bool_]:
    """Whether ``C_ijkl == C_jikl == C_ijlk`` holds within ``atol``, per tensor in the batch."""
    _check_fourth_order(c)
    swap_ij = np.swapaxes(c, -4, -3)
    swap_kl = np.swapaxes(c, -2, -1)
    left = np.all(np.abs(c - swap_ij) <= atol, axis=(-4, -3, -2, -1))
    right = np.all(np.abs(c - swap_kl) <= atol, axis=(-4, -3, -2, -1))
    return np.asarray(left & right, dtype=np.bool_)


def apply(c: NDArray[np.float64], strain: NDArray[np.float64]) -> NDArray[np.float64]:
    """``sigma = C : eps``, shape ``(..., d, d)``.

    ``c`` may be unbatched (``(d, d, d, d)``) while ``strain`` carries batch
    axes, or both may share the same batch shape - ``einsum``'s ellipsis
    broadcasting handles either.
    """
    _check_fourth_order(c)
    if strain.ndim < 2 or strain.shape[-1] != strain.shape[-2]:
        raise InputValidationError(f"strain must have shape (..., d, d), got {strain.shape}")
    if strain.shape[-1] != c.shape[-1]:
        raise InputValidationError(
            f"strain's dimension {strain.shape[-1]} does not match c's dimension {c.shape[-1]}"
        )
    return np.asarray(np.einsum("...ijkl,...kl->...ij", c, strain), dtype=np.float64)
