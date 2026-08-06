"""Tabulation of the prime (monomial) basis, and the tabulated batch value object.

Phase 3 described polynomial spaces structurally and refused to evaluate them at
arbitrary points; this module supplies that evaluation, because it is what shape
function construction needs. It tabulates *monomials* - the prime basis that
spans the space - not shape functions. The nodal basis is assembled from these
tables in :mod:`nanofem.numerics.interpolation.shape_functions`.

Everything here is differentiation and evaluation in *reference* coordinates.
No Jacobian, no mapping, no physical derivative, and no quadrature: the points
are supplied by the caller and are nothing but points.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.interpolation.polynomial import PolynomialSpace
from nanofem.utils.exceptions import InputValidationError

#: What every evaluation entry point accepts as points: an ``(n, dim)`` array, a
#: nested sequence of the same shape, or a single point as a flat sequence. The
#: alias exists because the flat form is part of the documented contract, and an
#: annotation that omitted it would be a lie the type checker enforces.
PointsLike: TypeAlias = NDArray[np.float64] | Sequence[Sequence[float]] | Sequence[float]


def as_points(points: PointsLike, dimension: int) -> NDArray[np.float64]:
    """Coerce ``points`` to a validated ``(n_points, dimension)`` float array.

    A single point may be given as a flat sequence of length ``dimension``; it
    is promoted to one row.
    """
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != dimension:
        raise InputValidationError(
            f"points must have shape (n_points, {dimension}), got {array.shape}"
        )
    if array.size == 0:
        raise InputValidationError("points must contain at least one point")
    if not np.all(np.isfinite(array)):
        raise InputValidationError("points contain non-finite entries")
    return array


def monomial_table(
    space: PolynomialSpace,
    points: NDArray[np.float64],
    derivative: tuple[int, ...] | None = None,
) -> NDArray[np.float64]:
    """Tabulate ``D^derivative m_j`` at each point, shape ``(n_points, n_monomials)``.

    Implements ``D^alpha x^beta = prod_d [beta_d! / (beta_d - alpha_d)!] *
    x_d^(beta_d - alpha_d)``, which vanishes whenever any ``beta_d < alpha_d``.
    With ``derivative`` omitted (or all zeros) this is plain evaluation of the
    monomials.

    This is the vectorized counterpart of
    :meth:`~nanofem.numerics.interpolation.dofs.DofFunctional.apply_to_monomial`,
    which does the same arithmetic one functional at a time to fill the
    unisolvence matrix. A test asserts the two agree.
    """
    dimension = space.n_variables
    alpha = (0,) * dimension if derivative is None else derivative
    if len(alpha) != dimension:
        raise InputValidationError(
            f"derivative multi-index has {len(alpha)} entries, expected {dimension}"
        )
    if any(order < 0 for order in alpha):
        raise InputValidationError(f"derivative multi-index {alpha} has a negative order")

    exponents = np.asarray(space.exponents, dtype=np.int64)  # (n_monomials, dimension)
    table = np.ones((points.shape[0], exponents.shape[0]), dtype=np.float64)
    for axis, order in enumerate(alpha):
        powers = exponents[:, axis]
        # perm(beta, alpha) is 0 when alpha > beta, which is the annihilation rule;
        # the residual power is then clamped so 0 ** negative never arises.
        factors = np.array([math.perm(int(power), order) for power in powers], dtype=np.float64)
        residual = np.maximum(powers - order, 0)
        table *= factors[None, :] * (points[:, axis : axis + 1] ** residual[None, :])
    return table


@dataclass(frozen=True)
class Tabulation:
    """A batch of shape function data evaluated at a fixed set of points.

    Arrays are read-only, so a cached batch can be handed to many consumers
    without defensive copying - the pattern SDS C-8 anticipates for sharing one
    tabulation across every element of a block.
    """

    points: NDArray[np.float64]
    values: NDArray[np.float64]
    gradients: NDArray[np.float64] | None = None
    hessians: NDArray[np.float64] | None = None

    @property
    def num_points(self) -> int:
        """Number of points in this batch."""
        return int(self.points.shape[0])

    @property
    def num_functions(self) -> int:
        """Number of shape functions tabulated."""
        return int(self.values.shape[1])

    @property
    def dimension(self) -> int:
        """Reference-space dimension of the points."""
        return int(self.points.shape[1])

    @property
    def max_derivative(self) -> int:
        """Highest derivative order present in this batch (0, 1, or 2)."""
        if self.hessians is not None:
            return 2
        if self.gradients is not None:
            return 1
        return 0

    def __repr__(self) -> str:
        return (
            f"Tabulation(points={self.num_points}, functions={self.num_functions}, "
            f"dim={self.dimension}, max_derivative={self.max_derivative})"
        )
