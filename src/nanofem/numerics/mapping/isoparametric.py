"""Isoparametric mapping: the geometry interpolated by its own shape functions (SDS 2.6).

``x(xi) = sum_a N_a(xi) x_a``, where ``N_a`` is a geometry basis and ``x_a`` are
the physical coordinates of its nodes. Everything the map needs follows by
differentiating that one line:

- ``J[i, d] = sum_a x_a[i] dN_a/dxi_d``
- ``K[i, a, b] = sum_a x_a[i] d2N_a/dxi_a dxi_b``

so the mapping is a *contraction of the shape function tables against the node
coordinates*, and this module writes no basis mathematics of its own. The basis
is a phase-4 ``ShapeFunctionFamily``, which already evaluates, differentiates,
verifies, and caches; the mapping consumes it.

The name
--------
"Isoparametric" means the geometry basis is the same one the field uses. Nothing
here requires that - the class takes whatever basis it is given, so a
sub- or super-parametric element (linear geometry under a quadratic field, the
usual choice for straight-sided high-order elements) is the same class with a
different argument. The field basis is not this layer's business.

Affineness is a property of the geometry, not the class
-------------------------------------------------------
A ``Q1`` quadrilateral is bilinear and generally *not* affine; it becomes affine
exactly when it is a parallelogram. So :pyattr:`is_affine` is answered by
inspecting the mapping Hessian rather than by the element's type, and the answer
unlocks the second-derivative shortcut the SDS 2.6 note describes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.interpolation.base import Interpolation
from nanofem.numerics.interpolation.shape_functions import ShapeFunctionFamily, shape_functions
from nanofem.numerics.interpolation.tabulation import PointsLike
from nanofem.numerics.mapping.base import GeometricMapping
from nanofem.numerics.mapping.enums import MappingType
from nanofem.numerics.mapping.errors import InverseMapError
from nanofem.numerics.reference.element import ReferenceElement
from nanofem.utils.exceptions import DegenerateCellError, InputValidationError

#: Newton settings for the physical-to-reference inversion.
_NEWTON_TOL: float = 1.0e-12
_NEWTON_MAX_ITERATIONS: int = 50

#: A mapping Hessian below this (relative to the element size) counts as affine.
_AFFINE_TOL: float = 1.0e-12


@dataclass(frozen=True, eq=False, repr=False)
class IsoparametricMapping(GeometricMapping):
    """A mapping whose geometry is interpolated by a nodal shape function basis.

    Parameters
    ----------
    geometry_basis
        The basis interpolating the geometry, or the interpolation to build one
        from. Must be nodal: an isoparametric map interpolates node
        *coordinates*, and a Hermite basis would demand nodal derivatives of the
        geometry, which are not node data.
    physical_nodes
        Physical coordinates of the basis's nodes, in the basis's own node
        order, shape ``(n_nodes, embedding_dim)``.
    """

    basis_: ShapeFunctionFamily
    nodes_: NDArray[np.float64]

    def __init__(
        self,
        geometry_basis: ShapeFunctionFamily | Interpolation,
        physical_nodes: NDArray[np.float64] | list[list[float]],
    ) -> None:
        """Bind a nodal geometry basis to its physical node coordinates."""
        basis = (
            geometry_basis
            if isinstance(geometry_basis, ShapeFunctionFamily)
            else shape_functions(geometry_basis)
        )
        if not basis.interpolation.is_nodal:
            raise InputValidationError(
                f"an isoparametric map interpolates node coordinates, so its geometry basis "
                f"must be nodal; {basis.family.value} attaches derivative dofs, which are "
                f"not node data"
            )
        nodes = np.asarray(physical_nodes, dtype=np.float64)
        if nodes.ndim != 2 or nodes.shape[0] != basis.num_functions:
            raise InputValidationError(
                f"physical_nodes must have shape ({basis.num_functions}, embedding_dim) for "
                f"{basis.family.value} order {basis.order} on "
                f"{basis.interpolation.cell_type.value}, got {nodes.shape}"
            )
        if not np.all(np.isfinite(nodes)):
            raise InputValidationError("physical_nodes contains non-finite entries")
        nodes = nodes.copy()
        nodes.setflags(write=False)
        object.__setattr__(self, "basis_", basis)
        object.__setattr__(self, "nodes_", nodes)

    # ---- declarations -------------------------------------------------------

    @property
    def mapping_type(self) -> MappingType:
        """An isoparametric map."""
        return MappingType.ISOPARAMETRIC

    @property
    def reference_element(self) -> ReferenceElement:
        """The reference domain, taken from the geometry basis."""
        return self.basis_.interpolation.reference_element

    @property
    def physical_nodes(self) -> NDArray[np.float64]:
        """The geometry nodes' physical coordinates, read-only."""
        return self.nodes_

    @property
    def geometry_basis(self) -> ShapeFunctionFamily:
        """The shape function family interpolating the geometry (phase 4)."""
        return self.basis_

    @property
    def geometry_order(self) -> int:
        """The polynomial order of the geometry basis."""
        return self.basis_.order

    @cached_property
    def is_affine(self) -> bool:
        """Whether the mapping Hessian vanishes, making the Jacobian constant.

        Asked of the geometry, not of the element type: a ``Q1`` quadrilateral
        is bilinear in general but affine when it is a parallelogram, and a
        ``P2`` triangle with mid-side nodes at the edge midpoints is affine
        despite its quadratic basis. Both are common enough that assuming from
        the type alone would be wrong in practice.
        """
        curvature = self.mapping_hessian(self.sample_points())
        scale = max(float(np.abs(self.nodes_).max()), 1.0)
        return bool(np.all(np.abs(curvature) <= _AFFINE_TOL * scale))

    # ---- the primitives -----------------------------------------------------

    def map(self, points: PointsLike) -> NDArray[np.float64]:
        """``x = sum_a N_a(xi) x_a``: the shape function table against the nodes."""
        array = self._as_reference(points)
        result: NDArray[np.float64] = self.basis_.evaluate(array) @ self.nodes_
        return result

    def jacobian(self, points: PointsLike) -> NDArray[np.float64]:
        """``J[i, d] = sum_a x_a[i] dN_a/dxi_d``: the gradient table against the nodes."""
        array = self._as_reference(points)

        def build() -> NDArray[np.float64]:
            gradients = self.basis_.gradient(array)
            result: NDArray[np.float64] = np.einsum("pad,ai->pid", gradients, self.nodes_)
            return result

        return self._cached("jacobian", array, build)

    def mapping_hessian(self, points: PointsLike) -> NDArray[np.float64]:
        """``K[i, a, b] = sum_n x_n[i] d2N_n/dxi_a dxi_b``: the curvature of the map."""
        array = self._as_reference(points)

        def build() -> NDArray[np.float64]:
            hessians = self.basis_.hessian(array)
            result: NDArray[np.float64] = np.einsum("pnab,ni->piab", hessians, self.nodes_)
            return result

        return self._cached("mapping_hessian", array, build)

    def inverse_map(self, points: PointsLike) -> NDArray[np.float64]:
        """``xi(x)`` by Newton iteration from the reference centroid.

        A general isoparametric map has no closed-form inverse - it is a
        polynomial system - so this solves ``x(xi) - x = 0`` with Newton's
        method, using the Jacobian the map already provides and its
        pseudo-inverse as the update operator. Quadratic convergence, and for an
        affine geometry it lands in one step because the residual is linear.

        Raises rather than returning a plausible wrong answer when the iteration
        does not converge, which is what happens for a point outside a strongly
        distorted element.
        """
        array = self._as_physical(points)
        result = np.repeat(self.reference_element.centroid.reshape(1, -1), array.shape[0], axis=0)
        scale = max(float(np.abs(array).max()), 1.0)
        for _ in range(_NEWTON_MAX_ITERATIONS):
            residual = self.map(result) - array
            if float(np.abs(residual).max()) <= _NEWTON_TOL * scale:
                return np.asarray(result, dtype=np.float64)
            # A diverging iterate can wander somewhere the map is degenerate. That is a
            # failure of *this solve*, not evidence about the element, so the degeneracy
            # is translated rather than propagated - otherwise a perfectly good element
            # gets blamed for the solver's excursion.
            try:
                inverse = self.inverse_jacobian(result)
            except DegenerateCellError as exc:
                raise InverseMapError(
                    f"{self._label()}: the Newton iterate left the element and reached a "
                    f"point where the map is degenerate, so the inverse could not be "
                    f"continued; the target point most likely has no preimage ({exc})"
                ) from None
            result = result - np.einsum("pai,pi->pa", inverse, residual)
            if not np.all(np.isfinite(result)):
                raise InverseMapError(
                    f"{self._label()}: the Newton iterate diverged to a non-finite value; "
                    f"the target point most likely has no preimage"
                )
        residual = self.map(result) - array
        worst = float(np.abs(residual).max())
        raise InverseMapError(
            f"{self._label()}: Newton did not converge after {_NEWTON_MAX_ITERATIONS} "
            f"iterations (worst residual {worst:.3e}); the point most likely has no "
            f"preimage - it may lie outside the element, or the map may be strongly "
            f"distorted"
        )

    # ---- value semantics ----------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IsoparametricMapping):
            return NotImplemented
        return self.basis_ == other.basis_ and np.array_equal(self.nodes_, other.nodes_)

    def __hash__(self) -> int:
        return hash((IsoparametricMapping, self.basis_, self.nodes_.tobytes()))
