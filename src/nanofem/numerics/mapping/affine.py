"""Affine mappings: ``x = A xi + b``, and the identity as its trivial case.

An affine map has a constant Jacobian ``A`` and a vanishing mapping Hessian, so
its inverse is closed-form and second-derivative transformations lose their
correction term. Simplices with straight edges map affinely by construction; a
quadrilateral does so only when it is a parallelogram.

Deriving A and b, and why the fit is verified
---------------------------------------------
Requiring ``x(v_i) = X_i`` at every reference vertex gives a linear system in
the ``d*m + m`` unknowns of ``(A, b)``. For a ``d``-simplex the ``d + 1``
vertices make it exactly determined; for a quadrilateral, four corners impose
eight conditions on six unknowns, so a solution exists only if the corners are
consistent - which is precisely the parallelogram condition ``x2 = x1 + x3 - x0``.

The construction therefore solves in the least-squares sense and then *checks
that the fit is exact*. A non-parallelogram quadrilateral has a non-zero
residual and raises :class:`NonAffineError`, naming the isoparametric map as the
answer. Nothing about the parallelogram rule is hard-coded; it falls out.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.interpolation.tabulation import PointsLike
from nanofem.numerics.mapping.base import GeometricMapping
from nanofem.numerics.mapping.enums import MappingType
from nanofem.numerics.mapping.errors import InverseMapError, NonAffineError
from nanofem.numerics.reference.element import ReferenceElement
from nanofem.numerics.reference.enums import CellType
from nanofem.numerics.reference.registry import reference_element
from nanofem.utils.exceptions import InputValidationError

#: Residual above which an affine fit is judged inexact, relative to the element size.
_AFFINE_FIT_RTOL: float = 1.0e-10


@dataclass(frozen=True, eq=False, repr=False)
class AffineMapping(GeometricMapping):
    """The affine map ``x = A xi + b`` taking a reference element to a physical one.

    Parameters
    ----------
    reference
        The reference domain, or its cell type / name.
    physical_vertices
        Physical coordinates of the reference vertices, in reference vertex
        order, shape ``(n_vertices, embedding_dim)``. The embedding dimension
        may exceed the element's own (a bar in a plane, a triangle in space).
    """

    reference_: ReferenceElement
    vertices_: NDArray[np.float64]

    def __init__(
        self,
        reference: ReferenceElement | CellType | str,
        physical_vertices: NDArray[np.float64] | list[list[float]],
    ) -> None:
        """Derive ``(A, b)`` from the vertex correspondence and verify the fit is exact."""
        element = _resolve_reference(reference)
        vertices = np.asarray(physical_vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[0] != element.num_vertices:
            raise InputValidationError(
                f"physical_vertices must have shape ({element.num_vertices}, "
                f"embedding_dim) for a {element.cell_type.value}, got {vertices.shape}"
            )
        if not np.all(np.isfinite(vertices)):
            raise InputValidationError("physical_vertices contains non-finite entries")
        vertices = vertices.copy()
        vertices.setflags(write=False)
        object.__setattr__(self, "reference_", element)
        object.__setattr__(self, "vertices_", vertices)
        _ = self._affine_parameters  # fail fast: derive and verify the fit now

    @classmethod
    def from_linear_map(
        cls,
        reference: ReferenceElement | CellType | str,
        linear: NDArray[np.float64] | list[list[float]],
        translation: NDArray[np.float64] | list[float],
    ) -> AffineMapping:
        """Build from ``A`` and ``b`` directly, by mapping the reference vertices."""
        element = _resolve_reference(reference)
        matrix = np.asarray(linear, dtype=np.float64)
        offset = np.asarray(translation, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != element.topological_dimension:
            raise InputValidationError(
                f"linear must have shape (embedding_dim, {element.topological_dimension}), "
                f"got {matrix.shape}"
            )
        if offset.shape != (matrix.shape[0],):
            raise InputValidationError(
                f"translation must have shape ({matrix.shape[0]},), got {offset.shape}"
            )
        vertices = element.vertex_coordinates @ matrix.T + offset
        return cls(element, vertices)

    # ---- declarations -------------------------------------------------------

    @property
    def mapping_type(self) -> MappingType:
        """An affine map."""
        return MappingType.AFFINE

    @property
    def reference_element(self) -> ReferenceElement:
        """The reference domain."""
        return self.reference_

    @property
    def physical_nodes(self) -> NDArray[np.float64]:
        """The physical vertices; an affine map needs no other geometry nodes."""
        return self.vertices_

    @property
    def is_affine(self) -> bool:
        """Always ``True``: the Jacobian is ``A`` everywhere."""
        return True

    # ---- the affine parameters ----------------------------------------------

    @cached_property
    def _affine_parameters(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Solve ``[V | 1] [A^T; b^T] = X`` and require the fit to be exact."""
        element = self.reference_
        reference_vertices = element.vertex_coordinates
        augmented = np.hstack([reference_vertices, np.ones((reference_vertices.shape[0], 1))])
        solution, *_ = np.linalg.lstsq(augmented, self.vertices_, rcond=None)
        residual = augmented @ solution - self.vertices_
        scale = max(float(np.abs(self.vertices_).max()), 1.0)
        worst = float(np.abs(residual).max())
        if worst > _AFFINE_FIT_RTOL * scale:
            raise NonAffineError(
                f"a {element.cell_type.value} with these vertices is not affinely mappable "
                f"(the exact fit misses by {worst:.3e}). Four corners impose eight "
                f"conditions on six unknowns, so only a parallelogram admits an affine map; "
                f"use IsoparametricMapping for the general case"
            )
        linear = np.ascontiguousarray(solution[:-1].T)
        translation = np.ascontiguousarray(solution[-1])
        linear.setflags(write=False)
        translation.setflags(write=False)
        return linear, translation

    @property
    def linear(self) -> NDArray[np.float64]:
        """The constant matrix ``A``, shape ``(embedding_dim, topological_dim)``, read-only."""
        return self._affine_parameters[0]

    @property
    def translation(self) -> NDArray[np.float64]:
        """The constant offset ``b``, shape ``(embedding_dim,)``, read-only."""
        return self._affine_parameters[1]

    @cached_property
    def _pseudo_inverse(self) -> NDArray[np.float64]:
        """``A^+``, the constant inverse used by the closed-form inverse map."""
        inverse: NDArray[np.float64] = np.linalg.pinv(self.linear).astype(np.float64, copy=False)
        inverse.setflags(write=False)
        return inverse

    # ---- the primitives -----------------------------------------------------

    def map(self, points: PointsLike) -> NDArray[np.float64]:
        """``x = A xi + b``, exact by construction."""
        array = self._as_reference(points)
        result: NDArray[np.float64] = array @ self.linear.T + self.translation
        return result

    def inverse_map(self, points: PointsLike) -> NDArray[np.float64]:
        """``xi = A^+ (x - b)``: closed-form, no iteration.

        For a square ``A`` this is exact for any point in physical space. For an
        embedded element the pseudo-inverse returns the closest point on the
        element's affine subspace, so a point genuinely off that subspace is
        rejected rather than silently projected.
        """
        array = self._as_physical(points)
        offset = array - self.translation
        reference: NDArray[np.float64] = offset @ self._pseudo_inverse.T
        if self.is_embedded:
            residual = reference @ self.linear.T - offset
            scale = max(float(np.abs(array).max()), 1.0)
            worst = float(np.abs(residual).max())
            if worst > 1.0e-8 * scale:
                raise InverseMapError(
                    f"{self._label()}: the point lies {worst:.3e} off the element's affine "
                    f"subspace, so it has no reference preimage"
                )
        return reference

    def jacobian(self, points: PointsLike) -> NDArray[np.float64]:
        """``J = A`` at every point; broadcast to the requested batch and cached."""
        array = self._as_reference(points)
        return self._cached(
            "jacobian",
            array,
            lambda: np.broadcast_to(self.linear, (array.shape[0], *self.linear.shape)).copy(),
        )

    def mapping_hessian(self, points: PointsLike) -> NDArray[np.float64]:
        """Identically zero: an affine map has no curvature."""
        array = self._as_reference(points)
        return self._cached(
            "mapping_hessian",
            array,
            lambda: np.zeros(
                (
                    array.shape[0],
                    self.embedding_dimension,
                    self.topological_dimension,
                    self.topological_dimension,
                ),
                dtype=np.float64,
            ),
        )

    # ---- verification -------------------------------------------------------

    def verify_affine_exactness(self) -> None:
        """Check the map really is ``A xi + b`` and its Jacobian really is constant."""
        points = self.sample_points()
        expected = points @ self.linear.T + self.translation
        if not np.allclose(self.map(points), expected, atol=1.0e-12):
            raise NonAffineError(f"{self._label()}: map(xi) is not A xi + b")
        jacobians = self.jacobian(points)
        if not np.allclose(jacobians, self.linear, atol=1.0e-12):
            raise NonAffineError(f"{self._label()}: the Jacobian is not constant")
        if not np.allclose(self.mapping_hessian(points), 0.0):
            raise NonAffineError(f"{self._label()}: the mapping Hessian does not vanish")
        images = self.map(self.reference_element.vertex_coordinates)
        if not np.allclose(images, self.vertices_, atol=1.0e-10):
            raise NonAffineError(
                f"{self._label()}: the map does not reproduce its own physical vertices"
            )

    def verify(self) -> None:
        """Run the generic identities plus affine exactness."""
        super().verify()
        self.verify_affine_exactness()

    # ---- value semantics ----------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AffineMapping):
            return NotImplemented
        return self.reference_ == other.reference_ and np.array_equal(
            self.vertices_, other.vertices_
        )

    def __hash__(self) -> int:
        return hash((AffineMapping, self.reference_, self.vertices_.tobytes()))


@dataclass(frozen=True, eq=False, repr=False)
class IdentityMapping(AffineMapping):
    """The map that leaves the reference element where it is: ``A = I``, ``b = 0``.

    A genuine special case of :class:`AffineMapping` rather than a parallel
    implementation, so it inherits every identity and every check. It is what a
    computation performed directly in reference coordinates uses, and it is the
    fixed point every verification suite wants as a control: physical gradients
    equal reference gradients, the metric is the identity, and the measure
    scaling is one.
    """

    def __init__(self, reference: ReferenceElement | CellType | str) -> None:
        """Map a reference element onto itself."""
        element = _resolve_reference(reference)
        super().__init__(element, element.vertex_coordinates)

    @property
    def mapping_type(self) -> MappingType:
        """An identity map."""
        return MappingType.IDENTITY


def _resolve_reference(reference: ReferenceElement | CellType | str) -> ReferenceElement:
    """Accept a reference element, a cell type, or a cell-type name."""
    if isinstance(reference, ReferenceElement):
        return reference
    return reference_element(reference)
