"""The ``GeometricMapping`` abstract base class (SDS 2.6).

This layer transforms reference coordinates into physical ones and carries the
differential-geometric consequences: Jacobian, its inverse, the metric tensor,
covariant and contravariant bases, and the push-forward of derivatives. It
performs no numerical integration, forms no element matrices, and knows nothing
about materials.

Composition, not duplication
---------------------------
The mapping owns neither geometry nor basis. It *consumes* them: the reference
domain and its topology come from ``numerics.reference`` (phase 2), and the
geometry basis of an isoparametric map is a ``ShapeFunctionFamily`` from
``numerics.interpolation`` (phase 4). Nothing here re-derives a shape function
or a facet incidence.

The physical element is not a class
-----------------------------------
A physical element *is* a reference topology plus node coordinates, and this
module exposes exactly that pair. It does not introduce a ``PhysicalElement``
type, for two reasons: the mesh layer already owns that concept (``Cell``), and
``numerics`` sits below ``mesh`` in the dependency order, so importing it is
forbidden and re-creating it would be duplication by another name.

Square and non-square Jacobians
-------------------------------
The Jacobian is ``(embedding_dimension, topological_dimension)``. When those are
equal it is square and the familiar formulas apply. When the element is
*embedded* - a bar in a plane, a shell in space, both of which the roadmap needs
for ``Truss2D`` and ``Frame2D`` - the Jacobian is tall, and:

- there is no signed determinant, so orientation is undefined;
- the measure scaling is the Gram determinant ``sqrt(det(J^T J))``;
- the inverse is the Moore-Penrose pseudo-inverse ``J^+ = (J^T J)^-1 J^T``,
  which is exactly the contravariant basis, and reduces to ``J^-1`` when square.

Writing every formula through the pseudo-inverse keeps one code path for both
cases, which is why embedded elements need no architectural change later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.interpolation.tabulation import PointsLike, as_points
from nanofem.numerics.mapping.enums import MappingType
from nanofem.numerics.mapping.errors import EmbeddedMappingError, MappingError
from nanofem.numerics.reference.element import ReferenceElement
from nanofem.utils.exceptions import DegenerateCellError, InputValidationError

#: Tolerance for the algebraic verifications (unit reference scale).
_ALGEBRAIC_TOL: float = 1.0e-9

#: Step and tolerance for the finite-difference Jacobian cross-check.
_FD_STEP: float = 1.0e-6
_FD_TOL: float = 1.0e-6

#: Degeneracy threshold on ``sigma_min / sigma_max``. Dimensionless on purpose:
#: an absolute tolerance on ``det J`` would call a legitimate nanometre-scale
#: element degenerate, which for a package aimed at MEMS/NEMS geometry is not a
#: hypothetical failure.
_DEGENERACY_TOL: float = 1.0e-12

#: Threshold below which a length is treated as vanishing, for chord-based ratios.
_LENGTH_TOL: float = 1.0e-30

#: Condition numbers above this earn a near-singular report.
_NEAR_SINGULAR_COND: float = 1.0e8


class GeometricMapping(ABC):
    """A map from a reference element to a physical element.

    Concrete mappings declare four things - :pyattr:`mapping_type`,
    :pyattr:`reference_element`, :pyattr:`physical_nodes`, and the three
    primitives :meth:`map`, :meth:`jacobian`, :meth:`mapping_hessian` - plus
    :meth:`inverse_map` and :pyattr:`is_affine`. Everything else is derived
    here and shared.
    """

    # ---- abstract declarations ---------------------------------------------

    @property
    @abstractmethod
    def mapping_type(self) -> MappingType:
        """The kind of map this is."""

    @property
    @abstractmethod
    def reference_element(self) -> ReferenceElement:
        """The reference domain the map starts from (phase 2)."""

    @property
    @abstractmethod
    def physical_nodes(self) -> NDArray[np.float64]:
        """Physical coordinates of the geometry nodes, ``(n_nodes, embedding_dim)``."""

    @property
    @abstractmethod
    def is_affine(self) -> bool:
        """Whether the Jacobian is constant over the element.

        The question the phase-0 seam poses: an affine map has a vanishing
        mapping Hessian, so second-derivative transformations lose their
        correction term and the inverse is closed-form.
        """

    @abstractmethod
    def map(self, points: PointsLike) -> NDArray[np.float64]:
        """Reference to physical: ``(n_points, topological_dim) -> (n_points, embedding_dim)``."""

    @abstractmethod
    def inverse_map(self, points: PointsLike) -> NDArray[np.float64]:
        """Physical to reference: ``(n_points, embedding_dim) -> (n_points, topological_dim)``."""

    @abstractmethod
    def jacobian(self, points: PointsLike) -> NDArray[np.float64]:
        """``J[p, i, d] = dx_i / dxi_d``, shape ``(n_points, embedding_dim, topological_dim)``."""

    @abstractmethod
    def mapping_hessian(self, points: PointsLike) -> NDArray[np.float64]:
        """``K[p, i, a, b] = d2x_i / dxi_a dxi_b``, shape ``(n_pts, emb, topo, topo)``.

        Identically zero for an affine map. It is the correction term that makes
        the physical Hessian of a field correct on a curved or bilinear element,
        and omitting it is the classic error the SDS 2.6 note warns about.
        """

    # ---- dimensions and coordinates ----------------------------------------

    @property
    def topological_dimension(self) -> int:
        """Intrinsic dimension of the element (the reference domain's dimension)."""
        return self.reference_element.topological_dimension

    @property
    def embedding_dimension(self) -> int:
        """Dimension of the physical space the element sits in."""
        return int(self.physical_nodes.shape[1])

    @property
    def is_embedded(self) -> bool:
        """Whether the physical space has more dimensions than the element itself."""
        return self.embedding_dimension > self.topological_dimension

    @property
    def reference_coordinates(self) -> NDArray[np.float64]:
        """Reference coordinates of the element's vertices (phase 2), read-only."""
        return self.reference_element.vertex_coordinates

    @property
    def physical_coordinates(self) -> NDArray[np.float64]:
        """Alias of :pyattr:`physical_nodes`: the physical element's node coordinates."""
        return self.physical_nodes

    @cached_property
    def physical_vertices(self) -> NDArray[np.float64]:
        """Images of the reference vertices, ``(n_vertices, embedding_dim)``, read-only.

        Computed *through the map* rather than sliced out of the node array, so
        it stays correct for any geometry basis whose node ordering differs.
        """
        vertices: NDArray[np.float64] = self.map(self.reference_element.vertex_coordinates)
        vertices.setflags(write=False)
        return vertices

    @cached_property
    def _cache(self) -> dict[tuple[Any, ...], NDArray[np.float64]]:
        """Per-instance cache of derived batches; see :meth:`cache_info`."""
        return {}

    def _cached(
        self,
        kind: str,
        points: NDArray[np.float64],
        build: Callable[[], NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Memoize a derived batch per (quantity, point set)."""
        key = (kind, points.tobytes(), points.shape)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        value: NDArray[np.float64] = build()
        value.setflags(write=False)
        self._cache[key] = value
        return value

    def _as_reference(self, points: PointsLike) -> NDArray[np.float64]:
        """Coerce and validate points in the reference domain."""
        return as_points(points, self.topological_dimension)

    def _as_physical(self, points: PointsLike) -> NDArray[np.float64]:
        """Coerce and validate points in physical space."""
        return as_points(points, self.embedding_dimension)

    # ---- Jacobian family ----------------------------------------------------

    def jacobian_transpose(self, points: PointsLike) -> NDArray[np.float64]:
        """``J^T``, shape ``(n_points, topological_dim, embedding_dim)``."""
        array = self._as_reference(points)
        return self._cached(
            "jacobian_transpose", array, lambda: np.swapaxes(self.jacobian(array), -1, -2).copy()
        )

    def inverse_jacobian(self, points: PointsLike) -> NDArray[np.float64]:
        """``J^+``, shape ``(n_points, topological_dim, embedding_dim)``.

        The Moore-Penrose pseudo-inverse ``(J^T J)^-1 J^T``, which equals
        ``J^-1`` exactly when the Jacobian is square. It is also the
        contravariant basis (see :meth:`contravariant_basis`), which is not a
        coincidence: both answer "how do I express a reference derivative in
        physical directions".
        """
        array = self._as_reference(points)

        def build() -> NDArray[np.float64]:
            jacobians = self.jacobian(array)
            self._require_non_degenerate(array, jacobians)
            # Computed by SVD rather than as (J^T J)^-1 J^T. The normal-equations form
            # is the definition, but forming J^T J squares the condition number, so a
            # merely awkward Jacobian yields a numerically singular metric and numpy
            # raises LinAlgError from inside what the caller thinks is a geometry
            # query. The pseudo-inverse is the same object without the squaring.
            result: NDArray[np.float64] = np.linalg.pinv(jacobians).astype(np.float64, copy=False)
            return result

        return self._cached("inverse_jacobian", array, build)

    def inverse_jacobian_transpose(self, points: PointsLike) -> NDArray[np.float64]:
        """``J^+T``, shape ``(n_points, embedding_dim, topological_dim)``.

        The operator that pushes a reference gradient forward to a physical one.
        """
        array = self._as_reference(points)
        return self._cached(
            "inverse_jacobian_transpose",
            array,
            lambda: np.swapaxes(self.inverse_jacobian(array), -1, -2).copy(),
        )

    def jacobian_determinant(self, points: PointsLike) -> NDArray[np.float64]:
        """Signed ``det J``, shape ``(n_points,)``. Square Jacobians only.

        The sign carries the orientation, which is why this is undefined for an
        embedded element: a tall Jacobian has no determinant, and a bar in a
        plane has no intrinsic handedness. Use :meth:`volume_scale` for the
        measure, which is defined in both cases.
        """
        if self.is_embedded:
            raise EmbeddedMappingError(
                f"{self._label()}: a signed determinant needs a square Jacobian, but this "
                f"element is {self.topological_dimension}-D embedded in "
                f"{self.embedding_dimension}-D; use volume_scale() for the measure scaling"
            )
        array = self._as_reference(points)
        return self._cached(
            "jacobian_determinant",
            array,
            lambda: np.linalg.det(self.jacobian(array)).astype(np.float64, copy=False),
        )

    def volume_scale(self, points: PointsLike) -> NDArray[np.float64]:
        """Local measure scaling ``sqrt(det(J^T J))``, shape ``(n_points,)``.

        The factor by which the map stretches length, area, or volume at a
        point - ``|det J|`` when square, the Gram determinant when embedded.
        This is differential geometry, not integration: nothing is summed here.
        """
        array = self._as_reference(points)
        return self._cached(
            "volume_scale",
            array,
            lambda: np.sqrt(np.linalg.det(self.metric_tensor(array))).astype(
                np.float64, copy=False
            ),
        )

    def jacobian_condition_number(self, points: PointsLike) -> NDArray[np.float64]:
        """``sigma_max / sigma_min`` of ``J``, shape ``(n_points,)``.

        The local anisotropy of the map: 1 for a similarity transform, large for
        a sliver. Distinct from :meth:`aspect_ratio`, which is a chord-based
        property of the element's shape rather than of the map at a point.
        """
        array = self._as_reference(points)
        singular = np.linalg.svd(self.jacobian(array), compute_uv=False)
        smallest = singular[..., -1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(smallest > 0.0, singular[..., 0] / smallest, np.inf)
        return np.asarray(ratio, dtype=np.float64)

    # ---- metric and bases ---------------------------------------------------

    def metric_tensor(self, points: PointsLike) -> NDArray[np.float64]:
        """``G = J^T J``, shape ``(n_points, topological_dim, topological_dim)``.

        The first fundamental form: ``G[a, b] = g_a . g_b``, the inner products
        of the covariant basis vectors. Symmetric and positive definite for a
        non-degenerate map, and square even when the Jacobian is not - which is
        what makes it the right object for embedded elements.
        """
        array = self._as_reference(points)

        def build() -> NDArray[np.float64]:
            jacobians = self.jacobian(array)
            result: NDArray[np.float64] = np.einsum("pia,pib->pab", jacobians, jacobians)
            return result

        return self._cached("metric_tensor", array, build)

    def inverse_metric_tensor(self, points: PointsLike) -> NDArray[np.float64]:
        """``G^-1``, shape ``(n_points, topological_dim, topological_dim)``.

        The contravariant metric, which raises indices: ``g^a = G^ab g_b``.
        """
        array = self._as_reference(points)
        return self._cached(
            "inverse_metric_tensor",
            array,
            lambda: np.linalg.inv(self.metric_tensor(array)).astype(np.float64, copy=False),
        )

    def covariant_basis(self, points: PointsLike) -> NDArray[np.float64]:
        """``g_a = dx / dxi_a``, shape ``(n_points, topological_dim, embedding_dim)``.

        The tangent vectors the reference axes map onto: row ``a`` is column
        ``a`` of ``J``. They are generally neither unit nor orthogonal, which is
        precisely why the metric tensor is needed.
        """
        return self.jacobian_transpose(points)

    def contravariant_basis(self, points: PointsLike) -> NDArray[np.float64]:
        """``g^a`` with ``g^a . g_b = delta^a_b``, shape ``(n_pts, topo_dim, emb_dim)``.

        The dual basis, ``g^a = G^ab g_b``. Working it out gives
        ``G^-1 J^T``, which *is* the pseudo-inverse of ``J`` - so this returns
        the same array as :meth:`inverse_jacobian`. The two names are kept
        because the two ideas are worth distinguishing, and a reader arriving
        from differential geometry should find the one they expect.
        """
        return self.inverse_jacobian(points)

    # ---- derivative transformations ----------------------------------------

    def physical_gradient(
        self, reference_gradient: NDArray[np.float64], points: PointsLike
    ) -> NDArray[np.float64]:
        """Push a reference gradient forward: ``(n_pts, n_fun, topo) -> (n_pts, n_fun, emb)``.

        The chain rule reads ``dN/dxi_a = sum_i (dN/dx_i)(dx_i/dxi_a)``, that is
        ``grad_xi N = J^T grad_x N``. Inverting gives
        ``grad_x N = J^+T grad_xi N``, which for a square Jacobian is the
        familiar ``J^-T grad_xi N``, and for an embedded element yields the
        *tangential* gradient - the only part that exists.

        Equivalently ``grad_x N = sum_a (dN/dxi_a) g^a``: the reference
        derivatives are the components of the gradient in the contravariant
        basis.
        """
        array = self._as_reference(points)
        gradients = np.asarray(reference_gradient, dtype=np.float64)
        if gradients.ndim != 3 or gradients.shape[0] != array.shape[0]:
            raise InputValidationError(
                f"reference_gradient must have shape (n_points, n_functions, "
                f"{self.topological_dimension}), got {gradients.shape} for "
                f"{array.shape[0]} points"
            )
        if gradients.shape[2] != self.topological_dimension:
            raise InputValidationError(
                f"reference_gradient's last axis is {gradients.shape[2]}, expected "
                f"{self.topological_dimension}"
            )
        inverse = self.inverse_jacobian(array)
        result: NDArray[np.float64] = np.einsum("pai,pna->pni", inverse, gradients)
        return result

    def reference_gradient(
        self, physical_gradient: NDArray[np.float64], points: PointsLike
    ) -> NDArray[np.float64]:
        """Pull a physical gradient back: ``(n_pts, n_fun, emb) -> (n_pts, n_fun, topo)``.

        The inverse of :meth:`physical_gradient`, and simply the chain rule read
        forwards: ``grad_xi N = J^T grad_x N``. No inverse is needed, which is
        why the pull-back is always well posed even where the push-forward is
        delicate.
        """
        array = self._as_reference(points)
        gradients = np.asarray(physical_gradient, dtype=np.float64)
        if gradients.ndim != 3 or gradients.shape[2] != self.embedding_dimension:
            raise InputValidationError(
                f"physical_gradient must have shape (n_points, n_functions, "
                f"{self.embedding_dimension}), got {gradients.shape}"
            )
        jacobians = self.jacobian(array)
        result: NDArray[np.float64] = np.einsum("pia,pni->pna", jacobians, gradients)
        return result

    def physical_hessian(
        self,
        reference_gradient: NDArray[np.float64],
        reference_hessian: NDArray[np.float64],
        points: PointsLike,
    ) -> NDArray[np.float64]:
        """Push a reference Hessian forward: ``(n_pts, n_fun, topo, topo) -> (..., emb, emb)``.

        Differentiating the chain rule a second time gives

        ``d2N/dxi_a dxi_b = sum_ij (d2N/dx_i dx_j) J[i,a] J[j,b]
        + sum_i (dN/dx_i) K[i,a,b]``

        where ``K`` is the mapping Hessian. Solving for the physical Hessian::

            H_x = J^+T (H_xi - grad_x N . K) J^+

        The subtracted term is what the SDS 2.6 note calls mandatory unless the
        map declares affineness: for an affine map ``K`` vanishes and the
        formula collapses to ``J^-T H_xi J^-1``, but on a bilinear or curved
        element dropping it is silently wrong - the error is invisible in the
        gradient and only shows up in a C1 theory.

        Restricted to square Jacobians. On an embedded element the second
        derivative of a field defined only on the manifold is not determined by
        the field alone: the second fundamental form (the manifold's curvature)
        enters, and pretending otherwise would return a plausible wrong answer.
        """
        if self.is_embedded:
            raise EmbeddedMappingError(
                f"{self._label()}: the physical Hessian of an embedded element is not "
                f"determined by the surface field alone - the second fundamental form "
                f"enters. Only square Jacobians are supported"
            )
        array = self._as_reference(points)
        hessians = np.asarray(reference_hessian, dtype=np.float64)
        expected = (array.shape[0], -1, self.topological_dimension, self.topological_dimension)
        if hessians.ndim != 4 or hessians.shape[0] != expected[0]:
            raise InputValidationError(
                f"reference_hessian must have shape (n_points, n_functions, "
                f"{self.topological_dimension}, {self.topological_dimension}), got "
                f"{hessians.shape}"
            )
        physical = self.physical_gradient(reference_gradient, array)
        correction = np.einsum("pni,piab->pnab", physical, self.mapping_hessian(array))
        corrected = hessians - correction
        inverse = self.inverse_jacobian(array)
        result: NDArray[np.float64] = np.einsum("pai,pnab,pbj->pnij", inverse, corrected, inverse)
        return result

    # ---- element geometry ---------------------------------------------------

    @cached_property
    def centroid(self) -> NDArray[np.float64]:
        """The image of the reference centroid, shape ``(embedding_dim,)``.

        Deliberately *not* the area centroid ``integral(x) / integral(1)``,
        which would need integration this layer must not perform. For an affine
        map the two coincide; for a curved one they do not, and the docstring
        says so rather than the name implying otherwise.
        """
        image: NDArray[np.float64] = self.map(self.reference_element.centroid.reshape(1, -1))[0]
        image.setflags(write=False)
        return image

    def bounding_box(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Axis-aligned box of the geometry nodes, as ``(lower, upper)``.

        Exact for a straight-sided element. For a curved one the nodes bound the
        control net rather than the image, which can be tighter than the true
        extent; this is the same caveat every node-based bound carries.
        """
        nodes = self.physical_nodes
        return (
            np.asarray(nodes.min(axis=0), dtype=np.float64),
            np.asarray(nodes.max(axis=0), dtype=np.float64),
        )

    def edge_lengths(self) -> NDArray[np.float64]:
        """Chord length of each reference edge's image, shape ``(n_edges,)``.

        Chords, not arc lengths: the true arc length of a curved edge needs
        integration. Exact whenever the geometry is straight-sided.
        """
        vertices = self.physical_vertices
        return np.asarray(
            [
                float(np.linalg.norm(vertices[b] - vertices[a]))
                for a, b in self.reference_element.edge_vertex_indices
            ],
            dtype=np.float64,
        )

    @property
    def diameter(self) -> float:
        """Largest distance between any two physical vertices."""
        vertices = self.physical_vertices
        differences = vertices[:, None, :] - vertices[None, :, :]
        return float(np.sqrt((differences**2).sum(axis=-1)).max())

    @property
    def characteristic_length(self) -> float:
        """A single representative size of the physical element (its diameter).

        Matches the convention ``ReferenceElement.characteristic_length`` uses,
        so the two are comparable.
        """
        return self.diameter

    @property
    def aspect_ratio(self) -> float:
        """Longest edge chord over shortest, ``>= 1``; ``inf`` for a collapsed edge."""
        lengths = self.edge_lengths()
        shortest = float(lengths.min())
        if shortest <= _LENGTH_TOL:
            return float("inf")
        return float(lengths.max()) / shortest

    @property
    def quality(self) -> float:
        """Minimum scaled Jacobian over the vertices, in ``(0, 1]`` when well shaped.

        ``q = volume_scale / prod_a |g_a|`` measures how far the local frame is
        from orthogonal with equal stretch: 1 when the map is a similarity of
        the reference element, falling toward 0 as the element skews or
        flattens, and negative for an inverted square-Jacobian element (where
        the signed determinant is used so the sign survives).

        Evaluated at the vertices because that is where a bilinear element's
        distortion is worst; for an affine map every point gives the same value.
        """
        points = self.reference_element.vertex_coordinates
        jacobians = self.jacobian(points)
        column_norms = np.linalg.norm(jacobians, axis=1)  # (n_pts, topo)
        denominator = np.prod(column_norms, axis=1)
        numerator = (
            self.jacobian_determinant(points) if not self.is_embedded else self.volume_scale(points)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = np.where(denominator > _LENGTH_TOL, numerator / denominator, 0.0)
        return float(np.min(scaled))

    def physical_measure(self) -> float:
        """Length, area, or volume of the physical element.

        Implemented for affine maps, where the measure is exactly
        ``reference_measure * volume_scale`` because the scaling is constant.
        For a non-affine map the measure is ``integral(volume_scale)`` over the
        reference domain, which is integration and therefore belongs to the
        quadrature layer, not this one.
        """
        if not self.is_affine:
            raise NotImplementedError(
                f"{self._label()}: the measure of a non-affine element is an integral of "
                f"the volume scaling, which arrives with the quadrature layer"
            )
        centre = self.reference_element.centroid.reshape(1, -1)
        return float(self.reference_element.reference_measure * self.volume_scale(centre)[0])

    def sample_points(self) -> NDArray[np.float64]:
        """A deterministic spread of reference points for validation and verification.

        Vertices, the centroid, and facet centroids - fixed rather than random,
        so a failure is always reproducible, and covering the places where a
        bilinear map's distortion is extreme.
        """
        element = self.reference_element
        candidates = np.vstack(
            [
                element.vertex_coordinates,
                element.centroid.reshape(1, -1),
                element.facet_centroids(),
            ]
        )
        return np.asarray(np.unique(np.round(candidates, 12), axis=0), dtype=np.float64)

    # ---- geometric validation ----------------------------------------------

    def is_valid(self) -> bool:
        """Return ``True`` if :meth:`validate` passes, ``False`` otherwise."""
        try:
            self.validate()
        except (MappingError, DegenerateCellError):
            return False
        return True

    def validate(self) -> None:
        """Check the physical element is a usable image of the reference one.

        Detects coincident nodes, degenerate and zero-measure maps, negative
        determinants (an inverted element, which is what bad node ordering looks
        like from here), and near-singular maps. Raises on the first failure.
        """
        self.validate_nodes()
        self.validate_jacobian()

    def validate_nodes(self) -> None:
        """Check the physical node set itself: shape, finiteness, distinctness."""
        nodes = self.physical_nodes
        if nodes.ndim != 2:
            raise MappingError(f"{self._label()}: physical_nodes must be 2-D, got {nodes.shape}")
        if not np.all(np.isfinite(nodes)):
            raise MappingError(f"{self._label()}: physical node coordinates are not all finite")
        if not 1 <= self.embedding_dimension <= 3:
            raise MappingError(
                f"{self._label()}: embedding dimension {self.embedding_dimension} not in 1..3"
            )
        if self.embedding_dimension < self.topological_dimension:
            raise MappingError(
                f"{self._label()}: a {self.topological_dimension}-D element cannot live in "
                f"{self.embedding_dimension}-D space"
            )
        unique = np.unique(np.round(nodes, 12), axis=0)
        if unique.shape[0] != nodes.shape[0]:
            raise DegenerateCellError(
                f"{self._label()}: coincident physical nodes; the map collapses"
            )

    def validate_jacobian(self) -> None:
        """Check the Jacobian at every sample point: non-degenerate, oriented, conditioned."""
        points = self.sample_points()
        jacobians = self.jacobian(points)
        self._require_non_degenerate(points, jacobians)
        if not self.is_embedded:
            determinants = self.jacobian_determinant(points)
            negative = np.nonzero(determinants < 0.0)[0]
            if negative.size:
                index = int(negative[0])
                raise DegenerateCellError(
                    f"{self._label()}: det J = {determinants[index]:.6g} < 0 at reference "
                    f"point {points[index].tolist()}; the element is inverted, which usually "
                    f"means its nodes are ordered the wrong way round"
                )
        conditions = self.jacobian_condition_number(points)
        worst = float(np.max(conditions))
        if worst > _NEAR_SINGULAR_COND:
            raise DegenerateCellError(
                f"{self._label()}: the map is near singular (worst Jacobian condition number "
                f"{worst:.3e}); the element is a sliver"
            )

    def _require_non_degenerate(
        self, points: NDArray[np.float64], jacobians: NDArray[np.float64]
    ) -> None:
        """Raise if the map collapses anywhere in ``points``.

        Judged from the singular values of ``J``, deliberately, on two counts.

        *Scale.* The measure scaling has units of length to the topological
        dimension, so any absolute threshold on it encodes an assumed element
        size. A 1 nm element in SI units has a legitimate area scaling near
        1e-18, and an absolute tolerance would reject it as degenerate - in a
        package built for nanoscale mechanics. The ratio
        ``sigma_min / sigma_max`` is dimensionless and therefore invariant
        under the uniform scaling that distinguishes those cases.

        *Precision.* ``det(J^T J)`` squares the condition number, so a
        rank-deficient map leaves a determinant near machine epsilon and a
        measure scaling near its square root - about 1e-8, which no sensible
        threshold catches. The singular values lose nothing.
        """
        singular = np.linalg.svd(jacobians, compute_uv=False)
        largest = singular[..., 0]
        smallest = singular[..., -1]
        collapsed = np.nonzero(smallest <= _DEGENERACY_TOL * largest)[0]
        if collapsed.size:
            index = int(collapsed[0])
            raise DegenerateCellError(
                f"{self._label()}: the Jacobian is rank deficient at reference point "
                f"{points[index].tolist()} (smallest singular value "
                f"{smallest[index]:.3e} against largest {largest[index]:.3e}); the element "
                f"is degenerate - it has collapsed onto a lower-dimensional shape"
            )

    # ---- verification -------------------------------------------------------

    def verify(self) -> None:
        """Run every identity the map must satisfy; raise on the first failure."""
        self.verify_jacobian()
        self.verify_metric_tensor()
        self.verify_inverse_consistency()
        self.verify_gradient_transformation()
        self.verify_orientation()
        self.verify_measure_scaling()

    def verify_jacobian(self) -> None:
        """Check the Jacobian against a central finite difference of the map.

        The independent standard: finite differencing shares no code with the
        analytic Jacobian, so it catches the transposed index that every other
        identity would happily agree with.
        """
        points = self.sample_points()
        analytic = self.jacobian(points)
        for axis in range(self.topological_dimension):
            offset = np.zeros(self.topological_dimension)
            offset[axis] = _FD_STEP
            numeric = (self.map(points + offset) - self.map(points - offset)) / (2.0 * _FD_STEP)
            if not np.allclose(analytic[:, :, axis], numeric, atol=_FD_TOL):
                worst = float(np.abs(analytic[:, :, axis] - numeric).max())
                raise MappingError(
                    f"{self._label()}: dx/dxi_{axis} disagrees with a central finite "
                    f"difference of the map by up to {worst:.3e}"
                )

    def verify_metric_tensor(self) -> None:
        """Check the metric is ``J^T J``, symmetric, and positive definite."""
        points = self.sample_points()
        metric = self.metric_tensor(points)
        jacobians = self.jacobian(points)
        expected = np.einsum("pia,pib->pab", jacobians, jacobians)
        if not np.allclose(metric, expected, atol=_ALGEBRAIC_TOL):
            raise MappingError(f"{self._label()}: the metric tensor is not J^T J")
        if not np.allclose(metric, np.swapaxes(metric, -1, -2), atol=_ALGEBRAIC_TOL):
            raise MappingError(f"{self._label()}: the metric tensor is not symmetric")
        eigenvalues = np.linalg.eigvalsh(metric)
        if float(eigenvalues.min()) <= 0.0:
            raise MappingError(
                f"{self._label()}: the metric tensor is not positive definite "
                f"(smallest eigenvalue {float(eigenvalues.min()):.3e})"
            )
        contravariant = self.contravariant_basis(points)
        covariant = self.covariant_basis(points)
        duality = np.einsum("pai,pbi->pab", contravariant, covariant)
        identity = np.broadcast_to(np.eye(self.topological_dimension), duality.shape)
        if not np.allclose(duality, identity, atol=_ALGEBRAIC_TOL):
            raise MappingError(
                f"{self._label()}: the contravariant basis is not dual to the covariant one "
                f"(g^a . g_b != delta^a_b)"
            )

    def verify_inverse_consistency(self) -> None:
        """Check the reference-physical round trip returns where it started."""
        points = self.sample_points()
        recovered = self.inverse_map(self.map(points))
        if not np.allclose(recovered, points, atol=1.0e-8):
            worst = float(np.abs(recovered - points).max())
            raise MappingError(
                f"{self._label()}: inverse_map(map(xi)) differs from xi by up to {worst:.3e}"
            )

    def verify_gradient_transformation(self) -> None:
        """Check the push-forward and pull-back are mutually inverse, and consistent.

        Two checks. First, the chain rule itself: pulling a pushed gradient back
        must return it. Second, and independently of any basis, the map's own
        gradient: since ``x = sum_a N_a x_a``, the physical gradient of the
        geometry basis must satisfy ``sum_a x_a (x) grad_x N_a = P``, the
        identity for a square Jacobian and the tangent projector for an embedded
        element. That identity catches a transposed index, which is the classic
        bug at this seam and the one the algebra elsewhere cannot see.
        """
        points = self.sample_points()
        rng = np.random.default_rng(20260717)
        reference = rng.normal(size=(points.shape[0], 4, self.topological_dimension))
        physical = self.physical_gradient(reference, points)
        recovered = self.reference_gradient(physical, points)
        if not np.allclose(recovered, reference, atol=_ALGEBRAIC_TOL):
            worst = float(np.abs(recovered - reference).max())
            raise MappingError(
                f"{self._label()}: pulling back a pushed-forward gradient changed it by up "
                f"to {worst:.3e}"
            )
        jacobians = self.jacobian(points)
        inverse = self.inverse_jacobian(points)
        projector = np.einsum("pia,paj->pij", jacobians, inverse)
        if self.is_embedded:
            if not np.allclose(projector, np.swapaxes(projector, -1, -2), atol=_ALGEBRAIC_TOL):
                raise MappingError(f"{self._label()}: J J^+ is not a symmetric projector")
            twice = np.einsum("pij,pjk->pik", projector, projector)
            if not np.allclose(twice, projector, atol=_ALGEBRAIC_TOL):
                raise MappingError(f"{self._label()}: J J^+ is not idempotent")
        else:
            identity = np.broadcast_to(np.eye(self.embedding_dimension), projector.shape)
            if not np.allclose(projector, identity, atol=_ALGEBRAIC_TOL):
                raise MappingError(
                    f"{self._label()}: J J^-1 is not the identity, so the gradient "
                    f"push-forward is misindexed"
                )

    def verify_orientation(self) -> None:
        """Check a square-Jacobian map preserves orientation everywhere."""
        if self.is_embedded:
            return
        determinants = self.jacobian_determinant(self.sample_points())
        if float(determinants.min()) <= 0.0:
            raise MappingError(
                f"{self._label()}: orientation is not preserved (smallest det J = "
                f"{float(determinants.min()):.3e})"
            )

    def verify_measure_scaling(self) -> None:
        """Check length and area scaling against closed-form physical geometry.

        For an affine map the measure scaling is constant, so the physical
        measure must be ``reference_measure * volume_scale`` - and that can be
        checked against an *independent* closed form (a chord length in 1-D, the
        shoelace area in 2-D) with no integration anywhere. Skipped for
        non-affine maps, whose measure is genuinely an integral.
        """
        if not self.is_affine:
            return
        expected = self._closed_form_measure()
        if expected is None:
            return
        computed = self.physical_measure()
        if not np.isclose(computed, expected, rtol=1.0e-9, atol=_ALGEBRAIC_TOL):
            raise MappingError(
                f"{self._label()}: reference_measure * volume_scale = {computed:.9g} but the "
                f"closed-form physical measure is {expected:.9g}"
            )

    def _closed_form_measure(self) -> float | None:
        """The physical measure from elementary geometry, or ``None`` if unavailable."""
        vertices = self.physical_vertices
        if self.topological_dimension == 1:
            return float(np.linalg.norm(vertices[1] - vertices[0]))
        if self.topological_dimension == 2 and self.embedding_dimension == 2:
            x, y = vertices[:, 0], vertices[:, 1]
            shoelace = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
            return float(0.5 * abs(shoelace))
        if self.topological_dimension == 2 and self.embedding_dimension == 3:
            total = 0.0
            for index in range(1, len(vertices) - 1):
                first = vertices[index] - vertices[0]
                second = vertices[index + 1] - vertices[0]
                total += 0.5 * float(np.linalg.norm(np.cross(first, second)))
            return total
        return None

    # ---- utilities ----------------------------------------------------------

    def cache_info(self) -> dict[str, int]:
        """Size of the derived-quantity cache, for diagnostics."""
        return {"entries": len(self._cache)}

    def clear_cache(self) -> None:
        """Drop every cached batch."""
        self._cache.clear()

    def _label(self) -> str:
        """Short identifier used in error messages."""
        return (
            f"{self.mapping_type.value} map of a "
            f"{self.reference_element.cell_type.value} in "
            f"{self.embedding_dimension}-D"
        )

    def pretty(self) -> str:
        """A compact, human-readable summary of the mapping."""
        lower, upper = self.bounding_box()
        lines = [
            f"{type(self).__name__} [{self._label()}]",
            f"  dimensions       : topological={self.topological_dimension} "
            f"embedding={self.embedding_dimension} "
            f"{'(embedded)' if self.is_embedded else ''}",
            f"  affine           : {self.is_affine}",
            f"  centroid         : {np.array2string(self.centroid, precision=6)}",
            f"  bounding box     : {np.array2string(lower, precision=6)} .. "
            f"{np.array2string(upper, precision=6)}",
            f"  characteristic L : {self.characteristic_length:.6g}",
            f"  aspect ratio     : {self.aspect_ratio:.6g}",
            f"  quality          : {self.quality:.6g}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(cell={self.reference_element.cell_type.value!r}, "
            f"topological_dim={self.topological_dimension}, "
            f"embedding_dim={self.embedding_dimension}, affine={self.is_affine})"
        )
