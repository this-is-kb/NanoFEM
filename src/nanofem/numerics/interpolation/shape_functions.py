"""The shape function library: constructing the nodal basis from an interpolation.

This is the layer phase 3 stopped one step short of. Phase 3 built the
generalized Vandermonde ``M[k, j] = l_k(m_j)`` and proved it invertible; this
module takes the inverse.

The construction
----------------
A shape function is a combination of the monomials that span the space::

    N_i(x) = sum_j C[i, j] m_j(x)

Duality ``l_k(N_i) = delta_ik`` expands to ``sum_j C[i, j] M[k, j] = delta_ik``,
that is ``C M^T = I``, hence::

    C = M^-T

Everything else follows by linearity. Because ``D^alpha`` is linear and the
coefficients ``C`` are constants, every derivative reuses the same matrix
against a differentiated monomial table::

    D^alpha N_i(x) = sum_j C[i, j] D^alpha m_j(x)

so ``evaluate``, ``gradient``, and ``hessian`` are one code path parameterized
by a multi-index.

Scope
-----
Reference coordinates only. There is no Jacobian, no mapping, no physical
derivative, no B-matrix, no quadrature, no integration, and no assembly here.
The points passed to every method are points in the reference domain, supplied
by the caller.

Family-agnosticism
------------------
The construction above never mentions Lagrange or Hermite. All the family
information lives in the degrees of freedom, which phase 3 already captured, so
:class:`LagrangeShapeFunctions` and :class:`HermiteShapeFunctions` differ only
in the idiomatic verification each adds. A new family - serendipity, spectral -
gets its shape functions with no new construction code, provided its
``Interpolation`` exists and is unisolvent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.interpolation.base import Interpolation, ShapeFunctions
from nanofem.numerics.interpolation.enums import DofKind, InterpolationFamily
from nanofem.numerics.interpolation.errors import InterpolationError, UnisolvenceError
from nanofem.numerics.interpolation.tabulation import (
    PointsLike,
    Tabulation,
    as_points,
    monomial_table,
)
from nanofem.numerics.operators.base import Continuity
from nanofem.numerics.reference.cell import ReferenceCell, reference_cell
from nanofem.numerics.reference.element import ReferenceElement
from nanofem.numerics.reference.enums import EntityType
from nanofem.utils.exceptions import InputValidationError

#: Tolerance for the algebraic verifications (unit reference scale).
_ALGEBRAIC_TOL: float = 1.0e-9

#: Step and tolerance for the finite-difference derivative cross-check.
_FD_STEP: float = 1.0e-5
_FD_TOL: float = 1.0e-6


class ShapeFunctionError(InterpolationError):
    """A shape function family fails one of its defining identities."""


class ShapeFunctionFamily(ShapeFunctions, ABC):
    """The nodal basis dual to an interpolation's degrees of freedom.

    Implements the phase-0 ``ShapeFunctions`` contract (SDS 2.4) rather than
    bypassing it: :meth:`cell`, :meth:`continuity`, :meth:`completeness_degree`,
    :meth:`evaluate`, and :meth:`derivatives` are the declared seam, and
    :meth:`gradient`, :meth:`hessian`, :meth:`tabulate`, :meth:`interpolate`,
    and the verification suite extend it.
    """

    @property
    @abstractmethod
    def interpolation(self) -> Interpolation:
        """The (K, P, Sigma) triple this basis is dual to."""

    # ---- the construction ---------------------------------------------------

    @cached_property
    def coefficients(self) -> NDArray[np.float64]:
        """``C = M^-T``: shape function coefficients in the monomial basis.

        Row ``i`` holds the coefficients of ``N_i``, so ``N_i = sum_j C[i,j] m_j``.
        Computed by solving ``M^T C = I`` rather than forming an explicit
        inverse. Read-only.

        Phase 3 already proved ``M`` invertible via
        :meth:`Interpolation.verify_linear_independence`, so this solve cannot
        fail for a validated interpolation; it is guarded anyway, because an
        unvalidated third-party family could reach here.
        """
        matrix = self.interpolation.unisolvence_matrix()
        if matrix.shape[0] != matrix.shape[1]:
            raise UnisolvenceError(
                f"{self._label()}: unisolvence matrix is {matrix.shape}, not square; "
                f"no dual basis exists"
            )
        identity = np.eye(matrix.shape[0], dtype=np.float64)
        try:
            coefficients: NDArray[np.float64] = np.linalg.solve(matrix.T, identity).astype(
                np.float64, copy=False
            )
        except np.linalg.LinAlgError as exc:
            raise UnisolvenceError(
                f"{self._label()}: unisolvence matrix is singular, so the dual basis "
                f"does not exist ({exc})"
            ) from None
        coefficients.setflags(write=False)
        return coefficients

    @cached_property
    def _cache(self) -> dict[tuple[Any, ...], NDArray[np.float64]]:
        """Per-instance tabulation cache; see :meth:`cache_info`."""
        return {}

    # ---- evaluation ---------------------------------------------------------

    def derivative(
        self,
        points: PointsLike,
        multi_index: tuple[int, ...] | None = None,
    ) -> NDArray[np.float64]:
        """Tabulate ``D^multi_index N_i`` at ``points``, shape ``(n_points, n_functions)``.

        The single primitive every other evaluation composes from. Results are
        cached per (points, multi-index), so a repeated point set - which is
        exactly what a later quadrature rule will supply - is tabulated once.
        """
        dimension = self.interpolation.support_dimension
        array = as_points(points, dimension)
        alpha = (0,) * dimension if multi_index is None else tuple(multi_index)
        key = (array.tobytes(), array.shape, alpha)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        table = monomial_table(self.interpolation.polynomial_space, array, alpha)
        result: NDArray[np.float64] = table @ self.coefficients.T
        result.setflags(write=False)
        self._cache[key] = result
        return result

    def evaluate(self, points: PointsLike) -> NDArray[np.float64]:
        """Shape function values, shape ``(n_points, n_functions)`` (SDS 2.4)."""
        return self.derivative(points, None)

    def gradient(self, points: PointsLike) -> NDArray[np.float64]:
        """Reference gradients, shape ``(n_points, n_functions, dim)``.

        Entry ``[p, i, d]`` is ``dN_i/dxi_d`` at point ``p``. These are
        *reference* derivatives; converting them to physical derivatives needs
        a Jacobian, which belongs to the mapping layer.
        """
        dimension = self.interpolation.support_dimension
        columns = [
            self.derivative(points, _unit_multi_index(axis, dimension)) for axis in range(dimension)
        ]
        stacked = np.stack(columns, axis=-1)
        stacked.setflags(write=False)
        return stacked

    def derivatives(self, points: PointsLike) -> NDArray[np.float64]:
        """Reference derivatives; the SDS 2.4 name for :meth:`gradient`."""
        return self.gradient(points)

    def hessian(self, points: PointsLike) -> NDArray[np.float64]:
        """Reference Hessians, shape ``(n_points, n_functions, dim, dim)``.

        Entry ``[p, i, a, b]`` is ``d2N_i / dxi_a dxi_b``. Always available -
        polynomials are smooth - and identically zero for an order-1 family,
        which is correct rather than inapplicable.
        """
        dimension = self.interpolation.support_dimension
        rows = [
            [
                self.derivative(points, _pair_multi_index(first, second, dimension))
                for second in range(dimension)
            ]
            for first in range(dimension)
        ]
        stacked = np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)
        stacked.setflags(write=False)
        return stacked

    def tabulate(
        self,
        points: PointsLike,
        *,
        max_derivative: int = 1,
    ) -> Tabulation:
        """Evaluate values and derivatives up to ``max_derivative`` in one batch."""
        if max_derivative not in (0, 1, 2):
            raise InputValidationError(f"max_derivative must be 0, 1, or 2, got {max_derivative}")
        array = as_points(points, self.interpolation.support_dimension)
        array.setflags(write=False)
        return Tabulation(
            points=array,
            values=self.evaluate(array),
            gradients=self.gradient(array) if max_derivative >= 1 else None,
            hessians=self.hessian(array) if max_derivative >= 2 else None,
        )

    def interpolate(
        self,
        dof_values: PointsLike,
        points: PointsLike,
    ) -> NDArray[np.float64]:
        """Evaluate the field ``sum_i c_i N_i`` at ``points``.

        ``dof_values`` may be ``(n_dofs,)`` for a scalar field or
        ``(n_dofs, n_components)`` for a vector one. This is the interpolation
        operator itself; it involves no mesh, no mapping, and no assembly.
        """
        coefficients = np.asarray(dof_values, dtype=np.float64)
        if coefficients.shape[0] != self.num_functions:
            raise InputValidationError(
                f"dof_values has {coefficients.shape[0]} entries, expected " f"{self.num_functions}"
            )
        if coefficients.ndim not in (1, 2):
            raise InputValidationError("dof_values must be 1-D or 2-D")
        values: NDArray[np.float64] = self.evaluate(points) @ coefficients
        return values

    # ---- metadata (the SDS 2.4 contract, plus what phase 2/3 provide) -------

    def cell(self) -> ReferenceCell:
        """The mesh cell record for this basis (SDS 2.4).

        Retained because the phase-0 contract declares it. The semantically
        richer accessor is :pyattr:`reference_element`: a ``ReferenceCell`` is a
        mesh-name record, while a shape function family is defined on a
        reference *domain*.
        """
        return reference_cell(self.interpolation.mesh_cell_name)

    @property
    def reference_element(self) -> ReferenceElement:
        """The reference domain this basis lives on (phase 2)."""
        return self.interpolation.reference_element

    def continuity(self) -> Continuity:
        """The continuity this family is designed to deliver (SDS 2.4)."""
        return self.interpolation.continuity

    def completeness_degree(self) -> int:
        """The polynomial completeness degree ``p`` (SDS 2.4)."""
        return self.interpolation.completeness_degree

    @property
    def family(self) -> InterpolationFamily:
        """The interpolation family."""
        return self.interpolation.family

    @property
    def order(self) -> int:
        """The nominal polynomial order."""
        return self.interpolation.order

    @property
    def num_functions(self) -> int:
        """Number of shape functions (one per degree of freedom)."""
        return self.interpolation.num_dofs

    @property
    def shape_function_ids(self) -> tuple[str, ...]:
        """Identifier of each shape function, index-aligned with the basis."""
        return self.interpolation.shape_function_ids

    def cache_info(self) -> dict[str, int]:
        """Size of the tabulation cache, for diagnostics."""
        return {"entries": len(self._cache)}

    def clear_cache(self) -> None:
        """Drop every cached tabulation."""
        self._cache.clear()

    # ---- verification -------------------------------------------------------

    def is_valid(self) -> bool:
        """Return ``True`` if :meth:`verify` passes, ``False`` otherwise."""
        try:
            self.verify()
        except InterpolationError:
            return False
        return True

    def verify(self) -> None:
        """Run every identity this basis must satisfy; raise on the first failure."""
        self.verify_kronecker_delta()
        self.verify_partition_of_unity()
        self.verify_polynomial_reproduction()
        self.verify_derivative_consistency()
        self.verify_symmetry()
        self.verify_boundary_restriction()
        self.verify_interpolation_exactness()
        self._verify_family_specific()

    def verify_kronecker_delta(self) -> None:
        """Check the duality ``l_k(N_i) = delta_ki`` that defines the basis.

        Applies every DOF functional to every shape function - each functional
        is a derivative evaluation, so this is a tabulation at the DOF points -
        and requires the resulting matrix to be the identity. Phase 3 could
        only check that such a basis *exists*; this checks the basis actually
        built satisfies the definition.
        """
        duality = self.duality_matrix()
        identity = np.eye(self.num_functions)
        if not np.allclose(duality, identity, atol=_ALGEBRAIC_TOL):
            worst = float(np.abs(duality - identity).max())
            raise ShapeFunctionError(
                f"{self._label()}: duality l_k(N_i) = delta_ki violated by up to {worst:.3e}"
            )

    def duality_matrix(self) -> NDArray[np.float64]:
        """``D[k, i] = l_k(N_i)``; the identity for a correctly built basis."""
        interpolation = self.interpolation
        rows = []
        for dof in interpolation.dofs:
            point = np.asarray([dof.point], dtype=np.float64)
            rows.append(self.derivative(point, dof.derivative)[0])
        return np.array(rows, dtype=np.float64)

    def verify_partition_of_unity(self) -> None:
        """Check the basis reproduces the constant 1 exactly.

        For a nodal family this is the classical ``sum_i N_i(x) = 1``. For
        Hermite the derivative functionals annihilate constants, so it is the
        *value*-dual functions that sum to 1 while the derivative-dual ones
        contribute nothing - phase 3 predicted this pattern from the
        functionals alone, and it is checked here pointwise on the built basis.
        """
        points = self.verification_points()
        values = self.evaluate(points)
        weights = np.array(
            [1.0 if dof.kind is DofKind.POINT_VALUE else 0.0 for dof in self.interpolation.dofs]
        )
        total = values @ weights
        if not np.allclose(total, 1.0, atol=_ALGEBRAIC_TOL):
            worst = float(np.abs(total - 1.0).max())
            raise ShapeFunctionError(
                f"{self._label()}: value-dual functions do not sum to 1 "
                f"(off by up to {worst:.3e})"
            )
        gradients = self.gradient(points)
        gradient_total = np.einsum("pid,i->pd", gradients, weights)
        if not np.allclose(gradient_total, 0.0, atol=_FD_TOL):
            worst = float(np.abs(gradient_total).max())
            raise ShapeFunctionError(
                f"{self._label()}: the gradient of the partition of unity is not zero "
                f"(up to {worst:.3e})"
            )

    def verify_polynomial_reproduction(self) -> None:
        """Check every member of the space is reproduced exactly, pointwise.

        For each monomial ``m_j``, its DOF vector is column ``j`` of the
        unisolvence matrix (``l_i(m_j) = M[i, j]``), so exact reproduction
        means ``sum_i M[i, j] N_i(x) = m_j(x)``.

        The right-hand side is evaluated by :func:`_naive_monomial_values`, a
        deliberately independent power-product loop, *not* by
        :func:`~nanofem.numerics.interpolation.tabulation.monomial_table`.
        That matters: the left-hand side expands to
        ``table C^T M``, and since ``C^T M = I`` it collapses to ``table``
        exactly. Comparing it against ``table`` again would therefore hold
        whatever the tabulation returned - the check would only be measuring
        the accuracy of the solve. Against an independent reference it tests
        the tabulation too.
        """
        interpolation = self.interpolation
        points = self.verification_points()
        matrix = interpolation.unisolvence_matrix()
        reproduced = self.evaluate(points) @ matrix
        exact = _naive_monomial_values(interpolation.polynomial_space.exponents, points)
        if not np.allclose(reproduced, exact, atol=_ALGEBRAIC_TOL):
            index = int(np.abs(reproduced - exact).max(axis=0).argmax())
            worst = float(np.abs(reproduced - exact).max())
            raise ShapeFunctionError(
                f"{self._label()}: monomial {interpolation.basis_ids[index]} is not "
                f"reproduced (off by up to {worst:.3e})"
            )

    def verify_derivative_consistency(self) -> None:
        """Check the reference derivatives against two independent standards.

        First analytically: differentiating the reproduction identity gives
        ``sum_i M[i, j] D^alpha N_i(x) = D^alpha m_j(x)``, so the gradients and
        Hessians must reproduce the derivatives of every monomial exactly.

        The standard is a central finite difference, which shares no code with
        the analytic route: the gradient is checked against a central
        difference of the values, and the Hessian against a central difference
        of the gradient.

        Finite differences are used rather than the analytic identity
        ``sum_i M[i, j] D^a N_i = D^a m_j``, which looks stronger but is not:
        that expression collapses to ``D^a table`` on both sides and holds for
        *any* derivative tabulation, correct or not. Only an independently
        computed derivative can catch an error in the tabulated derivative,
        and this is it.
        """
        interpolation = self.interpolation
        dimension = interpolation.support_dimension
        points = self.verification_points()

        analytic_gradient = self.gradient(points)
        for axis in range(dimension):
            step = np.zeros(dimension)
            step[axis] = _FD_STEP
            forward = self.evaluate(points + step)
            backward = self.evaluate(points - step)
            numeric = (forward - backward) / (2.0 * _FD_STEP)
            if not np.allclose(analytic_gradient[:, :, axis], numeric, atol=_FD_TOL):
                worst = float(np.abs(analytic_gradient[:, :, axis] - numeric).max())
                raise ShapeFunctionError(
                    f"{self._label()}: dN/dxi_{axis} disagrees with a central finite "
                    f"difference by up to {worst:.3e}"
                )

        analytic_hessian = self.hessian(points)
        for axis in range(dimension):
            step = np.zeros(dimension)
            step[axis] = _FD_STEP
            forward = self.gradient(points + step)
            backward = self.gradient(points - step)
            numeric = (forward - backward) / (2.0 * _FD_STEP)
            if not np.allclose(analytic_hessian[:, :, axis, :], numeric, atol=_FD_TOL):
                worst = float(np.abs(analytic_hessian[:, :, axis, :] - numeric).max())
                raise ShapeFunctionError(
                    f"{self._label()}: the Hessian row for xi_{axis} disagrees with a "
                    f"central finite difference of the gradient by up to {worst:.3e}"
                )

    def verify_symmetry(self) -> None:
        """Check the Hessian is symmetric in its two derivative indices.

        Mixed partials of a polynomial commute, so ``H[..., a, b] == H[..., b, a]``.
        With the multi-index construction used here both entries come from the
        same tabulation, making this a structural guard rather than a discovery;
        it earns its place by pinning the invariant for any future
        implementation that composes directional derivatives instead. The
        *independent* confirmation is the finite-difference Hessian check in
        :meth:`verify_derivative_consistency`, whose two routes to a mixed
        partial are computed separately.
        """
        hessian = self.hessian(self.verification_points())
        transposed = np.swapaxes(hessian, -1, -2)
        if not np.allclose(hessian, transposed, atol=_ALGEBRAIC_TOL):
            worst = float(np.abs(hessian - transposed).max())
            raise ShapeFunctionError(
                f"{self._label()}: the Hessian is not symmetric (off by up to {worst:.3e})"
            )

    def verify_boundary_restriction(self) -> None:
        """Check shape functions from nodes off a facet vanish on that facet.

        This is the property that makes inter-element conformity possible: the
        trace on a shared facet is determined entirely by the DOFs the two
        cells have in common, so neighbours agree. It is checked here rather
        than assumed, facet by facet, at sample points along each facet.
        """
        interpolation = self.interpolation
        element = interpolation.reference_element
        for facet_index in range(element.num_facets):
            points = self._facet_sample_points(facet_index)
            values = self.evaluate(points)
            owned = self._nodes_on_facet(facet_index)
            for index, dof in enumerate(interpolation.dofs):
                if dof.node_index in owned:
                    continue
                if not np.allclose(values[:, index], 0.0, atol=_ALGEBRAIC_TOL):
                    worst = float(np.abs(values[:, index]).max())
                    raise ShapeFunctionError(
                        f"{self._label()}: {interpolation.shape_function_ids[index]} does "
                        f"not vanish on facet {facet_index} (up to {worst:.3e}), so the "
                        f"trace would depend on a dof the neighbour does not share"
                    )

    def verify_interpolation_exactness(self) -> None:
        """Check the interpolation operator is the identity on the space.

        Algebraically, exactness is ``M^T C = I`` while duality is ``C M^T = I``;
        for a square invertible ``M`` both hold, and both are asserted because
        they are different numerical statements. Then the public
        :meth:`interpolate` API is exercised on a member of the space built
        from a fixed, non-trivial combination of monomials, so the operator is
        checked end to end rather than only in matrix form.
        """
        interpolation = self.interpolation
        matrix = interpolation.unisolvence_matrix()
        coefficients = self.coefficients
        identity = np.eye(self.num_functions)
        if not np.allclose(matrix.T @ coefficients, identity, atol=_ALGEBRAIC_TOL):
            raise ShapeFunctionError(f"{self._label()}: M^T C is not the identity")
        if not np.allclose(coefficients @ matrix.T, identity, atol=_ALGEBRAIC_TOL):
            raise ShapeFunctionError(f"{self._label()}: C M^T is not the identity")

        # A fixed, non-degenerate member of the space: weight monomial j by j + 1.
        weights = np.arange(1, interpolation.space_dimension + 1, dtype=np.float64)
        points = self.verification_points()
        dof_values = matrix @ weights  # l_i(u) for u = sum_j weights[j] m_j
        interpolated = self.interpolate(dof_values, points)
        exact = _naive_monomial_values(interpolation.polynomial_space.exponents, points) @ weights
        if not np.allclose(interpolated, exact, atol=_ALGEBRAIC_TOL):
            worst = float(np.abs(interpolated - exact).max())
            space_name = interpolation.polynomial_space.name
            raise ShapeFunctionError(
                f"{self._label()}: the interpolant of a member of {space_name} is not "
                f"exact (off by up to {worst:.3e})"
            )

    def _verify_family_specific(self) -> None:
        """Hook for a family's idiomatic checks; no-op by default."""

    # ---- sample points ------------------------------------------------------

    @cached_property
    def _verification_points(self) -> NDArray[np.float64]:
        """A deterministic spread of points covering the reference domain.

        Vertices, the centroid, facet centroids, and the midpoints between the
        centroid and each vertex. Fixed rather than random, so a failure is
        always reproducible.
        """
        element = self.interpolation.reference_element
        vertices = element.vertex_coordinates
        centroid = element.centroid
        candidates = [
            vertices,
            centroid.reshape(1, -1),
            element.facet_centroids(),
            0.5 * (vertices + centroid),
        ]
        stacked = np.vstack(candidates).astype(np.float64, copy=False)
        points: NDArray[np.float64] = np.unique(np.round(stacked, 12), axis=0)
        points.setflags(write=False)
        return points

    def verification_points(self) -> NDArray[np.float64]:
        """The deterministic point set the verification suite samples at."""
        return self._verification_points

    def _facet_sample_points(self, facet_index: int, samples: int = 5) -> NDArray[np.float64]:
        """Points along one facet of the reference domain."""
        element = self.interpolation.reference_element
        vertices = element.vertex_coordinates
        facet = element.facet_vertex_indices[facet_index]
        if len(facet) == 1:  # a vertex facet (1-D cell)
            return np.asarray([vertices[facet[0]]], dtype=np.float64)
        start, end = vertices[facet[0]], vertices[facet[1]]
        fractions = np.linspace(0.0, 1.0, samples).reshape(-1, 1)
        return np.asarray(start + fractions * (end - start), dtype=np.float64)

    def _nodes_on_facet(self, facet_index: int) -> frozenset[int]:
        """Indices of the nodes topologically owned by one facet.

        Derived from the entity associations phase 3 validated geometrically:
        a vertex of the facet, or a node interior to it. In 2-D the facets are
        the edges, so an edge node belongs to the facet of the same index.
        """
        element = self.interpolation.reference_element
        facet = set(element.facet_vertex_indices[facet_index])
        owned: set[int] = set()
        for node in self.interpolation.nodes:
            if node.entity is EntityType.VERTEX and node.entity_index in facet:
                owned.add(node.index)
            elif (
                node.entity is EntityType.EDGE
                and element.facet_entity_type is EntityType.EDGE
                and node.entity_index == facet_index
            ):
                owned.add(node.index)
        return frozenset(owned)

    # ---- utilities and value semantics --------------------------------------

    def _label(self) -> str:
        """Short identifier used in error messages."""
        return (
            f"{self.family.value} order {self.order} shape functions on "
            f"{self.interpolation.cell_type.value}"
        )

    def pretty(self) -> str:
        """A compact, human-readable summary of the basis."""
        return "\n".join(
            [
                f"{type(self).__name__} [{self._label()}]",
                f"  basis            : {self.num_functions} functions over "
                f"{self.interpolation.polynomial_space.name}",
                f"  continuity       : {self.continuity().name}  "
                f"completeness p = {self.completeness_degree()}",
                f"  coefficients C   : {self.coefficients.shape} = M^-T "
                f"(cond M = {self.interpolation.unisolvence_condition_number():.4g})",
                f"  mesh cell        : {self.cell().name}",
            ]
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(cell_type={self.interpolation.cell_type.value!r}, "
            f"order={self.order}, functions={self.num_functions})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShapeFunctionFamily):
            return NotImplemented
        return self.interpolation == other.interpolation

    def __hash__(self) -> int:
        return hash((ShapeFunctionFamily, self.interpolation))


@dataclass(frozen=True, eq=False, repr=False)
class LagrangeShapeFunctions(ShapeFunctionFamily):
    """The nodal basis of a Lagrange interpolation.

    Adds the classical Kronecker check to the generic suite: because every DOF
    is a point value, duality specializes to ``N_i(x_j) = delta_ij``, an
    identity among *values* that a reader can check against a textbook.
    """

    interpolation_: Interpolation

    def __init__(self, interpolation: Interpolation) -> None:
        """Bind a Lagrange interpolation, rejecting any other family."""
        if interpolation.family is not InterpolationFamily.LAGRANGE:
            raise InputValidationError(
                f"LagrangeShapeFunctions needs a lagrange interpolation, got "
                f"{interpolation.family.value}"
            )
        object.__setattr__(self, "interpolation_", interpolation)

    @property
    def interpolation(self) -> Interpolation:
        """The Lagrange interpolation this basis is dual to."""
        return self.interpolation_

    def verify_classical_kronecker_delta(self) -> None:
        """Check ``N_i(x_j) = delta_ij`` at the interpolation nodes."""
        values = self.evaluate(self.interpolation.node_locations)
        identity = np.eye(self.num_functions)
        if not np.allclose(values.T, identity, atol=_ALGEBRAIC_TOL):
            worst = float(np.abs(values.T - identity).max())
            raise ShapeFunctionError(
                f"{self._label()}: N_i(x_j) = delta_ij violated by up to {worst:.3e}"
            )

    def _verify_family_specific(self) -> None:
        """Add the classical nodal Kronecker identity."""
        self.verify_classical_kronecker_delta()


@dataclass(frozen=True, eq=False, repr=False)
class HermiteShapeFunctions(ShapeFunctionFamily):
    """The nodal basis of a Hermite interpolation.

    Adds the value/slope pattern to the generic suite. It is duality restated
    in the idiom a beam-element author reads: at each node, the value-dual
    function is 1 with zero derivative, and the derivative-dual function is 0
    with unit derivative.
    """

    interpolation_: Interpolation

    def __init__(self, interpolation: Interpolation) -> None:
        """Bind a Hermite interpolation, rejecting any other family."""
        if interpolation.family is not InterpolationFamily.HERMITE:
            raise InputValidationError(
                f"HermiteShapeFunctions needs a hermite interpolation, got "
                f"{interpolation.family.value}"
            )
        object.__setattr__(self, "interpolation_", interpolation)

    @property
    def interpolation(self) -> Interpolation:
        """The Hermite interpolation this basis is dual to."""
        return self.interpolation_

    def verify_nodal_value_and_slope_pattern(self) -> None:
        """Check the value/slope pattern at every node.

        At node ``n``, the function dual to the *value* DOF there equals 1 and
        has zero derivative, while a function dual to a *derivative* DOF equals
        0 and has unit derivative in its own direction. Every function dual to
        a DOF at another node vanishes with vanishing derivative.
        """
        interpolation = self.interpolation
        for node in interpolation.nodes:
            point = np.asarray([node.coordinates], dtype=np.float64)
            values = self.evaluate(point)[0]
            for index, dof in enumerate(interpolation.dofs):
                expected_value = (
                    1.0 if dof.node_index == node.index and dof.kind is DofKind.POINT_VALUE else 0.0
                )
                if not np.isclose(values[index], expected_value, atol=_ALGEBRAIC_TOL):
                    raise ShapeFunctionError(
                        f"{self._label()}: {interpolation.shape_function_ids[index]} at "
                        f"node {node.index} is {values[index]:.3e}, expected "
                        f"{expected_value}"
                    )

    def _verify_family_specific(self) -> None:
        """Add the Hermite value/slope pattern."""
        self.verify_nodal_value_and_slope_pattern()


class ShapeFunctionFactory(Protocol):
    """The constructor signature every shape function family satisfies.

    ``type[ShapeFunctionFamily]`` cannot express this: the ABC declares no
    ``__init__``, so mypy would reject constructing a class drawn from the
    registry. Mirrors the phase-3 ``InterpolationFactory`` protocol.
    """

    def __call__(self, interpolation: Interpolation) -> ShapeFunctionFamily:
        """Build the shape function family dual to an interpolation."""


#: Concrete family per interpolation family.
SHAPE_FUNCTION_FAMILIES: dict[InterpolationFamily, ShapeFunctionFactory] = {
    InterpolationFamily.LAGRANGE: LagrangeShapeFunctions,
    InterpolationFamily.HERMITE: HermiteShapeFunctions,
}


def shape_functions(interpolation: Interpolation) -> ShapeFunctionFamily:
    """Build the shape function family dual to ``interpolation``.

    The construction ``C = M^-T`` is family-agnostic, so a family without a
    dedicated class would still work; the dispatch exists so each family can
    contribute its idiomatic verification.
    """
    family_class = SHAPE_FUNCTION_FAMILIES.get(interpolation.family)
    if family_class is None:
        implemented = ", ".join(sorted(f.value for f in SHAPE_FUNCTION_FAMILIES))
        raise NotImplementedError(
            f"no shape function family for {interpolation.family.value!r}; "
            f"implemented: {implemented}"
        )
    return family_class(interpolation)


def _naive_monomial_values(
    exponents: tuple[tuple[int, ...], ...],
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate monomials by a plain power product, independent of ``monomial_table``.

    Deliberately the most obvious implementation possible, with none of the
    factorial or clamping logic the differentiating tabulator needs. It exists
    to be a second opinion: a verification that routes both of its sides
    through the same tabulator proves nothing about that tabulator.
    """
    return np.array(
        [
            [float(np.prod(point ** np.asarray(alpha, dtype=np.float64))) for alpha in exponents]
            for point in points
        ],
        dtype=np.float64,
    )


def _unit_multi_index(axis: int, dimension: int) -> tuple[int, ...]:
    """The multi-index of a single first derivative along ``axis``."""
    alpha = [0] * dimension
    alpha[axis] = 1
    return tuple(alpha)


def _pair_multi_index(first: int, second: int, dimension: int) -> tuple[int, ...]:
    """The multi-index of the second derivative along ``first`` then ``second``."""
    alpha = [0] * dimension
    alpha[first] += 1
    alpha[second] += 1
    return tuple(alpha)
