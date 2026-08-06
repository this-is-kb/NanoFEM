"""The two base contracts of the interpolation package (SDS 2.4).

This module holds both sides of a deliberate phase boundary:

``Interpolation`` (this phase) describes a finite element in the classical
Ciarlet sense - the reference domain, the polynomial space, and the degrees
of freedom as functionals - together with everything derivable from that
triple *without* forming a basis: node locations and entity associations,
identifiers, continuity, flags, and the verification suite (completeness,
degree, unisolvence, constant reproduction, node ordering).

``ShapeFunctions`` (a later phase) is the evaluation contract: given the
above, produce the nodal basis dual to the DOFs and tabulate its values and
reference derivatives at points. Tabulated batches are cached per
(cell, family, degree, quadrature rule) and shared across all elements of a
block (SDS C-8).

The boundary between them has a precise algebraic statement. The generalized
Vandermonde ``M[k, j] = l_k(m_j)`` pairs each DOF functional with each
monomial. ``Interpolation`` *builds* ``M`` and proves it is invertible;
``M^-1`` is exactly the coefficient matrix of the shape functions in the
monomial basis. This phase never inverts it.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.interpolation.dofs import DofFunctional, InterpolationNode
from nanofem.numerics.interpolation.enums import DofKind, InterpolationFamily
from nanofem.numerics.interpolation.errors import (
    InterpolationError,
    NodeOrderingError,
    PolynomialSpaceError,
    UnisolvenceError,
)
from nanofem.numerics.interpolation.polynomial import PolynomialSpace
from nanofem.numerics.operators.base import Continuity
from nanofem.numerics.reference.cell import ReferenceCell
from nanofem.numerics.reference.element import ReferenceElement
from nanofem.numerics.reference.enums import CellType, EntityType
from nanofem.utils.exceptions import InputValidationError

#: Tolerance for geometric and functional consistency checks (unit reference scale).
_TOL: float = 1.0e-12

#: Cells that are simplices, and cells that are tensor products. The line is both.
_SIMPLEX_CELLS: frozenset[CellType] = frozenset(
    {CellType.LINE, CellType.TRIANGLE, CellType.TETRAHEDRON}
)
_TENSOR_PRODUCT_CELLS: frozenset[CellType] = frozenset(
    {CellType.LINE, CellType.QUADRILATERAL, CellType.HEXAHEDRON}
)

#: Stem of the conventional mesh cell-type name for each shape.
_MESH_NAME_STEM: dict[CellType, str] = {
    CellType.LINE: "line",
    CellType.TRIANGLE: "tri",
    CellType.QUADRILATERAL: "quad",
    CellType.TETRAHEDRON: "tet",
    CellType.HEXAHEDRON: "hex",
}


def _lies_strictly_within_segment(
    point: NDArray[np.float64],
    start: NDArray[np.float64],
    end: NDArray[np.float64],
) -> bool:
    """Whether ``point`` lies on the open segment ``(start, end)``."""
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared <= _TOL:
        return False
    t = float(np.dot(point - start, direction)) / length_squared
    if not 0.0 + _TOL < t < 1.0 - _TOL:
        return False
    return bool(np.allclose(point, start + t * direction, atol=1.0e-10))


class ShapeFunctions(ABC):
    """Evaluate N and reference derivatives on a reference cell."""

    @abstractmethod
    def cell(self) -> ReferenceCell:
        """Return the reference cell this family is defined on."""

    @abstractmethod
    def continuity(self) -> Continuity:
        """Return the continuity class this family provides (checked by E-9)."""

    @abstractmethod
    def completeness_degree(self) -> int:
        """Return the polynomial completeness degree p."""

    @abstractmethod
    def evaluate(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return shape values, shape (n_pts, n_functions)."""

    @abstractmethod
    def derivatives(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return reference derivatives, shape (n_pts, n_functions, dim)."""


class Interpolation(ABC):
    """A finite element's domain, polynomial space, and degrees of freedom.

    Concrete families declare five things - :pyattr:`family`, :pyattr:`order`,
    :pyattr:`reference_element`, :pyattr:`polynomial_space`, :pyattr:`nodes`,
    :pyattr:`dofs` - plus their design :pyattr:`continuity`. Everything else
    (identifiers, flags, degree metadata, the unisolvence oracle, the
    verification suite, serialization, value semantics) is derived here and
    shared by every family.

    No shape function is formed, evaluated, or differentiated at this layer.
    """

    # ---- abstract declarations ---------------------------------------------

    @property
    @abstractmethod
    def family(self) -> InterpolationFamily:
        """The interpolation family this element belongs to."""

    @property
    @abstractmethod
    def order(self) -> int:
        """The family's nominal polynomial order ``k``."""

    @property
    @abstractmethod
    def reference_element(self) -> ReferenceElement:
        """The reference domain this interpolation is defined on."""

    @property
    @abstractmethod
    def polynomial_space(self) -> PolynomialSpace:
        """The polynomial space the DOFs must uniquely determine."""

    @property
    @abstractmethod
    def nodes(self) -> tuple[InterpolationNode, ...]:
        """The interpolation nodes, ordered vertices then edges then interior."""

    @property
    @abstractmethod
    def dofs(self) -> tuple[DofFunctional, ...]:
        """The degrees of freedom as functionals on the polynomial space."""

    @property
    @abstractmethod
    def continuity(self) -> Continuity:
        """The inter-element continuity this family is designed to deliver.

        This is a property of the element *and the mesh it sits in*: an
        element is designed for a continuity class and realizes it when its
        mesh conditions hold (see each family's documentation for the
        caveats). It is not a promise the reference element alone can keep.
        """

    # ---- derived metadata ---------------------------------------------------

    @property
    def cell_type(self) -> CellType:
        """The shape of the reference domain."""
        return self.reference_element.cell_type

    @property
    def support_dimension(self) -> int:
        """Spatial dimension over which the interpolation is supported."""
        return self.reference_element.topological_dimension

    @property
    def num_nodes(self) -> int:
        """Number of interpolation nodes (geometric points)."""
        return len(self.nodes)

    @property
    def num_dofs(self) -> int:
        """Number of degrees of freedom (equals the number of shape functions)."""
        return len(self.dofs)

    @property
    def space_dimension(self) -> int:
        """Dimension of the polynomial space; must equal :pyattr:`num_dofs`."""
        return self.polynomial_space.dimension

    @property
    def completeness_degree(self) -> int:
        """Largest ``p`` with ``P_p`` contained in the space (governs the rate)."""
        return self.polynomial_space.completeness_degree

    @property
    def max_total_degree(self) -> int:
        """Highest total monomial degree present in the space."""
        return self.polynomial_space.max_total_degree

    @property
    def node_locations(self) -> NDArray[np.float64]:
        """Node coordinates, shape ``(n_nodes, dim)``, read-only."""
        array = np.array([node.coordinates for node in self.nodes], dtype=np.float64)
        array.setflags(write=False)
        return array

    @property
    def dofs_per_node(self) -> tuple[int, ...]:
        """Number of DOFs attached to each node, index-aligned with :pyattr:`nodes`."""
        counts = [0] * self.num_nodes
        for dof in self.dofs:
            counts[dof.node_index] += 1
        return tuple(counts)

    @property
    def is_nodal(self) -> bool:
        """Whether every DOF is a point value (the classical Lagrange property).

        When ``True`` the shape functions satisfy the classical Kronecker
        delta ``N_i(x_j) = delta_ij``. When ``False`` (Hermite) only the
        generalized duality ``l_i(N_j) = delta_ij`` applies.
        """
        return all(dof.kind is DofKind.POINT_VALUE for dof in self.dofs)

    @property
    def is_simplex(self) -> bool:
        """Whether the reference domain is a simplex (line, triangle, tetrahedron)."""
        return self.cell_type in _SIMPLEX_CELLS

    @property
    def is_tensor_product(self) -> bool:
        """Whether the reference domain is a tensor-product cell (line, quad, hex).

        The line is both a 1-simplex and a 1-cube, so both flags are ``True``
        for it. That is not a contradiction: in one variable ``P_k`` and
        ``Q_k`` are the same space, so the ambiguity has no consequence.
        """
        return self.cell_type in _TENSOR_PRODUCT_CELLS

    @property
    def shape_function_ids(self) -> tuple[str, ...]:
        """Identifier of each shape function, dual to the DOF of the same index.

        These name the (as yet unbuilt) nodal basis functions: ``"N_v0"`` is
        dual to the value DOF at vertex 0, ``"N_v0_d10"`` to the ``d/dxi``
        DOF there. Identifiers only - no equations.
        """
        return tuple(f"N_{dof.identifier}" for dof in self.dofs)

    @property
    def basis_ids(self) -> tuple[str, ...]:
        """Identifier of each monomial spanning the polynomial space.

        Distinct from :pyattr:`shape_function_ids`: these name the *span*
        (``"1"``, ``"xi"``, ``"xi*eta"``), while shape functions name the
        basis dual to the DOFs. Both have length :pyattr:`num_dofs` when the
        element is unisolvent.
        """
        return self.polynomial_space.monomial_labels

    @property
    def mesh_cell_name(self) -> str:
        """Conventional mesh cell-type name matching this element's node count.

        For example Lagrange order 2 on a triangle is ``"tri6"``; the cubic
        Hermite line is ``"line2"``, because its extra DOFs are derivatives at
        the two existing nodes rather than new points.
        """
        return f"{_MESH_NAME_STEM[self.cell_type]}{self.num_nodes}"

    # ---- queries ------------------------------------------------------------

    def interpolation_nodes(self) -> tuple[InterpolationNode, ...]:
        """The interpolation nodes (alias of :pyattr:`nodes`)."""
        return self.nodes

    def node(self, index: int) -> InterpolationNode:
        """The node with the given index."""
        if not 0 <= index < self.num_nodes:
            raise InputValidationError(f"node index {index} out of range [0, {self.num_nodes})")
        return self.nodes[index]

    def dof(self, index: int) -> DofFunctional:
        """The degree of freedom with the given index."""
        if not 0 <= index < self.num_dofs:
            raise InputValidationError(f"dof index {index} out of range [0, {self.num_dofs})")
        return self.dofs[index]

    def nodes_on_entity(
        self, entity: EntityType, entity_index: int
    ) -> tuple[InterpolationNode, ...]:
        """Nodes owned by one sub-entity, e.g. the nodes interior to edge 1."""
        return tuple(
            node
            for node in self.nodes
            if node.entity is entity and node.entity_index == entity_index
        )

    def dofs_on_node(self, node_index: int) -> tuple[DofFunctional, ...]:
        """The degrees of freedom acting at one node."""
        if not 0 <= node_index < self.num_nodes:
            raise InputValidationError(
                f"node index {node_index} out of range [0, {self.num_nodes})"
            )
        return tuple(dof for dof in self.dofs if dof.node_index == node_index)

    def evaluation_points(self) -> NDArray[np.float64]:
        """Distinct points at which the DOF functionals act, shape ``(n, dim)``.

        For the point-based families implemented here this coincides with
        :pyattr:`node_locations`. It is a separate query because the two
        notions diverge: a hierarchical or moment-based family has degrees of
        freedom that are integrals, so it has no evaluation points at all.
        """
        seen: list[tuple[float, ...]] = []
        for dof in self.dofs:
            if dof.point not in seen:
                seen.append(dof.point)
        array = np.array(seen, dtype=np.float64)
        array.setflags(write=False)
        return array

    # ---- the unisolvence oracle --------------------------------------------

    def unisolvence_matrix(self) -> NDArray[np.float64]:
        """The generalized Vandermonde ``M[k, j] = l_k(m_j)``, shape ``(n_dofs, n_space)``.

        Row ``k`` applies DOF functional ``k`` to every monomial of the space.
        The element is unisolvent - the DOFs uniquely determine a member of
        the space - exactly when ``M`` is square and invertible, which is what
        :meth:`verify_linear_independence` checks.

        ``M^-1`` would be the coefficient matrix of the shape functions in the
        monomial basis. This layer builds ``M`` and never inverts it; forming
        the dual basis is a later phase.
        """
        exponents = self.polynomial_space.exponents
        rows = [[dof.apply_to_monomial(alpha) for alpha in exponents] for dof in self.dofs]
        return np.array(rows, dtype=np.float64)

    def unisolvence_condition_number(self) -> float:
        """2-norm condition number of the unisolvence matrix.

        A diagnostic, not a pass/fail gate: the matrix is invertible for every
        element here, but its conditioning degrades as the order rises on
        equispaced nodes (the Runge phenomenon in another guise). That growth
        is the reason spectral families place nodes at Gauss-Lobatto-Legendre
        points instead.
        """
        return float(np.linalg.cond(self.unisolvence_matrix()))

    # ---- verification -------------------------------------------------------

    def is_valid(self) -> bool:
        """Return ``True`` if :meth:`validate` passes, ``False`` otherwise."""
        try:
            self.validate()
        except InterpolationError:
            return False
        return True

    def validate(self) -> None:
        """Run every verification; raise on the first failure."""
        self.verify_polynomial_degree()
        self.verify_polynomial_completeness()
        self.verify_linear_independence()
        self.verify_kronecker_delta()
        self.verify_partition_of_unity()
        self.verify_node_ordering()

    def verify_polynomial_degree(self) -> None:
        """Check the space's declared order against the monomials it holds.

        The nominal order must be the space's completeness degree, and the
        space's dimension must match the dimension its construction rule
        predicts.
        """
        space = self.polynomial_space
        if space.order != self.order:
            raise PolynomialSpaceError(
                f"{self._label()}: space {space.name} has order {space.order}, "
                f"but the element declares order {self.order}"
            )
        if space.n_variables != self.support_dimension:
            raise PolynomialSpaceError(
                f"{self._label()}: space has {space.n_variables} variables, "
                f"but the reference domain has dimension {self.support_dimension}"
            )
        if space.dimension != space.expected_dimension:
            raise PolynomialSpaceError(
                f"{self._label()}: space {space.name} has dimension {space.dimension}, "
                f"but its construction rule predicts {space.expected_dimension}"
            )
        if space.completeness_degree != self.order:
            raise PolynomialSpaceError(
                f"{self._label()}: order {self.order} disagrees with the space's "
                f"completeness degree {space.completeness_degree}"
            )

    def verify_polynomial_completeness(self) -> None:
        """Check the space contains every monomial up to its completeness degree.

        Also checks that degree is *maximal*: the next degree must be
        incomplete, otherwise the reported approximation order understates the
        space.
        """
        space = self.polynomial_space
        degree = space.completeness_degree
        if degree < 0:
            raise PolynomialSpaceError(f"{self._label()}: space omits the constant")
        for candidate in range(degree + 1):
            if not space.contains_total_degree(candidate):
                raise PolynomialSpaceError(
                    f"{self._label()}: space is missing monomials of total degree {candidate}"
                )
        if space.contains_total_degree(degree + 1):
            raise PolynomialSpaceError(
                f"{self._label()}: space is complete to degree {degree + 1}, "
                f"but reports completeness degree {degree}"
            )

    def verify_linear_independence(self) -> None:
        """Check the DOF functionals are linearly independent over the space.

        Equivalent to unisolvence: the generalized Vandermonde must be square
        (``n_dofs == dim(space)``) and full rank. If it is not, no unique
        nodal basis dual to these DOFs exists and no shape functions can be
        built - which is why this check belongs here rather than in the phase
        that would build them.
        """
        matrix = self.unisolvence_matrix()
        if matrix.shape[0] != matrix.shape[1]:
            raise UnisolvenceError(
                f"{self._label()}: {self.num_dofs} dofs against a space of dimension "
                f"{self.space_dimension}; the element cannot be unisolvent"
            )
        rank = int(np.linalg.matrix_rank(matrix))
        if rank != matrix.shape[0]:
            raise UnisolvenceError(
                f"{self._label()}: unisolvence matrix is rank deficient "
                f"(rank {rank} of {matrix.shape[0]}); the dofs do not determine the space"
            )

    def verify_kronecker_delta(self) -> None:
        """Check the duality ``l_i(N_j) = delta_ij`` is well posed.

        Duality is how the nodal basis is *defined*, so what can fail is not
        the identity but the existence and uniqueness of a basis satisfying
        it. That needs two things, both checked here: the functionals must be
        pairwise distinct, and the element must be unisolvent.

        For a nodal family (:pyattr:`is_nodal`) this is the classical
        ``N_i(x_j) = delta_ij``, and the nodes are additionally checked to be
        distinct and in bijection with the DOFs. For Hermite the classical
        form does not apply - several functionals share a point - and only the
        generalized statement holds.
        """
        signatures = [(dof.point, dof.derivative) for dof in self.dofs]
        if len(set(signatures)) != len(signatures):
            raise UnisolvenceError(
                f"{self._label()}: duplicate dof functional (same point and derivative); "
                f"the dual basis would not be unique"
            )
        if self.is_nodal:
            if self.num_dofs != self.num_nodes:
                raise UnisolvenceError(
                    f"{self._label()}: nodal family with {self.num_dofs} dofs at "
                    f"{self.num_nodes} nodes; a nodal family needs one dof per node"
                )
            if len({node.coordinates for node in self.nodes}) != self.num_nodes:
                raise UnisolvenceError(f"{self._label()}: duplicate interpolation node")
        self.verify_linear_independence()

    def verify_partition_of_unity(self) -> None:
        """Check the element reproduces constants exactly.

        Given unisolvence, the nodal basis reproduces the constant function
        precisely when the constant monomial lies in the space; the DOF vector
        of ``u = 1`` is then ``l_k(1)``, so ``1 = sum_k l_k(1) N_k``. Since a
        derivative functional annihilates a constant, ``l_k(1)`` is 1 on the
        value DOFs and 0 on the derivative DOFs. Hence:

        - for a nodal family this is the classical partition of unity,
          ``sum_i N_i = 1``;
        - for Hermite the value-dual functions sum to 1 while the
          derivative-dual functions contribute nothing.

        Both are verified here from the functionals alone, with no shape
        function evaluated.
        """
        space = self.polynomial_space
        if not space.contains_constant:
            raise PolynomialSpaceError(
                f"{self._label()}: the constant is not in {space.name}; "
                f"constants cannot be reproduced"
            )
        constant = (0,) * self.support_dimension
        for dof in self.dofs:
            value = dof.apply_to_monomial(constant)
            expected = 1.0 if dof.kind is DofKind.POINT_VALUE else 0.0
            if not math.isclose(value, expected, abs_tol=_TOL):
                raise UnisolvenceError(
                    f"{self._label()}: dof {dof.identifier} applied to the constant gives "
                    f"{value}, expected {expected}"
                )
        self.verify_linear_independence()

    def verify_node_ordering(self) -> None:
        """Check node numbering and entity associations against the reference element.

        Verifies that node indices run ``0..n-1`` in order; every node lies in
        the reference domain; each node's declared entity association is
        geometrically true (a vertex node sits exactly on that vertex, an edge
        node lies strictly between that edge's endpoints, an interior node is
        strictly inside); vertex nodes come first and in reference order; and
        every DOF points at a node whose location it shares.
        """
        element = self.reference_element
        vertices = element.vertex_coordinates
        for index, node in enumerate(self.nodes):
            if node.index != index:
                raise NodeOrderingError(
                    f"{self._label()}: node at position {index} declares index {node.index}"
                )
            if node.dimension != self.support_dimension:
                raise NodeOrderingError(
                    f"{self._label()}: node {index} has dimension {node.dimension}, "
                    f"expected {self.support_dimension}"
                )
            point = np.asarray(node.coordinates, dtype=np.float64)
            if not element.contains(point, tol=_TOL):
                raise NodeOrderingError(
                    f"{self._label()}: node {index} at {node.coordinates} is outside "
                    f"the reference {self.cell_type.value}"
                )
            self._verify_node_entity(node, point, vertices)
        self._verify_vertex_nodes_come_first()
        for dof in self.dofs:
            if not 0 <= dof.node_index < self.num_nodes:
                raise NodeOrderingError(
                    f"{self._label()}: dof {dof.identifier} refers to unknown node "
                    f"{dof.node_index}"
                )
            if dof.point != self.nodes[dof.node_index].coordinates:
                raise NodeOrderingError(
                    f"{self._label()}: dof {dof.identifier} acts at {dof.point} but its "
                    f"node {dof.node_index} is at {self.nodes[dof.node_index].coordinates}"
                )

    def _verify_node_entity(
        self,
        node: InterpolationNode,
        point: NDArray[np.float64],
        vertices: NDArray[np.float64],
    ) -> None:
        """Check one node's declared entity association is geometrically true."""
        element = self.reference_element
        if node.entity is EntityType.VERTEX:
            if node.entity_index >= element.num_vertices:
                raise NodeOrderingError(
                    f"{self._label()}: node {node.index} claims unknown vertex "
                    f"{node.entity_index}"
                )
            if not np.allclose(point, vertices[node.entity_index], atol=_TOL):
                raise NodeOrderingError(
                    f"{self._label()}: node {node.index} claims vertex {node.entity_index} "
                    f"but sits at {node.coordinates}"
                )
        elif node.entity is EntityType.EDGE:
            if node.entity_index >= element.num_edges:
                raise NodeOrderingError(
                    f"{self._label()}: node {node.index} claims unknown edge "
                    f"{node.entity_index}"
                )
            start, end = element.edge_vertex_indices[node.entity_index]
            if not _lies_strictly_within_segment(point, vertices[start], vertices[end]):
                raise NodeOrderingError(
                    f"{self._label()}: node {node.index} claims edge {node.entity_index} "
                    f"but does not lie strictly on it"
                )
        elif node.entity is EntityType.CELL:
            if element.signed_distance_to_boundary(point) >= -_TOL:
                raise NodeOrderingError(
                    f"{self._label()}: node {node.index} claims to be interior but sits "
                    f"on the boundary at {node.coordinates}"
                )
        else:
            raise NodeOrderingError(
                f"{self._label()}: node {node.index} claims entity {node.entity.value}, "
                f"which this phase does not associate nodes with"
            )

    def _verify_vertex_nodes_come_first(self) -> None:
        """Check vertex nodes occupy the leading indices, in reference vertex order."""
        vertex_nodes = [node for node in self.nodes if node.entity is EntityType.VERTEX]
        if not vertex_nodes:
            return
        for position, node in enumerate(vertex_nodes):
            if node.index != position or node.entity_index != position:
                raise NodeOrderingError(
                    f"{self._label()}: vertex nodes must occupy indices 0..n-1 in reference "
                    f"order; node {node.index} claims vertex {node.entity_index}"
                )

    # ---- serialization and utilities ---------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible record of the interpolation's complete metadata."""
        return {
            "schema": "nanofem-interpolation/1",
            "family": self.family.value,
            "order": self.order,
            "cell_type": self.cell_type.value,
            "continuity": self.continuity.name,
            "support_dimension": self.support_dimension,
            "is_nodal": self.is_nodal,
            "is_simplex": self.is_simplex,
            "is_tensor_product": self.is_tensor_product,
            "num_nodes": self.num_nodes,
            "num_dofs": self.num_dofs,
            "dofs_per_node": list(self.dofs_per_node),
            "mesh_cell_name": self.mesh_cell_name,
            "completeness_degree": self.completeness_degree,
            "max_total_degree": self.max_total_degree,
            "polynomial_space": self.polynomial_space.to_dict(),
            "nodes": [node.to_dict() for node in self.nodes],
            "dofs": [dof.to_dict() for dof in self.dofs],
            "shape_function_ids": list(self.shape_function_ids),
            "basis_ids": list(self.basis_ids),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize :meth:`to_dict` to JSON (keys sorted for determinism)."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    def pretty(self) -> str:
        """A compact, human-readable summary of the interpolation."""
        space = self.polynomial_space
        return "\n".join(
            [
                f"{type(self).__name__} [{self.family.value} order {self.order} "
                f"on {self.cell_type.value}]",
                f"  space            : {space.name} (dimension {space.dimension}, "
                f"complete to degree {self.completeness_degree}, "
                f"max total degree {self.max_total_degree})",
                f"  dofs / nodes     : {self.num_dofs} dofs at {self.num_nodes} nodes "
                f"({'nodal' if self.is_nodal else 'non-nodal'})",
                f"  continuity       : {self.continuity.name}",
                f"  flags            : simplex={self.is_simplex} "
                f"tensor_product={self.is_tensor_product} "
                f"support_dimension={self.support_dimension}",
                f"  mesh cell name   : {self.mesh_cell_name}",
            ]
        )

    def debug_summary(self) -> str:
        """A verbose dump: the summary plus every node, DOF, and monomial."""
        parts = [self.pretty(), "  nodes:"]
        for node in self.nodes:
            coordinates = ", ".join(f"{x:+.6f}" for x in node.coordinates)
            parts.append(
                f"    [{node.index}] {node.identifier:<8} ({coordinates})  "
                f"{node.entity.value} {node.entity_index}"
            )
        parts.append("  dofs (functionals):")
        for dof in self.dofs:
            parts.append(
                f"    [{dof.index}] {dof.identifier:<10} {dof.kind.value:<17} "
                f"dual to {self.shape_function_ids[dof.index]}"
            )
        parts.append(f"  monomial basis   : {', '.join(self.basis_ids)}")
        parts.append(f"  unisolvence cond : {self.unisolvence_condition_number():.4g}")
        return "\n".join(parts)

    def _label(self) -> str:
        """Short identifier used in error messages."""
        return f"{self.family.value} order {self.order} on {self.cell_type.value}"

    # ---- value semantics ----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(cell_type={self.cell_type.value!r}, order={self.order}, "
            f"dofs={self.num_dofs}, space={self.polynomial_space.name})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Interpolation):
            return NotImplemented
        return (self.family, self.cell_type, self.order) == (
            other.family,
            other.cell_type,
            other.order,
        )

    def __hash__(self) -> int:
        return hash((Interpolation, self.family, self.cell_type, self.order))
