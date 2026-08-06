"""Interpolation families: metadata, node numbering, and the verification suite.

The success criterion of this phase lives here: build an element, query its
nodes and degree, validate its metadata, without evaluating a shape function.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.numerics.interpolation import (
    DofFunctional,
    HermiteInterpolation,
    Interpolation,
    InterpolationFamily,
    InterpolationNode,
    LagrangeInterpolation,
    NodeOrderingError,
    PolynomialSpace,
    PolynomialSpaceError,
    PolynomialSpaceType,
    UnisolvenceError,
    available_interpolations,
    interpolation,
)
from nanofem.numerics.operators.base import Continuity
from nanofem.numerics.reference.enums import CellType, EntityType
from nanofem.utils.exceptions import InputValidationError

ALL: list[Interpolation] = [interpolation(f, c, k) for f, c, k in available_interpolations()]


# ---- the success criterion --------------------------------------------------


def test_success_criterion_chain() -> None:
    """ReferenceTriangle -> Lagrange(order=2) -> nodes -> degree -> validate -> serialize."""
    element = LagrangeInterpolation(CellType.TRIANGLE, 2)
    assert element.reference_element.cell_type is CellType.TRIANGLE
    assert element.num_nodes == 6
    assert element.order == 2
    assert element.completeness_degree == 2
    element.validate()
    assert element.to_json()


# ---- declared metadata ------------------------------------------------------


@pytest.mark.parametrize(
    ("cell", "order", "space", "n_nodes", "mesh_name"),
    [
        (CellType.LINE, 1, "P1", 2, "line2"),
        (CellType.LINE, 2, "P2", 3, "line3"),
        (CellType.LINE, 3, "P3", 4, "line4"),
        (CellType.TRIANGLE, 1, "P1", 3, "tri3"),
        (CellType.TRIANGLE, 2, "P2", 6, "tri6"),
        (CellType.TRIANGLE, 3, "P3", 10, "tri10"),
        (CellType.QUADRILATERAL, 1, "Q1", 4, "quad4"),
        (CellType.QUADRILATERAL, 2, "Q2", 9, "quad9"),
        (CellType.QUADRILATERAL, 3, "Q3", 16, "quad16"),
    ],
)
def test_lagrange_metadata(
    cell: CellType, order: int, space: str, n_nodes: int, mesh_name: str
) -> None:
    """Lagrange spaces, node counts, and mesh names match the classical elements."""
    element = LagrangeInterpolation(cell, order)
    assert element.family is InterpolationFamily.LAGRANGE
    assert element.polynomial_space.name == space
    assert element.num_nodes == n_nodes
    assert element.num_dofs == n_nodes == element.space_dimension
    assert element.mesh_cell_name == mesh_name
    assert element.is_nodal
    assert element.continuity is Continuity.C0
    assert element.dofs_per_node == (1,) * n_nodes


@pytest.mark.parametrize(
    ("cell", "space", "n_nodes", "n_dofs", "mesh_name"),
    [
        (CellType.LINE, "P3", 2, 4, "line2"),
        (CellType.QUADRILATERAL, "Q3", 4, 16, "quad4"),
    ],
)
def test_hermite_metadata(
    cell: CellType, space: str, n_nodes: int, n_dofs: int, mesh_name: str
) -> None:
    """Hermite adds derivative DOFs at existing nodes rather than new points."""
    element = HermiteInterpolation(cell, 3)
    assert element.family is InterpolationFamily.HERMITE
    assert element.polynomial_space.name == space
    assert element.num_nodes == n_nodes
    assert element.num_dofs == n_dofs == element.space_dimension
    assert element.mesh_cell_name == mesh_name
    assert not element.is_nodal
    assert element.continuity is Continuity.C1
    assert element.dofs_per_node == (n_dofs // n_nodes,) * n_nodes


def test_hermite_line_uses_the_classical_beam_dof_order() -> None:
    """DOFs are (w1, theta1, w2, theta2): value then slope, grouped per vertex."""
    element = HermiteInterpolation(CellType.LINE, 3)
    assert element.shape_function_ids == ("N_v0", "N_v0_d1", "N_v1", "N_v1_d1")
    assert [dof.derivative for dof in element.dofs] == [(0,), (1,), (0,), (1,)]


def test_bfs_quad_uses_value_two_gradients_and_the_cross_derivative() -> None:
    """The BFS set at each vertex is (u, du/dxi, du/deta, d2u/dxi deta), in that order."""
    element = HermiteInterpolation(CellType.QUADRILATERAL, 3)
    assert element.shape_function_ids[:4] == ("N_v0", "N_v0_d10", "N_v0_d01", "N_v0_d11")
    assert [dof.derivative for dof in element.dofs[:4]] == [(0, 0), (1, 0), (0, 1), (1, 1)]


def test_flags_track_the_cell_shape_and_the_line_is_both() -> None:
    """Simplex/tensor-product flags follow the cell; the line is legitimately both."""
    tri = LagrangeInterpolation(CellType.TRIANGLE, 1)
    assert tri.is_simplex and not tri.is_tensor_product
    quad = LagrangeInterpolation(CellType.QUADRILATERAL, 1)
    assert quad.is_tensor_product and not quad.is_simplex
    line = LagrangeInterpolation(CellType.LINE, 1)
    assert line.is_simplex and line.is_tensor_product  # a 1-simplex is also a 1-cube
    assert line.polynomial_space.exponents == ((0,), (1,))  # P1 == Q1 in one variable


def test_support_dimension_follows_the_reference_element() -> None:
    """Support dimension is the reference domain's topological dimension."""
    assert LagrangeInterpolation(CellType.LINE, 2).support_dimension == 1
    assert LagrangeInterpolation(CellType.TRIANGLE, 2).support_dimension == 2
    assert HermiteInterpolation(CellType.QUADRILATERAL, 3).support_dimension == 2


def test_shape_function_ids_and_basis_ids_are_different_things() -> None:
    """Shape functions are dual to DOFs; basis ids name the monomial span."""
    element = LagrangeInterpolation(CellType.TRIANGLE, 1)
    assert element.shape_function_ids == ("N_v0", "N_v1", "N_v2")
    assert element.basis_ids == ("1", "eta", "xi")
    assert len(element.shape_function_ids) == len(element.basis_ids) == element.num_dofs


# ---- node numbering ---------------------------------------------------------


def test_lagrange_vertex_nodes_come_first_in_reference_order() -> None:
    """The first n_vertices nodes are the reference vertices, in order."""
    for element in ALL:
        reference = element.reference_element
        for index in range(reference.num_vertices):
            node = element.node(index)
            assert node.entity is EntityType.VERTEX and node.entity_index == index
            assert np.allclose(node.coordinates, reference.vertex_coordinates[index])


def test_tri6_edge_nodes_follow_the_reference_edge_order() -> None:
    """Edge node i is the midpoint of reference edge i, not of gmsh's edge i.

    The reference triangle's edges are ((1,2), (2,0), (0,1)) - facet i opposite
    vertex i - so node 3 is the midpoint of (V1, V2). External formats number
    these differently; translating is the mesh I/O adapter's job.
    """
    element = LagrangeInterpolation(CellType.TRIANGLE, 2)
    assert np.allclose(element.node(3).coordinates, (0.5, 0.5))  # edge 0 = (V1, V2)
    assert np.allclose(element.node(4).coordinates, (0.0, 0.5))  # edge 1 = (V2, V0)
    assert np.allclose(element.node(5).coordinates, (0.5, 0.0))  # edge 2 = (V0, V1)
    for index in (3, 4, 5):
        assert element.node(index).entity is EntityType.EDGE
        assert element.node(index).entity_index == index - 3


def test_quad9_has_four_edge_midpoints_and_one_centre() -> None:
    """Q2 on a quadrilateral: 4 vertices, 4 edge midpoints (bottom/right/top/left), centre."""
    element = LagrangeInterpolation(CellType.QUADRILATERAL, 2)
    assert np.allclose(element.node_locations[4], (0.0, -1.0))  # facet 0, bottom
    assert np.allclose(element.node_locations[5], (1.0, 0.0))  # facet 1, right
    assert np.allclose(element.node_locations[6], (0.0, 1.0))  # facet 2, top
    assert np.allclose(element.node_locations[7], (-1.0, 0.0))  # facet 3, left
    assert np.allclose(element.node_locations[8], (0.0, 0.0))  # interior
    assert element.node(8).entity is EntityType.CELL


def test_tri10_has_one_interior_node_at_the_centroid() -> None:
    """P3 on a triangle: 3 vertices + 2 per edge + 1 interior at the centroid."""
    element = LagrangeInterpolation(CellType.TRIANGLE, 3)
    interior = element.nodes_on_entity(EntityType.CELL, 0)
    assert len(interior) == 1
    assert np.allclose(interior[0].coordinates, (1.0 / 3.0, 1.0 / 3.0))
    assert len(element.nodes_on_entity(EntityType.EDGE, 0)) == 2


def test_line_interior_nodes_belong_to_the_cell_not_an_edge() -> None:
    """A line's only edge is the cell itself, so its interior nodes are cell nodes."""
    element = LagrangeInterpolation(CellType.LINE, 3)
    assert element.node(2).entity is EntityType.CELL
    assert np.allclose(element.node_locations[2:], [[-1.0 / 3.0], [1.0 / 3.0]])


def test_evaluation_points_are_distinct_functional_points() -> None:
    """Point-based families evaluate at their nodes; Hermite shares points across DOFs."""
    lagrange = LagrangeInterpolation(CellType.TRIANGLE, 2)
    assert np.allclose(lagrange.evaluation_points(), lagrange.node_locations)
    hermite = HermiteInterpolation(CellType.LINE, 3)
    assert hermite.evaluation_points().shape == (2, 1)  # 4 dofs, 2 distinct points
    assert hermite.num_dofs == 4


def test_node_and_dof_lookups_and_queries() -> None:
    """Index queries work and reject out-of-range indices."""
    element = HermiteInterpolation(CellType.LINE, 3)
    assert element.node(1).index == 1
    assert element.dof(3).identifier == "v1_d1"
    assert len(element.dofs_on_node(0)) == 2
    for bad in (-1, 99):
        with pytest.raises(InputValidationError):
            element.node(bad)
        with pytest.raises(InputValidationError):
            element.dof(bad)
        with pytest.raises(InputValidationError):
            element.dofs_on_node(bad)


# ---- verification suite -----------------------------------------------------


@pytest.mark.parametrize("element", ALL, ids=lambda e: f"{e.family.value}-{e.cell_type.value}")
def test_every_element_validates(element: Interpolation) -> None:
    """Every implemented element passes the whole verification suite."""
    element.validate()
    assert element.is_valid()


@pytest.mark.parametrize("element", ALL, ids=lambda e: f"{e.family.value}-{e.cell_type.value}")
def test_unisolvence_matrix_is_square_and_invertible(element: Interpolation) -> None:
    """The generalized Vandermonde is square and full rank for every element."""
    matrix = element.unisolvence_matrix()
    assert matrix.shape == (element.num_dofs, element.space_dimension)
    assert matrix.shape[0] == matrix.shape[1]
    assert np.linalg.matrix_rank(matrix) == matrix.shape[0]
    assert np.isfinite(element.unisolvence_condition_number())


def test_conditioning_degrades_with_order_on_equispaced_nodes() -> None:
    """The Runge phenomenon in another guise: this is why spectral nodes exist."""
    conditions = [
        LagrangeInterpolation(CellType.LINE, order).unisolvence_condition_number()
        for order in (1, 2, 3)
    ]
    assert conditions == sorted(conditions)
    assert conditions[0] < conditions[-1]


def test_verify_linear_independence_catches_a_dimension_mismatch() -> None:
    """Too few DOFs for the space cannot be unisolvent, and says so."""

    class _UnderDeterminedTriangle(LagrangeInterpolation):
        @property
        def polynomial_space(self) -> PolynomialSpace:
            return PolynomialSpace.total_degree(2, 2)  # 6 monomials against 3 dofs

    bad = _UnderDeterminedTriangle(CellType.TRIANGLE, 1)
    with pytest.raises(UnisolvenceError, match="cannot be unisolvent"):
        bad.verify_linear_independence()
    assert not bad.is_valid()


def test_verify_linear_independence_catches_coincident_nodes() -> None:
    """Duplicated nodes make the Vandermonde rank deficient."""

    class _CollapsedTriangle(LagrangeInterpolation):
        @property
        def dofs(self) -> tuple[DofFunctional, ...]:
            base = super().dofs
            moved = DofFunctional(
                index=2,
                node_index=2,
                point=base[0].point,  # collapse vertex 2 onto vertex 0
                derivative=(0, 0),
                entity=EntityType.VERTEX,
                entity_index=2,
            )
            return (base[0], base[1], moved)

    bad = _CollapsedTriangle(CellType.TRIANGLE, 1)
    with pytest.raises(UnisolvenceError, match="rank deficient|duplicate"):
        bad.verify_kronecker_delta()


def test_verify_polynomial_degree_catches_a_declared_order_mismatch() -> None:
    """A space whose order disagrees with the element's is caught."""

    class _MislabelledLine(LagrangeInterpolation):
        @property
        def order(self) -> int:
            return 2

    bad = _MislabelledLine(CellType.LINE, 1)
    with pytest.raises(PolynomialSpaceError, match="order"):
        bad.verify_polynomial_degree()


def test_verify_partition_of_unity_catches_a_space_without_the_constant() -> None:
    """Dropping the constant makes constant reproduction impossible."""

    class _NoConstantLine(LagrangeInterpolation):
        @property
        def polynomial_space(self) -> PolynomialSpace:
            return PolynomialSpace(PolynomialSpaceType.TOTAL_DEGREE, 1, 1, ((1,), (2,)))

    bad = _NoConstantLine(CellType.LINE, 1)
    with pytest.raises(PolynomialSpaceError, match="constant"):
        bad.verify_partition_of_unity()


def test_partition_of_unity_distinguishes_value_and_derivative_dofs() -> None:
    """l_k(1) is 1 on value DOFs and 0 on derivative DOFs: Hermite's value duals sum to 1."""
    hermite = HermiteInterpolation(CellType.LINE, 3)
    hermite.verify_partition_of_unity()
    applied = [dof.apply_to_monomial((0,)) for dof in hermite.dofs]
    assert applied == [1.0, 0.0, 1.0, 0.0]
    lagrange = LagrangeInterpolation(CellType.LINE, 2)
    assert [dof.apply_to_monomial((0,)) for dof in lagrange.dofs] == [1.0, 1.0, 1.0]


def test_verify_node_ordering_catches_a_node_outside_the_domain() -> None:
    """A node outside the reference cell is rejected using phase-2 containment."""

    class _EscapedNode(LagrangeInterpolation):
        @property
        def nodes(self) -> tuple[InterpolationNode, ...]:
            base = super().nodes
            return (base[0], base[1], InterpolationNode(2, (5.0, 5.0), EntityType.VERTEX, 2))

    with pytest.raises(NodeOrderingError, match="outside"):
        _EscapedNode(CellType.TRIANGLE, 1).verify_node_ordering()


def test_verify_node_ordering_catches_a_false_entity_claim() -> None:
    """A node claiming an edge it does not lie on is rejected."""

    class _LyingEdgeNode(LagrangeInterpolation):
        @property
        def nodes(self) -> tuple[InterpolationNode, ...]:
            base = list(super().nodes)
            base[3] = InterpolationNode(3, (0.5, 0.5), EntityType.EDGE, 2)  # really edge 0
            return tuple(base)

    with pytest.raises(NodeOrderingError, match="does not lie strictly"):
        _LyingEdgeNode(CellType.TRIANGLE, 2).verify_node_ordering()


def test_verify_node_ordering_catches_a_misnumbered_node() -> None:
    """A node whose declared index contradicts its position is rejected."""

    class _MisnumberedNode(LagrangeInterpolation):
        @property
        def nodes(self) -> tuple[InterpolationNode, ...]:
            base = super().nodes
            return (base[0], InterpolationNode(7, base[1].coordinates, EntityType.VERTEX, 1))

    with pytest.raises(NodeOrderingError, match="declares index"):
        _MisnumberedNode(CellType.LINE, 1).verify_node_ordering()


def test_verify_node_ordering_catches_an_interior_node_on_the_boundary() -> None:
    """A node claiming to be interior while sitting on the boundary is rejected."""

    class _BoundaryInterior(LagrangeInterpolation):
        @property
        def nodes(self) -> tuple[InterpolationNode, ...]:
            base = list(super().nodes)
            base[8] = InterpolationNode(8, (1.0, 0.0), EntityType.CELL, 0)
            return tuple(base)

    with pytest.raises(NodeOrderingError, match="interior"):
        _BoundaryInterior(CellType.QUADRILATERAL, 2).verify_node_ordering()


# ---- verification: the remaining rules are each proven to fire ---------------


def test_verify_polynomial_degree_catches_a_variable_count_mismatch() -> None:
    """A space in the wrong number of variables for the cell is caught."""

    class _OneVariableTriangle(LagrangeInterpolation):
        @property
        def polynomial_space(self) -> PolynomialSpace:
            return PolynomialSpace.total_degree(2, 1)

    with pytest.raises(PolynomialSpaceError, match="variables"):
        _OneVariableTriangle(CellType.TRIANGLE, 2).verify_polynomial_degree()


def test_verify_polynomial_degree_catches_a_dimension_that_defies_its_rule() -> None:
    """A P_k space whose monomial count contradicts C(k+d, d) is caught."""

    class _TruncatedP1(LagrangeInterpolation):
        @property
        def polynomial_space(self) -> PolynomialSpace:
            return PolynomialSpace(PolynomialSpaceType.TOTAL_DEGREE, 1, 2, ((0, 0), (1, 0)))

    with pytest.raises(PolynomialSpaceError, match="construction rule predicts"):
        _TruncatedP1(CellType.TRIANGLE, 1).verify_polynomial_degree()


def test_verify_polynomial_degree_catches_an_order_completeness_disagreement() -> None:
    """An order-1 element whose space skips the linear monomial is caught."""

    class _GappedLine(LagrangeInterpolation):
        @property
        def polynomial_space(self) -> PolynomialSpace:
            return PolynomialSpace(PolynomialSpaceType.TOTAL_DEGREE, 1, 1, ((0,), (2,)))

    with pytest.raises(PolynomialSpaceError, match="completeness degree"):
        _GappedLine(CellType.LINE, 1).verify_polynomial_degree()


def test_verify_polynomial_completeness_catches_a_space_omitting_the_constant() -> None:
    """Completeness reports -1 when the constant is missing, and that raises."""

    class _NoConstant(LagrangeInterpolation):
        @property
        def polynomial_space(self) -> PolynomialSpace:
            return PolynomialSpace(PolynomialSpaceType.TOTAL_DEGREE, 1, 1, ((1,), (2,)))

    with pytest.raises(PolynomialSpaceError, match="omits the constant"):
        _NoConstant(CellType.LINE, 1).verify_polynomial_completeness()


def test_verify_polynomial_completeness_catches_an_understated_degree() -> None:
    """The maximality guard: a space complete past its reported degree is caught.

    This defends the reported approximation order against a space type whose
    completeness degree is hand-declared rather than computed - which is what
    the serendipity family will need.
    """

    class _UnderreportingSpace(PolynomialSpace):
        @property
        def completeness_degree(self) -> int:
            return 0  # the space is really complete to degree 1

    class _UnderreportingLine(LagrangeInterpolation):
        @property
        def polynomial_space(self) -> PolynomialSpace:
            return _UnderreportingSpace(PolynomialSpaceType.TOTAL_DEGREE, 1, 1, ((0,), (1,)))

    with pytest.raises(PolynomialSpaceError, match="complete to degree 1"):
        _UnderreportingLine(CellType.LINE, 1).verify_polynomial_completeness()


def test_verify_kronecker_delta_catches_a_nodal_family_missing_a_dof() -> None:
    """A nodal family needs exactly one DOF per node."""

    class _MissingDof(LagrangeInterpolation):
        @property
        def dofs(self) -> tuple[DofFunctional, ...]:
            return super().dofs[:-1]

    with pytest.raises(UnisolvenceError, match="one dof per node"):
        _MissingDof(CellType.TRIANGLE, 1).verify_kronecker_delta()


def test_verify_node_ordering_catches_a_node_of_the_wrong_dimension() -> None:
    """A 1-D node on a 2-D cell is caught before any geometric test runs."""

    class _FlatNode(LagrangeInterpolation):
        @property
        def nodes(self) -> tuple[InterpolationNode, ...]:
            base = list(super().nodes)
            base[0] = InterpolationNode(0, (0.0,), EntityType.VERTEX, 0)
            return tuple(base)

    with pytest.raises(NodeOrderingError, match="dimension"):
        _FlatNode(CellType.TRIANGLE, 1).verify_node_ordering()


def test_verify_node_ordering_catches_claims_on_entities_that_do_not_exist() -> None:
    """Vertex and edge indices beyond the reference element's lattice are caught."""

    class _GhostVertex(LagrangeInterpolation):
        @property
        def nodes(self) -> tuple[InterpolationNode, ...]:
            base = list(super().nodes)
            base[2] = InterpolationNode(2, base[2].coordinates, EntityType.VERTEX, 9)
            return tuple(base)

    with pytest.raises(NodeOrderingError, match="unknown vertex"):
        _GhostVertex(CellType.TRIANGLE, 1).verify_node_ordering()

    class _GhostEdge(LagrangeInterpolation):
        @property
        def nodes(self) -> tuple[InterpolationNode, ...]:
            base = list(super().nodes)
            base[3] = InterpolationNode(3, base[3].coordinates, EntityType.EDGE, 9)
            return tuple(base)

    with pytest.raises(NodeOrderingError, match="unknown edge"):
        _GhostEdge(CellType.TRIANGLE, 2).verify_node_ordering()


def test_verify_node_ordering_catches_an_unsupported_entity_association() -> None:
    """This phase associates nodes with vertices, edges, and the cell only."""

    class _FaceNode(LagrangeInterpolation):
        @property
        def nodes(self) -> tuple[InterpolationNode, ...]:
            base = list(super().nodes)
            base[8] = InterpolationNode(8, (0.0, 0.0), EntityType.FACE, 0)
            return tuple(base)

    with pytest.raises(NodeOrderingError, match="does not associate nodes with"):
        _FaceNode(CellType.QUADRILATERAL, 2).verify_node_ordering()


def test_verify_node_ordering_catches_permuted_vertex_nodes() -> None:
    """Vertex nodes must occupy the leading indices in reference vertex order.

    A genuine permutation - vertices 1 and 2 listed in swapped positions -
    passes every per-node check: each node really does sit on the vertex it
    claims. Only the ordering rule catches it, which is why the rule exists
    separately from the entity-association check.
    """

    class _PermutedVertices(LagrangeInterpolation):
        @property
        def nodes(self) -> tuple[InterpolationNode, ...]:
            reference = self.reference_element.vertex_coordinates
            return (
                InterpolationNode(0, tuple(float(x) for x in reference[0]), EntityType.VERTEX, 0),
                InterpolationNode(1, tuple(float(x) for x in reference[2]), EntityType.VERTEX, 2),
                InterpolationNode(2, tuple(float(x) for x in reference[1]), EntityType.VERTEX, 1),
            )

    element = _PermutedVertices(CellType.TRIANGLE, 1)
    for node in element.nodes:  # each node is individually consistent
        assert np.allclose(
            node.coordinates, element.reference_element.vertex_coordinates[node.entity_index]
        )
    with pytest.raises(NodeOrderingError, match="reference order"):
        element.verify_node_ordering()


def test_verify_node_ordering_catches_a_dof_detached_from_its_node() -> None:
    """A DOF must act at the location of the node it names."""

    class _DetachedDof(LagrangeInterpolation):
        @property
        def dofs(self) -> tuple[DofFunctional, ...]:
            base = list(super().dofs)
            base[1] = DofFunctional(1, 1, (0.25, 0.25), (0, 0), EntityType.VERTEX, 1)
            return tuple(base)

    with pytest.raises(NodeOrderingError, match="acts at"):
        _DetachedDof(CellType.TRIANGLE, 1).verify_node_ordering()

    class _OrphanDof(LagrangeInterpolation):
        @property
        def dofs(self) -> tuple[DofFunctional, ...]:
            base = list(super().dofs)
            base[1] = DofFunctional(1, 9, base[1].point, (0, 0), EntityType.VERTEX, 1)
            return tuple(base)

    with pytest.raises(NodeOrderingError, match="unknown node"):
        _OrphanDof(CellType.TRIANGLE, 1).verify_node_ordering()


def test_interpolation_nodes_query_returns_the_nodes() -> None:
    """The node query is the spec's named entry point for the same tuple."""
    element = LagrangeInterpolation(CellType.TRIANGLE, 2)
    assert element.interpolation_nodes() == element.nodes
    assert len(element.interpolation_nodes()) == 6


def test_collinear_nodes_are_distinct_yet_not_unisolvent() -> None:
    """Rank deficiency without duplication: three collinear nodes cannot span P1.

    Distinct points are necessary but not sufficient for unisolvence on a
    simplex - the classical degenerate case is nodes that all lie on one edge,
    so no member of the space can vary transverse to it. The Vandermonde's
    eta column vanishes and the rank check catches it.
    """

    class _CollinearTriangle(LagrangeInterpolation):
        @property
        def nodes(self) -> tuple[InterpolationNode, ...]:
            base = list(super().nodes)
            base[2] = InterpolationNode(2, (0.5, 0.0), EntityType.VERTEX, 2)
            return tuple(base)

    bad = _CollinearTriangle(CellType.TRIANGLE, 1)
    points = {node.coordinates for node in bad.nodes}
    assert len(points) == 3  # genuinely distinct
    with pytest.raises(UnisolvenceError, match="rank deficient"):
        bad.verify_linear_independence()


def test_verify_node_ordering_catches_a_vertex_node_at_the_wrong_vertex() -> None:
    """A node inside the domain that claims a vertex it does not sit on is caught.

    Distinct from the permutation rule above: here the numbering is orderly but
    the geometry contradicts it, so the per-node entity check is what fires.
    """

    class _MisplacedVertex(LagrangeInterpolation):
        @property
        def nodes(self) -> tuple[InterpolationNode, ...]:
            base = list(super().nodes)
            reference = self.reference_element.vertex_coordinates
            base[1] = InterpolationNode(
                1, tuple(float(x) for x in reference[2]), EntityType.VERTEX, 1
            )
            return tuple(base)

    with pytest.raises(NodeOrderingError, match="claims vertex 1 but sits at"):
        _MisplacedVertex(CellType.TRIANGLE, 1).verify_node_ordering()
