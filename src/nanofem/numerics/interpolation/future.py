"""Declared placeholders for the remaining interpolation families.

Each is named now so registries, enumerations, and interfaces anticipate it,
and each carries a ``PROVISIONAL_METADATA`` mapping recording its intended
shape. Constructing one raises ``NotImplementedError`` with the reason.

These are not arbitrary deferrals. Each family needs something the phase-3
framework deliberately does not have:

- **Serendipity** needs a space that is neither ``P_k`` nor ``Q_k`` but a
  hand-specified monomial set (``S_k``), so ``PolynomialSpaceType.SERENDIPITY``
  has no dimension formula yet. Everything else - nodes on the boundary only,
  unisolvence by the same Vandermonde oracle - the framework already supports.
- **Hierarchical** needs non-nodal degrees of freedom: its DOFs are modal
  coefficients on entities, not point evaluations, so it has no interpolation
  nodes at all and ``DofKind.MOMENT`` would carry the weight. This is the
  family that makes the ``evaluation_points`` / ``node_locations`` distinction
  in the base class earn its keep.
- **Spectral** needs Gauss-Lobatto-Legendre nodes, which are the roots of a
  derivative of a Legendre polynomial - quadrature machinery, which this
  phase forbids. The family therefore *cannot* land before the quadrature
  layer, and the reason is structural rather than a matter of scheduling.
  Its motivation is already visible here:
  :meth:`Interpolation.unisolvence_condition_number` grows with order on the
  equispaced nodes the Lagrange family uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from nanofem.numerics.interpolation.base import Interpolation
from nanofem.numerics.interpolation.dofs import DofFunctional, InterpolationNode
from nanofem.numerics.interpolation.enums import InterpolationFamily
from nanofem.numerics.interpolation.polynomial import PolynomialSpace
from nanofem.numerics.operators.base import Continuity
from nanofem.numerics.reference.element import ReferenceElement
from nanofem.numerics.reference.enums import CellType


@dataclass(frozen=True, eq=False, repr=False, init=False)
class _FutureInterpolation(Interpolation):
    """Shared placeholder base: satisfies the interface but refuses construction."""

    #: Intended metadata, for documentation and as the future build target.
    PROVISIONAL_METADATA: ClassVar[dict[str, Any]] = {}

    #: Why this family cannot be built yet, in terms of what it needs.
    BLOCKED_BY: ClassVar[str] = ""

    def __init__(self, cell_type: CellType | str, order: int) -> None:
        """Refuse construction, naming what the family still needs."""
        raise NotImplementedError(
            f"{type(self).__name__} is a declared placeholder: {self.BLOCKED_BY}"
        )

    @property
    def family(self) -> InterpolationFamily:  # pragma: no cover - construction raises first
        """Not available for a placeholder."""
        raise NotImplementedError

    @property
    def order(self) -> int:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    @property
    def reference_element(self) -> ReferenceElement:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    @property
    def polynomial_space(self) -> PolynomialSpace:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    @property
    def nodes(self) -> tuple[InterpolationNode, ...]:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    @property
    def dofs(self) -> tuple[DofFunctional, ...]:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError

    @property
    def continuity(self) -> Continuity:  # pragma: no cover
        """Not available for a placeholder."""
        raise NotImplementedError


@dataclass(frozen=True, eq=False, repr=False, init=False)
class SerendipityInterpolation(_FutureInterpolation):
    """Placeholder for the serendipity family (``S_k`` spaces).

    Intended: boundary-only nodes, no interior nodes. On a quadrilateral,
    ``S_2`` has 8 nodes (the classical ``quad8``) against ``Q_2``'s 9, and
    ``S_3`` has 12 against ``Q_3``'s 16. Completeness degree ``k``, so the
    approximation rate matches ``Q_k`` at lower cost, at the price of losing
    the tensor-product structure.

    Note that the phase-0 cell registry already contains ``quad8`` - the
    serendipity cell - while ``quad9``, which Lagrange order 2 on a
    quadrilateral needs, is absent. That gap closes when this family lands.
    """

    PROVISIONAL_METADATA: ClassVar[dict[str, Any]] = {
        "family": InterpolationFamily.SERENDIPITY.value,
        "space_type": "S",
        "quadrilateral": {
            2: {"num_nodes": 8, "mesh_cell_name": "quad8"},
            3: {"num_nodes": 12, "mesh_cell_name": "quad12"},
        },
        "continuity": Continuity.C0.name,
        "is_nodal": True,
    }
    BLOCKED_BY = (
        "the S_k monomial set is hand-specified rather than generated by a rule, so "
        "PolynomialSpaceType.SERENDIPITY has no dimension formula yet; the node "
        "placement and unisolvence machinery already exist"
    )


@dataclass(frozen=True, eq=False, repr=False, init=False)
class HierarchicalInterpolation(_FutureInterpolation):
    """Placeholder for the hierarchical (modal) family used for p-refinement.

    Intended: a basis in which the order-``k`` functions are a superset of the
    order-``(k-1)`` functions, so raising the order adds functions instead of
    replacing them. Its DOFs are modal coefficients attached to vertices,
    edges, and the interior - not point evaluations - so it is the first
    family with no interpolation nodes, and ``is_nodal`` will be ``False``
    with an empty ``evaluation_points()``.
    """

    PROVISIONAL_METADATA: ClassVar[dict[str, Any]] = {
        "family": InterpolationFamily.HIERARCHICAL.value,
        "dof_kind": "moment",
        "is_nodal": False,
        "has_interpolation_nodes": False,
        "continuity": Continuity.C0.name,
    }
    BLOCKED_BY = (
        "its degrees of freedom are moments (integrals against entity modes), not point "
        "functionals; DofFunctional.apply_to_monomial would need an integration rule, "
        "which is quadrature machinery this phase excludes"
    )


@dataclass(frozen=True, eq=False, repr=False, init=False)
class SpectralInterpolation(_FutureInterpolation):
    """Placeholder for the spectral family on Gauss-Lobatto-Legendre nodes.

    Intended: the same ``P_k``/``Q_k`` spaces and the same point-value DOFs as
    Lagrange, but with nodes at the GLL points instead of equispaced ones.
    The nodes cluster near the boundary, which keeps the unisolvence matrix
    well conditioned at high order, and collocating quadrature with the nodes
    makes the mass matrix diagonal.
    """

    PROVISIONAL_METADATA: ClassVar[dict[str, Any]] = {
        "family": InterpolationFamily.SPECTRAL.value,
        "node_set": "gauss_lobatto_legendre",
        "is_nodal": True,
        "continuity": Continuity.C0.name,
        "motivation": "conditioning at high order; diagonal mass under collocation",
    }
    BLOCKED_BY = (
        "its nodes are the Gauss-Lobatto-Legendre points, which are roots of a derivative "
        "of a Legendre polynomial; that is quadrature machinery, so this family cannot "
        "precede the quadrature layer"
    )
