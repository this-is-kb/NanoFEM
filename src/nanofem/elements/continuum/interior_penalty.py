"""Interior-penalty facet element: couples a discontinuous nonlocal strain across one facet.

With ``EringenDifferentialTheory(nonlocal_strain_continuity=Continuity.DISCONTINUOUS)`` each
cell owns its own e*. The cell integrals are exactly ``NonlocalContinuumElement``'s; what the
broken space gives up - continuity of e* across cells, which the Helmholtz gradient term needs
when ``mu > 0`` - is imposed weakly on every interior facet F shared by cells T+ and T- by the
symmetric interior penalty (SIPG) form

    a_F(e, de) = mu t int_F ( - {d_n de} . D [[e]] - [[de]] . D {d_n e} + eta_F [[de]] . D [[e]] )

with ``[[e]] = e+ - e-``, ``{d_n e} = (grad e+ + grad e-) . n / 2``, ``n`` the unit normal
pointing out of T+, ``t`` the section thickness, and ``eta_F = sigma |F| / min(|T+|, |T-|)``.
Boundary facets carry no term: e* has natural boundary conditions only. The e*-e* operator
enters the block system as ``-K_ee`` (see ``nonlocal_continuum.py``), so this element emits
``-a_F`` on the concatenated e* DOFs of T+ and T-.

This is a genuine element of the mesh's facet skeleton, not a cell element, so it emits
``ContributionKind.FACET``; the assembler and sparsity pattern accept it unchanged because a
contribution carries its own global row/column DOFs.

Penalty: ``DEFAULT_INTERIOR_PENALTY = 3``. On the T3/Q4 meshes studied, the SIPG operator was
positive semi-definite, with exactly the constant null space, for every sigma >= 2; sigma = 3
keeps a margin while giving grid-scale damping closest to the continuum's, and resolved
quantities (strain energy) changed by less than the discretization error across sigma in
[3, 40].
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.elements.base import Element, ElementDofSignature
from nanofem.numerics.assembly.contributions import Contribution, ContributionKind, OperatorRole
from nanofem.numerics.interpolation import LagrangeInterpolation, shape_functions
from nanofem.numerics.mapping import (
    AffineMapping,
    GeometricMapping,
    IsoparametricMapping,
    NonAffineError,
)
from nanofem.numerics.quadrature import quadrature
from nanofem.numerics.reference.enums import CellType
from nanofem.utils.exceptions import InputValidationError

#: Dimensionless SIPG penalty sigma (see the module docstring for how it was chosen).
DEFAULT_INTERIOR_PENALTY = 3.0


@dataclass(frozen=True)
class FacetSide:
    """One cell's view of a shared facet: its geometry and its own e* DOFs.

    ``e_dofs`` are node-major then component, matching ``NonlocalContinuumElement``'s e* block;
    ``measure`` is the cell's area (no thickness factor).
    """

    cell_id: int
    cell_type: CellType
    node_ids: tuple[int, ...]
    coordinates: NDArray[np.float64]
    e_dofs: NDArray[np.int64]
    measure: float


def _side_tabulation(
    side: FacetSide, facet_nodes: tuple[int, int], s: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Values ``(n_pts, n_fun)``, physical gradients ``(n_pts, n_fun, dim)`` and physical points
    ``(n_pts, dim)`` at facet parameters ``s`` in [0, 1], measured from ``facet_nodes[0]``."""
    interpolation = LagrangeInterpolation(side.cell_type, 1)
    basis = shape_functions(interpolation)
    reference = interpolation.reference_element
    try:
        local = [side.node_ids.index(n) for n in facet_nodes]
    except ValueError:
        raise InputValidationError(
            f"cell {side.cell_id} does not contain facet nodes {facet_nodes}"
        ) from None
    vertices = reference.vertex_coordinates
    points = vertices[local[0]] + np.outer(s, vertices[local[1]] - vertices[local[0]])

    coords = np.asarray(side.coordinates, dtype=np.float64)
    mapping: GeometricMapping
    try:
        mapping = AffineMapping(side.cell_type, coords[: reference.num_vertices])
    except NonAffineError:
        mapping = IsoparametricMapping(basis, coords)
    values = basis.evaluate(points)
    gradients = mapping.physical_gradient(basis.derivatives(points), points)
    physical = values @ coords
    return values, gradients, physical


def _voigt_block(values: NDArray[np.float64], n_voigt: int) -> NDArray[np.float64]:
    """``(n_pts, n_fun) -> (n_pts, n_voigt, n_fun*n_voigt)``, component v of node a at a*nv+v."""
    expanded = np.einsum("pa,vw->pvaw", values, np.eye(n_voigt))
    return np.asarray(expanded.reshape(values.shape[0], n_voigt, -1), dtype=np.float64)


class NonlocalInteriorPenaltyFacet(Element):
    """SIPG coupling of a discontinuous e* across one interior facet (two neighbouring cells)."""

    def __init__(
        self,
        facet_nodes: tuple[int, int],
        plus: FacetSide,
        minus: FacetSide,
        constitutive_matrix: NDArray[np.float64],
        helmholtz_coefficient: float,
        section_measure: float,
        penalty: float = DEFAULT_INTERIOR_PENALTY,
        quadrature_order: int = 2,
    ) -> None:
        if penalty <= 0.0:
            raise InputValidationError(f"interior penalty must be positive, got {penalty}")
        self.facet_nodes = facet_nodes
        self.plus = plus
        self.minus = minus
        d = np.asarray(constitutive_matrix, dtype=np.float64)
        n_voigt = d.shape[0]

        rule = quadrature(CellType.LINE, order=quadrature_order)
        s = 0.5 * (rule.points[:, 0] + 1.0)  # reference line is [-1, 1]
        n_p, g_p, x_p = _side_tabulation(plus, facet_nodes, s)
        n_m, g_m, x_m = _side_tabulation(minus, facet_nodes, s)
        if not np.allclose(x_p, x_m, rtol=0.0, atol=1e-12 * max(1.0, float(np.abs(x_p).max()))):
            raise InputValidationError(
                f"facet {facet_nodes}: cells {plus.cell_id} and {minus.cell_id} disagree on its "
                f"geometry"
            )

        endpoints = np.asarray(plus.coordinates)[[plus.node_ids.index(n) for n in facet_nodes]]
        tangent = endpoints[1] - endpoints[0]
        length = float(np.linalg.norm(tangent))
        normal = np.array([tangent[1], -tangent[0]]) / length
        centroid = np.asarray(plus.coordinates).mean(axis=0)
        if (centroid - endpoints[0]) @ normal > 0.0:
            normal = -normal  # out of T+
        weights = 0.5 * length * rule.weights

        jump = np.concatenate([_voigt_block(n_p, n_voigt), -_voigt_block(n_m, n_voigt)], axis=2)
        average = 0.5 * np.concatenate(
            [_voigt_block(g_p @ normal, n_voigt), _voigt_block(g_m @ normal, n_voigt)], axis=2
        )
        eta = penalty * length / min(plus.measure, minus.measure)
        consistency = np.einsum("p,pvi,vw,pwj->ij", weights, average, d, jump)
        stabilization = np.einsum("p,pvi,vw,pwj->ij", weights, jump, d, jump)
        sipg = -(consistency + consistency.T) + eta * stabilization

        self._dofs = np.concatenate([plus.e_dofs, minus.e_dofs]).astype(np.int64)
        self._block: NDArray[np.float64] = np.asarray(
            -(helmholtz_coefficient * section_measure) * sipg, dtype=np.float64
        )
        self._n_voigt = n_voigt

    def local_stiffness(self) -> NDArray[np.float64]:
        """The ``-mu t a_F`` block on ``[e*(T+); e*(T-)]``, symmetric by construction."""
        return self._block

    @property
    def global_dofs(self) -> NDArray[np.int64]:
        """The e* DOFs of T+ followed by those of T-."""
        return self._dofs

    def dof_signature(self) -> ElementDofSignature:
        """One e* entry per node of T+, then of T- (cell-owned: a shared node appears twice)."""
        names = tuple(f"e_star.{v}" for v in range(self._n_voigt))
        n_nodes = len(self.plus.node_ids) + len(self.minus.node_ids)
        return ElementDofSignature(tuple(names for _ in range(n_nodes)))

    def contributions(self, role: OperatorRole) -> Iterator[Contribution]:
        """Emit the FACET STIFFNESS block; nothing for any other role."""
        if role is not OperatorRole.STIFFNESS:
            return
        yield Contribution(ContributionKind.FACET, role, self._dofs, self._dofs, self._block)
