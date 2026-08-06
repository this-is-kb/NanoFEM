"""``TractionLoad`` as a FACET FORCE ``ContributionProvider``: ``integral N_a(s) t_c dS``.

Mirrors ``elements/continuum/continuum.py``'s ``ContinuumBodyForceProvider``
one dimension down: a body force integrates a shape function against a
density over a *cell*; a traction integrates a shape function against a
traction over one of that cell's *facets*. For the T3/Q4 minimal element
library (Stage 3's own scope - linear elements only), a facet is always a
2-node line, so the facet's own restriction of the parent element's linear
shape functions is exactly the 2-node line's own linear Lagrange basis (a
standard fact about P1/Q1 elements: every other node's shape function
vanishes identically on a facet it is not part of) - no restriction
operator is needed, just a fresh, independent 1-D Lagrange interpolation on
the facet's own geometry. ``AffineMapping`` already supports this directly:
a 2-node line mapped into 2-D physical space is exactly the "bar embedded in
a plane" case v0.5.0 built for ``Truss2D``, so the arc-length Jacobian
(``volume_scale``) needs no new mapping code either.

``numerics.interpolation``/``numerics.mapping``/``numerics.quadrature`` are
imported *inside* :func:`_facet_basis`/:func:`_facet_quadrature`/
:meth:`TractionLoadProvider.contributions` rather than at module scope, for
the same reason ``elements/factory.py`` defers ``ContinuumElement`` (v0.14.0,
dev note N-66): this module is reached from ``nanofem/__init__.py``'s eager
top-level re-exports on *every* ``import nanofem.anything`` (via
``analysis.static``), and several ``numerics`` leaf packages' own
independence proofs assert those three are absent from a bare process that
only touched their own layer.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nanofem.constraints.loads import TractionLoad
from nanofem.core.dof_handler import DofHandler
from nanofem.mesh.mesh import Mesh
from nanofem.numerics.assembly.contributions import Contribution, ContributionKind, OperatorRole
from nanofem.numerics.reference.enums import CellType

if TYPE_CHECKING:
    from nanofem.numerics.interpolation.shape_functions import ShapeFunctionFamily
    from nanofem.numerics.quadrature.rules import QuadratureRule

#: The facet quadrature order for the Stage-3 linear element library.
_FACET_QUADRATURE_ORDER = 2


@lru_cache(maxsize=1)
def _facet_basis() -> ShapeFunctionFamily:
    """The 2-node line's own linear Lagrange basis, built once and cached."""
    from nanofem.numerics.interpolation import LagrangeInterpolation, shape_functions

    return shape_functions(LagrangeInterpolation(CellType.LINE, 1))


@lru_cache(maxsize=1)
def _facet_quadrature() -> QuadratureRule:
    """The line quadrature rule every facet integral uses, built once and cached."""
    from nanofem.numerics.quadrature import quadrature

    return quadrature(CellType.LINE, order=_FACET_QUADRATURE_ORDER)


@dataclass(frozen=True, eq=False)
class TractionLoadProvider:
    """A ``TractionLoad`` as a FACET FORCE ``ContributionProvider`` (SDS Section 10)."""

    load: TractionLoad
    mesh: Mesh
    dof_handler: DofHandler
    field_components: tuple[str, ...]
    factor: float = 1.0

    def contributions(self, role: OperatorRole) -> Iterator[Contribution]:
        """Yield one FACET FORCE block per facet in the load's facet region."""
        if role is not OperatorRole.FORCE:
            return
        from nanofem.numerics.mapping import AffineMapping

        rule = _facet_quadrature()
        values = _facet_basis().evaluate(rule.points)
        traction = self.factor * self.load.traction
        for cell_id, local_facet_index in self.mesh.facets_in_region(self.load.region):
            node_ids = self.mesh.facet_node_ids(cell_id, local_facet_index)
            coords = np.array([self.mesh.node(n).coordinates for n in node_ids])
            mapping = AffineMapping(CellType.LINE, coords)
            volume_scale = mapping.volume_scale(rule.points)
            shape_integral = np.einsum("q,q,qa->a", rule.weights, volume_scale, values)
            block: NDArray[np.float64] = np.kron(shape_integral, traction)
            dofs = np.array(
                [
                    self.dof_handler.global_dof(node_id, self.load.field, component)
                    for node_id in node_ids
                    for component in self.field_components
                ],
                dtype=np.int64,
            )
            yield Contribution(ContributionKind.FACET, role, dofs, None, block)
