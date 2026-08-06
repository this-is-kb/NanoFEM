"""``NonlocalAxialLoad``/``NonlocalTransverseLoad`` as CELL FORCE ``ContributionProvider``s
(Stage 4, Eringen differential).

**Bar** (``NonlocalAxialLoadProvider``): two terms per element, both derived in
``docs/design/ERINGEN_DIFFERENTIAL_BAR.md``: ``integral(N_a * q_h(x)) dx`` (the classical
consistent load, identical in form to ``ContinuumBodyForceProvider``'s cell integral one
dimension down) plus ``nonlocal_parameter * integral(dN_a/dx * q_h'(x)) dx`` (the nonlocal
correction). Both ``q_h(x)`` and its derivative are read from the *same* linear shape functions
and physical gradients already used for the classical term.

**Beam** (``NonlocalTransverseLoadProvider``): the same two-term pattern one derivative order
up, derived in ``docs/design/ERINGEN_DIFFERENTIAL_BEAM.md``. The one genuinely new subtlety
(beyond the bar's own pattern): a Hermite beam's shape-function *values* need the same
reference-to-physical rescaling ``_reference_derivative_scale`` (N-53) already applies to
curvature - confirmed only by a numerical check against the real Hermite/mapping stack, which
first got a non-machine-precision ``mu=0`` residual until this correction was added (see
``docs/dev/notes.md``).

``numerics.interpolation``/``numerics.mapping``/``numerics.quadrature`` are imported *inside*
the per-family basis/quadrature builders and ``contributions()`` methods rather than at module
scope, for the same reason ``constraints/traction.py`` does (dev note N-66): this module is
reached from ``nanofem/__init__.py``'s eager top-level re-exports on *every*
``import nanofem.anything`` (via ``analysis.static``), and several ``numerics`` leaf packages'
own independence proofs assert those three are absent from a bare process that only touched
their own layer.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nanofem.constraints.loads import NonlocalAxialLoad, NonlocalTransverseLoad
from nanofem.core.dof_handler import DofHandler
from nanofem.mesh.mesh import Mesh
from nanofem.numerics.assembly.contributions import Contribution, ContributionKind, OperatorRole
from nanofem.numerics.reference.enums import CellType
from nanofem.utils.exceptions import ModelError

if TYPE_CHECKING:
    from nanofem.numerics.interpolation.shape_functions import ShapeFunctionFamily
    from nanofem.numerics.quadrature.rules import QuadratureRule

#: Exact for this provider's integrands: N_a*q_h is degree-2 (linear*linear), dN_a/dx*q_h' is
#: degree-0 (constant*constant) in reference coordinates, both covered by a 2-point Gauss rule.
_BAR_QUADRATURE_ORDER = 2

#: The beam's load integrands involve a cubic (Hermite) basis; order 6 is comfortably exact.
_BEAM_QUADRATURE_ORDER = 6


@lru_cache(maxsize=1)
def _bar_basis() -> ShapeFunctionFamily:
    """The 2-node line's own linear Lagrange basis, built once and cached."""
    from nanofem.numerics.interpolation import LagrangeInterpolation, shape_functions

    return shape_functions(LagrangeInterpolation(CellType.LINE, 1))


@lru_cache(maxsize=1)
def _bar_quadrature() -> QuadratureRule:
    """The line quadrature rule this provider's cell integrals use, built once and cached."""
    from nanofem.numerics.quadrature import quadrature

    return quadrature(CellType.LINE, order=_BAR_QUADRATURE_ORDER)


@dataclass(frozen=True, eq=False)
class NonlocalAxialLoadProvider:
    """A ``NonlocalAxialLoad`` as a CELL FORCE ``ContributionProvider`` (SDS Section 10)."""

    load: NonlocalAxialLoad
    mesh: Mesh
    dof_handler: DofHandler
    field_components: tuple[str, ...]
    factor: float = 1.0

    def contributions(self, role: OperatorRole) -> Iterator[Contribution]:
        """Yield one CELL FORCE block per ``line2`` cell in the load's region."""
        if role is not OperatorRole.FORCE:
            return
        if self.load.nodal_intensity.shape[0] != self.mesh.num_nodes:
            raise ModelError(
                f"NonlocalAxialLoad.nodal_intensity has {self.load.nodal_intensity.shape[0]} "
                f"entries, but the mesh has {self.mesh.num_nodes} nodes - one entry per node "
                "is required"
            )
        if len(self.field_components) != 1:
            raise ModelError(
                f"NonlocalAxialLoad is a 1-D axial load (one scalar component); field "
                f"'{self.load.field}' declares {len(self.field_components)} components "
                f"{self.field_components}"
            )
        from nanofem.numerics.mapping import AffineMapping

        basis = _bar_basis()
        rule = _bar_quadrature()
        values = basis.evaluate(rule.points)  # (n_qp, 2)
        mu = self.factor * self.load.nonlocal_parameter

        for cell_id in self.mesh.cells_in_region(self.load.region):
            cell = self.mesh.cell(cell_id)
            if cell.cell_type != "line2":
                raise ModelError(
                    f"NonlocalAxialLoad region '{self.load.region}': only 'line2' cells are "
                    f"supported, got '{cell.cell_type}' for cell {cell_id}"
                )
            node_ids = cell.connectivity
            coords = np.array([self.mesh.node(n).coordinates for n in node_ids])
            mapping = AffineMapping(CellType.LINE, coords)

            ref_grad = basis.derivatives(rule.points)  # (n_qp, 2, 1)
            phys_grad = mapping.physical_gradient(ref_grad, rule.points)[:, :, 0]  # (n_qp, 2)
            volume_scale = mapping.volume_scale(rule.points)

            q_local = self.load.nodal_intensity[list(node_ids)]  # (2,)
            q_vals = values @ q_local  # (n_qp,) -- q_h(x) at quadrature points
            q_prime_vals = phys_grad @ q_local  # (n_qp,) -- q_h'(x) at quadrature points

            f_classical = self.factor * np.einsum(
                "q,q,qa,q->a", rule.weights, volume_scale, values, q_vals
            )
            f_nonlocal = mu * np.einsum(
                "q,q,qa,q->a", rule.weights, volume_scale, phys_grad, q_prime_vals
            )
            block: NDArray[np.float64] = f_classical + f_nonlocal

            dofs = np.array(
                [
                    self.dof_handler.global_dof(node_id, self.load.field, component)
                    for node_id in node_ids
                    for component in self.field_components
                ],
                dtype=np.int64,
            )
            yield Contribution(ContributionKind.CELL, role, dofs, None, block)


@lru_cache(maxsize=1)
def _beam_basis() -> ShapeFunctionFamily:
    """The cubic Hermite beam basis, built once and cached."""
    from nanofem.numerics.interpolation import HermiteInterpolation, shape_functions

    return shape_functions(HermiteInterpolation(CellType.LINE, order=3))


@lru_cache(maxsize=1)
def _beam_quadrature() -> QuadratureRule:
    """The line quadrature rule this provider's cell integrals use, built once and cached."""
    from nanofem.numerics.quadrature import quadrature

    return quadrature(CellType.LINE, order=_BEAM_QUADRATURE_ORDER)


@dataclass(frozen=True, eq=False)
class NonlocalTransverseLoadProvider:
    """A ``NonlocalTransverseLoad`` as a CELL FORCE ``ContributionProvider`` (SDS Section 10).

    Unlike ``NonlocalAxialLoadProvider``/``NodalLoadProvider``/``TractionLoadProvider``, this
    provider does not take a ``field_components`` argument: a beam's consistent load vector is
    conjugate to *both* ``u.y`` and ``r.z`` at each node (the classical Hermite consistent-load
    pattern), never a single field's own component list - the DOF resolution mirrors
    ``elements/factory.py``'s own ``_bending_global_dofs`` exactly.
    """

    load: NonlocalTransverseLoad
    mesh: Mesh
    dof_handler: DofHandler
    factor: float = 1.0

    def contributions(self, role: OperatorRole) -> Iterator[Contribution]:
        """Yield one CELL FORCE block per ``line2`` cell in the load's region."""
        if role is not OperatorRole.FORCE:
            return
        if self.load.nodal_intensity.shape[0] != self.mesh.num_nodes:
            raise ModelError(
                f"NonlocalTransverseLoad.nodal_intensity has "
                f"{self.load.nodal_intensity.shape[0]} entries, but the mesh has "
                f"{self.mesh.num_nodes} nodes - one entry per node is required"
            )
        if self.load.field != "u":
            raise ModelError(
                f"NonlocalTransverseLoad targets the beam's transverse displacement field "
                f"'u'; got '{self.load.field}'"
            )
        from nanofem.numerics.mapping import AffineMapping
        from nanofem.physics.elasticity.euler_bernoulli import _reference_derivative_scale

        basis = _beam_basis()
        rule = _beam_quadrature()
        dof_orders = tuple(dof.derivative[0] for dof in basis.interpolation.dofs)  # (0,1,0,1)
        mu = self.factor * self.load.nonlocal_parameter

        for cell_id in self.mesh.cells_in_region(self.load.region):
            cell = self.mesh.cell(cell_id)
            if cell.cell_type != "line2":
                raise ModelError(
                    f"NonlocalTransverseLoad region '{self.load.region}': only 'line2' cells "
                    f"are supported, got '{cell.cell_type}' for cell {cell_id}"
                )
            node_ids = cell.connectivity
            coords = np.array([self.mesh.node(n).coordinates for n in node_ids])
            mapping = AffineMapping(CellType.LINE, coords)
            jacobian = mapping.jacobian_determinant(rule.points)
            scale = _reference_derivative_scale(dof_orders, float(jacobian[0]))

            ref_grad = basis.derivatives(rule.points)
            phys_grad = mapping.physical_gradient(ref_grad, rule.points)[:, :, 0]
            # Both the shape VALUES and their derivatives need the reference-to-physical
            # rescaling: interpolation is w(x) = sum_a N_a(x) d_a_ref with d_a_ref = scale_a *
            # d_a_physical, so the effective shape function multiplying a physical DOF is
            # N_a(x)*scale_a everywhere - not only in the (already-corrected) curvature B
            # matrix. Confirmed only by a numerical check that first got a non-machine-
            # precision residual without this (docs/dev/notes.md).
            values = basis.evaluate(rule.points) * scale
            w_prime_shapes = phys_grad * scale

            q_local = np.array(
                [
                    self.load.nodal_intensity[node_ids[0]],
                    0.0,
                    self.load.nodal_intensity[node_ids[1]],
                    0.0,
                ]
            )
            q_vals = values @ q_local
            q_prime_vals = w_prime_shapes @ q_local

            f_classical = self.factor * np.einsum(
                "q,q,qa,q->a", rule.weights, jacobian, values, q_vals
            )
            f_nonlocal = mu * np.einsum(
                "q,q,qa,q->a", rule.weights, jacobian, w_prime_shapes, q_prime_vals
            )
            block: NDArray[np.float64] = f_classical + f_nonlocal

            dofs = np.array(
                [
                    self.dof_handler.global_dof(node_ids[0], "u", "y"),
                    self.dof_handler.global_dof(node_ids[0], "r", "z"),
                    self.dof_handler.global_dof(node_ids[1], "u", "y"),
                    self.dof_handler.global_dof(node_ids[1], "r", "z"),
                ],
                dtype=np.int64,
            )
            yield Contribution(ContributionKind.CELL, role, dofs, None, block)
