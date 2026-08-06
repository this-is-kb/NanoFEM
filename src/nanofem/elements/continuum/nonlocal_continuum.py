"""Mixed (displacement, nonlocal-strain) continuum element for Eringen differential elasticity.

Where ``ContinuumElement`` composes a *single* field's kinematics against a pointwise
constitutive law, ``NonlocalContinuumElement`` composes *two* coupled fields - a displacement
field ``u`` and an auxiliary nonlocal-strain field ``e*`` - against the symmetric weak form
derived and verified in ``physics/elasticity/eringen_differential.py``:

    K_ue . e* = f                            (equilibrium, tested with delta_u)
    K_eu . u  = K_ee . e*,   K_eu = K_ue^T   (Helmholtz relation, tested with D:delta_e*)

giving the coupled block system ``[[0, K_ue], [K_eu, -K_ee]] [u; e*] = [f; 0]`` - the standard
KKT/Stokes-type saddle-point sign pattern (zero diagonal block on ``u``, *negative* diagonal
block on ``e*``, from moving ``K_eu.u`` to the same side as ``-K_ee.e*``). Getting this sign
right needed two rounds of verification: a first check (symmetry, and the ``mu=0``/
constant-strain reductions via a *substitution* ``e* = K_ee^-1 K_eu u``) passed but is
insensitive to this exact sign, since a substitution never assembles the monolithic matrix; a
follow-up full ``Model``-pipeline solve caught a sign-flipped displacement (right magnitude,
wrong sign) that only the ``+K_ee`` version produced - fixed here, and re-verified via the same
full pipeline. See ``docs/dev/notes.md`` for the complete account. With the correct sign, the
block system reduces exactly to the classical stiffness at ``mu=0`` for a constant-strain
element, is symmetric by construction, and has a symmetric-positive-semi-definite Schur
complement with the correct three 2-D rigid-body zero-energy modes.

This element is deliberately *not* named after Eringen: the pattern (a displacement field
coupled to one auxiliary tensor field through a Helmholtz-type relation) is generic enough to
serve Strain Gradient or similar theories later, without redesigning this class - only a new
``Theory``/``ConstitutiveModel`` pair would be needed, matching the project's "theory-independent
backbone" design philosophy. What *is* Eringen-specific is confined to
``physics/elasticity/eringen_differential.py``.

DOF layout: ``global_dofs`` is the concatenation of ``u``'s node-major-then-component block
followed by ``e*``'s node-major-then-component block (not a single per-node SDS C-2
interleaving of both fields) - a deliberate, documented simplification, since
``dof_signature()``/``ElementDofSignature`` has no consumers anywhere in the codebase to date
(confirmed by grep before this class was written) and per-node interleaving of two
differently-shaped fields would add real implementation complexity for no current benefit.
Correctness does not depend on this choice: the ``Assembler`` only requires that
``local_stiffness()``'s rows/columns line up 1:1 with ``global_dofs``, in whatever order that is.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from nanofem.core.fields import component_names
from nanofem.elements.base import Element, ElementDofSignature
from nanofem.materials.material import Material
from nanofem.numerics.assembly.contributions import Contribution, ContributionKind, OperatorRole
from nanofem.numerics.interpolation import LagrangeInterpolation, shape_functions
from nanofem.numerics.mapping import (
    AffineMapping,
    GeometricMapping,
    IsoparametricMapping,
    NonAffineError,
)
from nanofem.numerics.operators.symmetric_gradient import symmetric_gradient_matrix
from nanofem.numerics.quadrature import quadrature
from nanofem.numerics.reference.enums import CellType
from nanofem.numerics.tensors.conventions import VOIGT_LENGTH
from nanofem.physics.elasticity.eringen_differential import (
    EringenDifferentialMaterial,
    EringenDifferentialTheory,
)
from nanofem.utils.exceptions import InputValidationError


def _block_expand(values: NDArray[np.float64], n_components: int) -> NDArray[np.float64]:
    """``(n_qp, n_fun) -> (n_qp, n_components, n_fun*n_components)``, block-diagonal per node.

    The same ``kron(scalar, I)`` block-expansion ``ContinuumElement.local_mass()`` already
    uses (dev note N-51), generalized from ``I_dim`` to ``I_n_components`` - here building a
    vector shape-function matrix for a multi-component field from its scalar shape values.
    """
    eye = np.eye(n_components)
    n_qp, n_fun = values.shape
    expanded = np.einsum("qa,vw->qvaw", values, eye)
    return np.asarray(expanded.reshape(n_qp, n_components, n_fun * n_components), dtype=np.float64)


class NonlocalContinuumElement(Element):
    """A dimension-generic mixed element: displacement + nonlocal-strain, Helmholtz-coupled."""

    def __init__(
        self,
        cell_id: int,
        node_ids: tuple[int, ...],
        coordinates: NDArray[np.float64],
        global_dofs: NDArray[np.int64],
        cell_type: CellType | str,
        interpolation_order: int,
        theory: EringenDifferentialTheory,
        constitutive: EringenDifferentialMaterial,
        material: Material,
        quadrature_order: int | None = None,
        section_measure: float = 1.0,
    ) -> None:
        self.cell_id = cell_id
        self.node_ids = node_ids
        self.global_dofs = global_dofs
        self.section_measure = section_measure
        self._theory = theory
        self._constitutive = constitutive
        self._material = material

        interpolation = LagrangeInterpolation(cell_type, interpolation_order)
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.shape[0] != interpolation.num_nodes:
            raise InputValidationError(
                f"coordinates must have one row per interpolation node "
                f"({interpolation.num_nodes}), got {coords.shape[0]}"
            )
        basis = shape_functions(interpolation)
        num_vertices = interpolation.reference_element.num_vertices
        mapping: GeometricMapping
        try:
            mapping = AffineMapping(cell_type, coords[:num_vertices])
        except NonAffineError:
            mapping = IsoparametricMapping(basis, coords)

        dim = mapping.embedding_dimension
        if dim != theory.dim:
            raise InputValidationError(
                f"NonlocalContinuumElement: theory dim={theory.dim} but the mapping's "
                f"embedding dimension is {dim}"
            )
        n_voigt = VOIGT_LENGTH[dim]
        n_fun = interpolation.num_nodes
        n_dof_u = n_fun * dim
        n_dof_e = n_fun * n_voigt
        self._n_dof_u = n_dof_u
        self._n_dof_e = n_dof_e
        self._dim = dim
        self._n_voigt = n_voigt

        rule = quadrature(cell_type, order=quadrature_order or 2 * interpolation_order)
        reference_gradients = basis.derivatives(rule.points)
        physical_gradients = mapping.physical_gradient(reference_gradients, rule.points)
        values = basis.evaluate(rule.points)
        weights = rule.weights
        volume_scale = mapping.volume_scale(rule.points)

        b_u = symmetric_gradient_matrix(physical_gradients).reshape(
            rule.points.shape[0], n_voigt, n_dof_u
        )
        n_e = _block_expand(values, n_voigt)
        # grad_e[q, v, k, dof]: the k-th spatial derivative feeding e*'s v-th voigt component.
        grad_e = np.einsum("qak,vw->qvkaw", physical_gradients, np.eye(n_voigt)).reshape(
            rule.points.shape[0], n_voigt, dim, n_dof_e
        )
        self._b_u = b_u
        self._n_e = n_e
        self._values = values
        self._weights = weights
        self._volume_scale = volume_scale

        # D matrix via the same zero-strain-probe pattern ContinuumElement uses.
        n_qp = rule.points.shape[0]
        local_law = constitutive.local_law
        zero_strain = np.zeros((1, n_qp, local_law.response_components()))
        local_properties = {
            key: np.full((1, n_qp), material.value(key)) for key in local_law.required_properties()
        }
        _, tangent = local_law.respond_batch(zero_strain, local_properties)
        d_matrix = tangent[0, 0]  # constant across quadrature points for a linear elastic law
        self._d_matrix = d_matrix
        # e0a is Eringen's own canonical characteristic-length parameter (SDS Section 6,
        # "0 = local limit, deliberately legal"); mu = (e0a)^2 is what the Helmholtz relation
        # actually uses (numerics/operators/helmholtz.py's own length_scale**2 convention).
        e0a = float(material.value("e0a"))
        mu = e0a**2
        self._mu = mu

        k_ue = self.section_measure * np.einsum(
            "q,q,qvi,vw,qwj->ij", weights, volume_scale, b_u, d_matrix, n_e
        )
        k_ee_mass = self.section_measure * np.einsum(
            "q,q,qvi,vw,qwj->ij", weights, volume_scale, n_e, d_matrix, n_e
        )
        k_ee_diff = self.section_measure * np.einsum(
            "q,q,qvki,vw,qwkj->ij", weights, volume_scale, grad_e, d_matrix, grad_e
        )
        k_ee = k_ee_mass + mu * k_ee_diff
        k_eu = k_ue.T  # exact transpose by construction - guarantees global symmetry

        # The saddle-point sign: row "u" is K_ue.e* = f (equilibrium); row "e*" is
        # K_eu.u - K_ee.e* = 0 (the Helmholtz relation, K_ee.e* = K_eu.u rearranged) - the
        # standard [[0, B^T], [B, -C]] KKT/Stokes-type sign pattern. Placing +K_ee here
        # instead is a genuine sign bug caught only by solving the *monolithic* system
        # directly (a substitution-based e* = K_ee^-1 K_eu u check, done first, is
        # insensitive to this sign and did not catch it - see docs/dev/notes.md).
        n_dof = n_dof_u + n_dof_e
        k_e = np.zeros((n_dof, n_dof), dtype=np.float64)
        k_e[:n_dof_u, n_dof_u:] = k_ue
        k_e[n_dof_u:, :n_dof_u] = k_eu
        k_e[n_dof_u:, n_dof_u:] = -k_ee
        self._k_e = k_e

    def local_stiffness(self) -> NDArray[np.float64]:
        """The full ``(n_dof_u + n_dof_e)``-square coupled block stiffness matrix."""
        return self._k_e

    def quadrature_point_response(
        self, local_displacement: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """``(nonlocal_strain, nonlocal_stress, point_measure)``, one row per quadrature point.

        ``local_displacement`` is this element's own full local DOF vector (``u`` block then
        ``e*`` block, matching :attr:`global_dofs`) - typically
        ``global_solution[self.global_dofs]`` after a solve. ``nonlocal_stress = D @
        nonlocal_strain`` via the wrapped local law, exactly as ``ContinuumElement``'s own
        recovery does for the classical strain.
        """
        d = np.asarray(local_displacement, dtype=np.float64)
        n_dof = self._n_dof_u + self._n_dof_e
        if d.shape != (n_dof,):
            raise InputValidationError(
                f"local_displacement must have shape ({n_dof},), got {d.shape}"
            )
        e_star_dofs = d[self._n_dof_u :]
        nonlocal_strain = np.einsum("qva,a->qv", self._n_e, e_star_dofs)
        nonlocal_stress = np.einsum("vw,qw->qv", self._d_matrix, nonlocal_strain)
        point_measure = self._weights * self._volume_scale
        return nonlocal_strain, nonlocal_stress, point_measure

    def measure(self) -> float:
        """``section_measure * sum_q w_q J_q``: this element's area/volume."""
        return float(self.section_measure * np.sum(self._weights * self._volume_scale))

    def dof_signature(self) -> ElementDofSignature:
        """One entry per node, per field-block (SDS E-1) - see the module docstring for why
        this element uses field-block rather than fully-interleaved DOF ordering."""
        u_components = component_names(self._dim)
        from nanofem.physics.elasticity.eringen_differential import voigt_component_names

        e_components = voigt_component_names(self._dim)
        u_names = tuple(f"u.{c}" for c in u_components)
        e_names = tuple(f"e_star.{c}" for c in e_components)
        return ElementDofSignature(
            tuple(u_names for _ in self.node_ids) + tuple(e_names for _ in self.node_ids)
        )

    def contributions(self, role: OperatorRole) -> Iterator[Contribution]:
        """Emit the coupled CELL STIFFNESS block; nothing for any other role (E-6)."""
        if role is not OperatorRole.STIFFNESS:
            return
        yield Contribution(
            ContributionKind.CELL, role, self.global_dofs, self.global_dofs, self.local_stiffness()
        )
