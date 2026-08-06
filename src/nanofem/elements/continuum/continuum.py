"""Dimension-generic continuum element (ADR-011): the general integration path.

Where ``Bar`` is the closed-form ADR-002 exception - its stiffness *is* the
discrete weak form, no shape functions or quadrature involved -
``ContinuumElement`` is the general composition SDS clause E-5 describes: it
requests its theory's kinematic operator and supplies the tabulated batch,
combining ``numerics.interpolation`` + ``numerics.mapping`` +
``numerics.quadrature`` + ``numerics.operators`` + a ``Theory``/
``ConstitutiveModel``/``Material`` into local element matrices only - no
global assembly, which stays ``numerics.assembly``'s job.

The one genuinely new decision this module makes, deferred by every prior
numerics phase (``docs/design/OPERATORS.md``, ``docs/dev/notes.md`` N-42):
flattening an operator's ``(n_qp, rows, n_fun, dim)`` output into the
``(rows, n_dof)`` B-matrix SDS Section 8's notation anticipates, using the
node-major-then-component DOF order SDS C-2 specifies. That flatten turns out
to be a pure ``reshape`` - see :func:`_flatten_b_matrix`.

Scope: single-field theories only (one ``(name, n_components)`` entry from
``theory.fields()``, with ``n_components`` equal to the mapping's embedding
dimension). Geometry is mapped affinely (``AffineMapping``) whenever the
vertex correspondence admits an exact affine fit - always true for a
simplex (``Bar``, T3), true for a parallelogram Q4 - and falls back to
``IsoparametricMapping`` (built from the same tabulated shape-function
basis the field already uses) only when ``AffineMapping`` itself raises
``NonAffineError`` for a genuinely non-parallelogram quadrilateral (v0.13.0).
Every downstream quantity (``physical_gradient``, ``volume_scale``) is read
through ``GeometricMapping``'s shared, per-quadrature-point interface, so
nothing past mapping construction needs to know which concrete mapping is
in play - confirmed by v0.13.0's Q4 patch test reproducing the same
constant-strain/rigid-body results as T3's, with no branch in the
integration code itself.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.constraints.loads import BodyForce
from nanofem.core.fields import component_names
from nanofem.elements.base import Element, ElementDofSignature
from nanofem.materials.material import Material
from nanofem.mesh.mesh import Mesh
from nanofem.numerics.assembly.contributions import Contribution, ContributionKind, OperatorRole
from nanofem.numerics.interpolation import LagrangeInterpolation, shape_functions
from nanofem.numerics.mapping import (
    AffineMapping,
    GeometricMapping,
    IsoparametricMapping,
    NonAffineError,
)
from nanofem.numerics.operators.helmholtz import mass_term
from nanofem.numerics.operators.symmetric_gradient import symmetric_gradient_matrix
from nanofem.numerics.quadrature import quadrature
from nanofem.numerics.reference.enums import CellType
from nanofem.physics.base import ConstitutiveModel, Theory
from nanofem.utils.exceptions import InputValidationError


def _flatten_b_matrix(b: NDArray[np.float64], n_dof: int) -> NDArray[np.float64]:
    """``(n_qp, n_voigt, n_fun, dim) -> (n_qp, n_voigt, n_dof)``, node-major-then-component.

    Local dof index ``d = a * dim + c`` is exactly what a C-order ``reshape``
    gives when merging the trailing ``(n_fun, dim)`` axes into one - no data
    movement, no ``einsum``.
    """
    n_qp, n_voigt, n_fun, dim = b.shape
    if n_fun * dim != n_dof:
        raise InputValidationError(
            f"cannot flatten a ({n_fun}, {dim}) B-matrix into {n_dof} local dofs"
        )
    return b.reshape(n_qp, n_voigt, n_dof)


def _integrate(
    weights: NDArray[np.float64],
    volume_scale: NDArray[np.float64],
    integrand: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``sum_q w_q * J_q * integrand[q, ...]``, reducing the leading quadrature axis."""
    n_qp = weights.shape[0]
    if volume_scale.shape != (n_qp,) or integrand.shape[0] != n_qp:
        raise InputValidationError(
            f"weights {weights.shape}, volume_scale {volume_scale.shape}, and integrand's "
            f"leading axis {integrand.shape[0]} must all agree on n_qp"
        )
    return np.asarray(
        np.einsum("q,q,q...->...", weights, volume_scale, integrand), dtype=np.float64
    )


class ContinuumElement(Element):
    """A dimension-generic continuum element: shape functions + mapping + quadrature + operators."""

    def __init__(
        self,
        cell_id: int,
        node_ids: tuple[int, ...],
        coordinates: NDArray[np.float64],
        global_dofs: NDArray[np.int64],
        cell_type: CellType | str,
        interpolation_order: int,
        theory: Theory,
        constitutive: ConstitutiveModel,
        material: Material,
        field_name: str = "u",
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
        self._field_name = field_name

        fields = dict(theory.fields())
        if list(fields) != [field_name]:
            raise InputValidationError(
                f"ContinuumElement supports single-field theories only; theory declares "
                f"fields {tuple(fields)}, expected exactly ('{field_name}',)"
            )
        n_components = fields[field_name]
        self._components = component_names(n_components)

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
        if dim != n_components:
            raise InputValidationError(
                f"field '{field_name}' declares {n_components} components but the mapping's "
                f"embedding dimension is {dim}; a displacement-type field's component count "
                f"must match the space it is embedded in"
            )
        self._dim = dim
        n_fun = interpolation.num_nodes
        n_dof = n_fun * dim

        rule = quadrature(cell_type, order=quadrature_order or 2 * interpolation_order)
        reference_gradients = basis.derivatives(rule.points)
        physical_gradients = mapping.physical_gradient(reference_gradients, rule.points)
        self._values = basis.evaluate(rule.points)
        self._weights = rule.weights
        self._volume_scale = mapping.volume_scale(rule.points)
        self._b_flat = _flatten_b_matrix(symmetric_gradient_matrix(physical_gradients), n_dof=n_dof)

        n_eps = constitutive.response_components()
        n_qp = rule.points.shape[0]
        strains = np.zeros((1, n_qp, n_eps))
        properties = {
            key: np.full((1, n_qp), material.value(key))
            for key in constitutive.required_properties()
        }
        _, tangent = constitutive.respond_batch(strains, properties)
        d_tangent = tangent[0]

        per_point = np.einsum("qvi,qvw,qwj->qij", self._b_flat, d_tangent, self._b_flat)
        self._k_e = self.section_measure * _integrate(self._weights, self._volume_scale, per_point)
        self._shape_integral = _integrate(self._weights, self._volume_scale, self._values)
        self._m_e: NDArray[np.float64] | None = None

    def local_stiffness(self) -> NDArray[np.float64]:
        """``K_e = section_measure * sum_q w_q J_q B_q^T D_q B_q``."""
        return self._k_e

    def local_mass(self) -> NDArray[np.float64]:
        """``M_e = section_measure * rho * kron(mass_term(N), I_dim)``.

        ``rho`` is looked up lazily here (not at construction) and cached on
        first call, so a material that never defines it still builds a valid
        element as long as MASS is never requested.
        """
        if self._m_e is None:
            rho = float(self._material.value("rho"))
            mass_scalar = mass_term(self._values, self._weights, self._volume_scale)
            m_e = self.section_measure * rho * np.kron(mass_scalar, np.eye(self._dim))
            self._m_e = np.asarray(m_e, dtype=np.float64)
        return self._m_e

    def local_body_force(self, density: NDArray[np.float64]) -> NDArray[np.float64]:
        """``f_e = section_measure * kron(shape_integral, density)``, shape ``(n_dof,)``."""
        density_array = np.asarray(density, dtype=np.float64)
        if density_array.shape != (self._dim,):
            raise InputValidationError(
                f"body-force density must have shape ({self._dim},), got {density_array.shape}"
            )
        f_e = self.section_measure * np.kron(self._shape_integral, density_array)
        return np.asarray(f_e, dtype=np.float64)

    def quadrature_point_response(
        self, local_displacement: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """``(strain, stress, point_measure)``, one row per quadrature point.

        ``local_displacement`` is this element's own local DOF vector - ordered
        exactly as :attr:`global_dofs`, node-major-then-component (SDS C-2) -
        typically ``global_displacements[self.global_dofs]`` after a solve.
        ``strain``/``stress`` are kinematic/kinetic Voigt vectors; ``point_measure``
        (``w_q J_q``, shape ``(n_qp,)``) is the integration weight each point
        carries, letting a caller build the quadrature-weighted element-average
        via ``np.average(strain, axis=0, weights=point_measure)`` rather than a
        plain arithmetic mean. Reuses the same tabulated ``B`` matrix and
        ``Constitutive``/``Material`` already held for :meth:`local_stiffness`,
        so this is a cheap post-solve query, not a re-derivation (SDS 2.19's
        post-processing stage consuming this element layer's own cached data).
        """
        u = np.asarray(local_displacement, dtype=np.float64)
        n_dof = self._b_flat.shape[2]
        if u.shape != (n_dof,):
            raise InputValidationError(
                f"local_displacement must have shape ({n_dof},), got {u.shape}"
            )
        strain = np.einsum("qvd,d->qv", self._b_flat, u)
        n_qp = strain.shape[0]
        properties = {
            key: np.full((1, n_qp), self._material.value(key))
            for key in self._constitutive.required_properties()
        }
        stress, _ = self._constitutive.respond_batch(strain[np.newaxis, ...], properties)
        point_measure = self._weights * self._volume_scale
        return strain, stress[0], point_measure

    def measure(self) -> float:
        """``section_measure * sum_q w_q J_q``: this element's length/area/volume."""
        return float(self.section_measure * np.sum(self._weights * self._volume_scale))

    def dof_signature(self) -> ElementDofSignature:
        """One entry per node, naming ``field.component`` in field-declaration order (E-1)."""
        names = tuple(f"{self._field_name}.{c}" for c in self._components)
        return ElementDofSignature(tuple(names for _ in self.node_ids))

    def contributions(self, role: OperatorRole) -> Iterator[Contribution]:
        """Emit STIFFNESS always; MASS only if the theory declares that role (E-6)."""
        if role is OperatorRole.STIFFNESS:
            yield Contribution(
                ContributionKind.CELL,
                role,
                self.global_dofs,
                self.global_dofs,
                self.local_stiffness(),
            )
            return
        if role is OperatorRole.MASS and OperatorRole.MASS in self._theory.operator_roles():
            yield Contribution(
                ContributionKind.CELL, role, self.global_dofs, self.global_dofs, self.local_mass()
            )
            return
        return


@dataclass(frozen=True, eq=False)
class ContinuumBodyForceProvider:
    """A ``BodyForce`` as a CELL FORCE ``ContributionProvider``, mirroring ``NodalLoadProvider``.

    Lives here rather than in ``constraints/loads.py`` because it needs
    already-built ``ContinuumElement``s to reuse their tabulated quadrature
    and mapping data - ``constraints`` sits below ``elements`` in the layer
    contract and cannot hold a reference the other way.
    """

    body_force: BodyForce
    mesh: Mesh
    elements_by_cell: Mapping[int, ContinuumElement]

    def contributions(self, role: OperatorRole) -> Iterator[Contribution]:
        """Yield one CELL FORCE block per cell in the body force's region."""
        if role is not OperatorRole.FORCE:
            return
        for cell_id in self.mesh.cells_in_region(self.body_force.region):
            element = self.elements_by_cell[cell_id]
            block = element.local_body_force(self.body_force.density)
            yield Contribution(ContributionKind.CELL, role, element.global_dofs, None, block)
