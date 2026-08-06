"""Element factory: dispatch (region -> theory) domain bindings to concrete elements (SDS 2.11).

Dispatch is by theory *type*, one family per theory this phase supports -
still explicit ``isinstance`` branches, not a plugin registry: a registry
buys indirection with no present benefit while the mapping from theory to
element family is one-to-one and small (four families total, per Stage 3's
own "minimal classical element library" scope - Bar, Euler-Bernoulli beam,
Timoshenko beam, T3/Q4 via ``ContinuumElement``). Reused verbatim by future
``ModalAnalysis``/``LinearBucklingAnalysis``/``TransientAnalysis`` once they
grow real ``run()`` methods, which is the actual reason this lives here
rather than inlined into ``analysis/static.py``.

Two families are closed-form (ADR-002: ``Bar``, ``EulerBernoulliBeam``,
``TimoshenkoBeam`` compute stiffness directly from raw material/section
properties, never through a ``ConstitutiveModel`` object) and one is the
general integration path (``ContinuumElement``, which *does* need a
``ConstitutiveModel`` - ``domain.constitutive`` must name one of
``model.constitutives``). ``IsotropicElasticity`` is the one theory that maps
to two different element families depending on its own ``dim``: ``dim=1``
(axial) is 1-D and closed-form (``Bar``); ``dim=2`` (plane stress/strain) is
2-D and general (``ContinuumElement``, dispatched further by mesh cell type
into T3 or Q4).

``ContinuumElement``, ``PlaneGeometry``, and ``cell_type_of_name`` are
imported *inside* :func:`_build_continuum_elements` rather than at module
scope: this module is reached from ``nanofem/__init__.py``'s eager
top-level re-exports (``analysis.static`` imports this factory) on *every*
``import nanofem.anything``, and ``ContinuumElement`` pulls in
``numerics.quadrature`` - a real dependency of the T3/Q4 path, but one that
several ``numerics`` leaf packages' own independence proofs
(``test_*_verification.py``'s "does not import quadrature/mapping" checks)
assert is absent from a bare process that only touched their own layer.
``Bar``/``EulerBernoulliBeam``/``TimoshenkoBeam`` need no such deferral -
none of the three imports quadrature (confirmed by reading their own import
lists), so eagerly loading them costs nothing and keeps the common case
simple.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nanofem.core.dof_handler import DofHandler
from nanofem.core.model import DomainDefinition, Model
from nanofem.elements.base import Element
from nanofem.elements.structural.bar import Bar
from nanofem.elements.structural.beam_eb import EulerBernoulliBeam
from nanofem.elements.structural.beam_timoshenko import TimoshenkoBeam
from nanofem.geometry.base import CrossSection
from nanofem.materials.material import Material
from nanofem.mesh.mesh import Mesh
from nanofem.physics.elasticity.eringen_differential import EringenDifferentialTheory
from nanofem.physics.elasticity.euler_bernoulli import EulerBernoulliBendingTheory
from nanofem.physics.elasticity.isotropic import IsotropicElasticity
from nanofem.physics.elasticity.timoshenko import TimoshenkoBeamTheory
from nanofem.utils.exceptions import MissingSectionError, ModelError

if TYPE_CHECKING:
    from nanofem.elements.continuum.continuum import ContinuumElement
    from nanofem.elements.continuum.nonlocal_continuum import NonlocalContinuumElement

#: Mesh cell-type names ``ContinuumElement`` supports this phase (Stage 3's "minimal
#: classical element library": T3/Q4 only - no T6, Q8, or higher-order continuum cells).
_CONTINUUM_CELL_TYPES: dict[str, int] = {"tri3": 1, "quad4": 1}


def build_elements(model: Model, dof_handler: DofHandler) -> tuple[Element, ...]:
    """Construct one element per cell, dispatched by each domain's theory type."""
    elements: list[Element] = []
    for domain in model.domains.values():
        theory = model.theories[domain.theory]
        material = model.materials[domain.material]
        if isinstance(theory, IsotropicElasticity) and theory.dim == 1:
            elements.extend(_build_bar_elements(domain, model, material, dof_handler))
        elif isinstance(theory, IsotropicElasticity) and theory.dim == 2:
            elements.extend(_build_continuum_elements(domain, model, theory, material, dof_handler))
        elif isinstance(theory, EulerBernoulliBendingTheory):
            elements.extend(_build_euler_bernoulli_elements(domain, model, material, dof_handler))
        elif isinstance(theory, TimoshenkoBeamTheory):
            elements.extend(_build_timoshenko_elements(domain, model, material, dof_handler))
        elif isinstance(theory, EringenDifferentialTheory):
            elements.extend(
                _build_nonlocal_continuum_elements(domain, model, theory, material, dof_handler)
            )
        else:
            raise ModelError(
                f"domain '{domain.name}': no element family registered for theory "
                f"{type(theory).__name__}"
            )
    return tuple(elements)


def _section_for(domain: DomainDefinition, model: Model, needs: str) -> CrossSection:
    if domain.geometry is None:
        raise MissingSectionError(f"domain '{domain.name}': {needs} requires a cross-section")
    section = model.sections[domain.geometry]
    if not isinstance(section, CrossSection):
        raise ModelError(
            f"domain '{domain.name}': {needs} needs a CrossSection geometry, got "
            f"{type(section).__name__}"
        )
    return section


def _cells_of(mesh: Mesh, domain: DomainDefinition, expected_cell_type: str) -> list[int]:
    cell_ids = list(mesh.cells_in_region(domain.region))
    for cell_id in cell_ids:
        cell_type = mesh.cell(cell_id).cell_type
        if cell_type != expected_cell_type:
            raise ModelError(
                f"domain '{domain.name}': only '{expected_cell_type}' cells are supported for "
                f"this theory, got '{cell_type}' for cell {cell_id}"
            )
    return cell_ids


def _build_bar_elements(
    domain: DomainDefinition, model: Model, material: Material, dof_handler: DofHandler
) -> list[Bar]:
    mesh = model.mesh
    section = _section_for(domain, model, "IsotropicElasticity(dim=1)/Bar")
    young_modulus = float(material.value("E"))
    area = section.area()
    elements: list[Bar] = []
    for cell_id in _cells_of(mesh, domain, "line2"):
        node_a, node_b = mesh.cell(cell_id).connectivity
        coordinates = np.array([mesh.node(node_a).coordinates, mesh.node(node_b).coordinates])
        global_dofs = (
            dof_handler.global_dof(node_a, "u", "x"),
            dof_handler.global_dof(node_b, "u", "x"),
        )
        elements.append(
            Bar(cell_id, (node_a, node_b), coordinates, global_dofs, young_modulus, area)
        )
    return elements


def _bending_global_dofs(
    dof_handler: DofHandler, node_a: int, node_b: int
) -> tuple[int, int, int, int]:
    """``(u.y, r.z)`` per node - the shared DOF layout of both beam families."""
    return (
        dof_handler.global_dof(node_a, "u", "y"),
        dof_handler.global_dof(node_a, "r", "z"),
        dof_handler.global_dof(node_b, "u", "y"),
        dof_handler.global_dof(node_b, "r", "z"),
    )


def _build_euler_bernoulli_elements(
    domain: DomainDefinition, model: Model, material: Material, dof_handler: DofHandler
) -> list[EulerBernoulliBeam]:
    mesh = model.mesh
    section = _section_for(domain, model, "EulerBernoulliBendingTheory")
    young_modulus = float(material.value("E"))
    second_moment = section.second_moment_z()
    elements: list[EulerBernoulliBeam] = []
    for cell_id in _cells_of(mesh, domain, "line2"):
        node_a, node_b = mesh.cell(cell_id).connectivity
        coordinates = np.array([mesh.node(node_a).coordinates, mesh.node(node_b).coordinates])
        global_dofs = _bending_global_dofs(dof_handler, node_a, node_b)
        elements.append(
            EulerBernoulliBeam(
                cell_id, (node_a, node_b), coordinates, global_dofs, young_modulus, second_moment
            )
        )
    return elements


def _build_timoshenko_elements(
    domain: DomainDefinition, model: Model, material: Material, dof_handler: DofHandler
) -> list[TimoshenkoBeam]:
    mesh = model.mesh
    section = _section_for(domain, model, "TimoshenkoBeamTheory")
    young_modulus = float(material.value("E"))
    shear_modulus = float(material.value("G"))
    second_moment = section.second_moment_z()
    shear_area = section.area() * section.shear_correction(float(material.value("nu")))
    elements: list[TimoshenkoBeam] = []
    for cell_id in _cells_of(mesh, domain, "line2"):
        node_a, node_b = mesh.cell(cell_id).connectivity
        coordinates = np.array([mesh.node(node_a).coordinates, mesh.node(node_b).coordinates])
        global_dofs = _bending_global_dofs(dof_handler, node_a, node_b)
        elements.append(
            TimoshenkoBeam(
                cell_id,
                (node_a, node_b),
                coordinates,
                global_dofs,
                young_modulus,
                shear_modulus,
                second_moment,
                shear_area,
            )
        )
    return elements


def _build_continuum_elements(
    domain: DomainDefinition,
    model: Model,
    theory: IsotropicElasticity,
    material: Material,
    dof_handler: DofHandler,
) -> list[ContinuumElement]:
    # Deferred imports: see the module docstring for why (numerics.quadrature independence).
    from nanofem.elements.continuum.continuum import ContinuumElement
    from nanofem.geometry.plane import PlaneGeometry
    from nanofem.numerics.reference.registry import cell_type_of_name

    mesh = model.mesh
    if domain.constitutive is None:
        raise ModelError(
            f"domain '{domain.name}': IsotropicElasticity(dim=2)/ContinuumElement requires a "
            f"constitutive law (PlaneStressConstitutive or PlaneStrainConstitutive)"
        )
    constitutive = model.constitutives[domain.constitutive]
    if domain.geometry is None:
        raise MissingSectionError(
            f"domain '{domain.name}': IsotropicElasticity(dim=2)/ContinuumElement requires a "
            f"PlaneGeometry (thickness)"
        )
    geometry = model.sections[domain.geometry]
    if not isinstance(geometry, PlaneGeometry):
        raise ModelError(
            f"domain '{domain.name}': ContinuumElement needs a PlaneGeometry section, got "
            f"{type(geometry).__name__}"
        )

    elements: list[ContinuumElement] = []
    for cell_id in mesh.cells_in_region(domain.region):
        cell = mesh.cell(cell_id)
        order = _CONTINUUM_CELL_TYPES.get(cell.cell_type)
        if order is None:
            raise ModelError(
                f"domain '{domain.name}': only {sorted(_CONTINUUM_CELL_TYPES)} cells are "
                f"supported for ContinuumElement, got '{cell.cell_type}' for cell {cell_id}"
            )
        node_ids = cell.connectivity
        coordinates = np.array([mesh.node(n).coordinates for n in node_ids])
        global_dofs: NDArray[np.int64] = np.array(
            [dof_handler.global_dof(n, "u", c) for n in node_ids for c in ("x", "y")],
            dtype=np.int64,
        )
        elements.append(
            ContinuumElement(
                cell_id=cell_id,
                node_ids=node_ids,
                coordinates=coordinates,
                global_dofs=global_dofs,
                cell_type=cell_type_of_name(cell.cell_type),
                interpolation_order=order,
                theory=theory,
                constitutive=constitutive,
                material=material,
                section_measure=geometry.thickness,
            )
        )
    return elements


def _build_nonlocal_continuum_elements(
    domain: DomainDefinition,
    model: Model,
    theory: EringenDifferentialTheory,
    material: Material,
    dof_handler: DofHandler,
) -> list[NonlocalContinuumElement]:
    # Deferred imports: see the module docstring for why (numerics.quadrature independence).
    from nanofem.elements.continuum.nonlocal_continuum import NonlocalContinuumElement
    from nanofem.geometry.plane import PlaneGeometry
    from nanofem.numerics.reference.registry import cell_type_of_name
    from nanofem.physics.elasticity.eringen_differential import (
        NONLOCAL_STRAIN_FIELD,
        voigt_component_names,
    )

    mesh = model.mesh
    if domain.constitutive is None:
        raise ModelError(
            f"domain '{domain.name}': EringenDifferentialTheory/NonlocalContinuumElement "
            f"requires a constitutive law (EringenDifferentialMaterial)"
        )
    from nanofem.physics.elasticity.eringen_differential import EringenDifferentialMaterial

    constitutive = model.constitutives[domain.constitutive]
    if not isinstance(constitutive, EringenDifferentialMaterial):
        raise ModelError(
            f"domain '{domain.name}': NonlocalContinuumElement needs an "
            f"EringenDifferentialMaterial constitutive law, got {type(constitutive).__name__}"
        )
    if domain.geometry is None:
        raise MissingSectionError(
            f"domain '{domain.name}': EringenDifferentialTheory/NonlocalContinuumElement "
            f"requires a PlaneGeometry (thickness)"
        )
    geometry = model.sections[domain.geometry]
    if not isinstance(geometry, PlaneGeometry):
        raise ModelError(
            f"domain '{domain.name}': NonlocalContinuumElement needs a PlaneGeometry section, "
            f"got {type(geometry).__name__}"
        )

    e_components = voigt_component_names(theory.dim)
    elements: list[NonlocalContinuumElement] = []
    for cell_id in mesh.cells_in_region(domain.region):
        cell = mesh.cell(cell_id)
        order = _CONTINUUM_CELL_TYPES.get(cell.cell_type)
        if order is None:
            raise ModelError(
                f"domain '{domain.name}': only {sorted(_CONTINUUM_CELL_TYPES)} cells are "
                f"supported for NonlocalContinuumElement, got '{cell.cell_type}' for cell "
                f"{cell_id}"
            )
        node_ids = cell.connectivity
        coordinates = np.array([mesh.node(n).coordinates for n in node_ids])
        global_dofs_u = [dof_handler.global_dof(n, "u", c) for n in node_ids for c in ("x", "y")]
        global_dofs_e = [
            dof_handler.global_dof(n, NONLOCAL_STRAIN_FIELD, c)
            for n in node_ids
            for c in e_components
        ]
        global_dofs: NDArray[np.int64] = np.array(global_dofs_u + global_dofs_e, dtype=np.int64)
        elements.append(
            NonlocalContinuumElement(
                cell_id=cell_id,
                node_ids=node_ids,
                coordinates=coordinates,
                global_dofs=global_dofs,
                cell_type=cell_type_of_name(cell.cell_type),
                interpolation_order=order,
                theory=theory,
                constitutive=constitutive,
                material=material,
                section_measure=geometry.thickness,
            )
        )
    return elements
