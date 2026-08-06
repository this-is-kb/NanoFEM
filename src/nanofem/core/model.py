"""Model: the central declarative object (SDS 2.18; Section 1 stages 1-3).

Holds mesh, materials, section geometries, theories, domain definitions,
boundary conditions, loads, and constraints; validates completeness and
physical consistency (property requirements!) before any analysis; and
materializes FieldSpec instances from theory field requirements (dev note
N-4). No matrix exists here, ever.

Layering note (dev note N-9): core sits below constraints, so the Model
cannot import constraint classes. It holds them behind structural Protocols
declaring exactly the shape validation needs - the phase-1 seam discovery.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from nanofem.core.dof_handler import DofHandler
from nanofem.core.fields import FieldSpec, component_names
from nanofem.geometry.base import CrossSection
from nanofem.geometry.plane import PlaneGeometry
from nanofem.materials.material import Material
from nanofem.mesh.mesh import Mesh
from nanofem.numerics.operators.base import Continuity
from nanofem.physics.base import ConstitutiveModel, Theory
from nanofem.utils.exceptions import MissingPropertyError, ModelError
from nanofem.utils.validation import require_identifier

#: A domain's geometry binding: a structural cross-section (Bar/beam families) or a
#: plane-continuum thickness carrier (T3/Q4 families) - the two geometry records SDS 2.2 defines.
DomainGeometry = CrossSection | PlaneGeometry


@runtime_checkable
class ConstraintLike(Protocol):
    """Shape the Model needs from a boundary condition (structural typing, N-9)."""

    @property
    def region(self) -> str: ...

    @property
    def field(self) -> str: ...


@runtime_checkable
class LoadCaseLike(Protocol):
    """Shape the Model needs from a load case."""

    @property
    def name(self) -> str: ...


@dataclass(frozen=True)
class DomainDefinition:
    """Bind a cell region to (theory, material[, section geometry][, constitutive]) by name.

    Elements consume these bindings; region-level assignment is canonical
    (dev note N-8). ``constitutive`` is required only for element families
    that take a separate ``ConstitutiveModel`` object rather than computing
    their stiffness closed-form from raw material properties - today that is
    ``ContinuumElement`` (T3/Q4) alone; ``Bar``/``EulerBernoulliBeam``/
    ``TimoshenkoBeam`` are all ADR-002 closed-form elements and leave it unset.
    """

    name: str
    region: str
    theory: str
    material: str
    geometry: str | None = None
    constitutive: str | None = None

    def __post_init__(self) -> None:
        require_identifier(self.name, "domain name")
        require_identifier(self.region, "domain region")
        require_identifier(self.theory, "domain theory")
        require_identifier(self.material, "domain material")


class Model:
    """One coherent entry point for everything an Analysis needs."""

    def __init__(self, mesh: Mesh) -> None:
        self._mesh = mesh
        self._materials: dict[str, Material] = {}
        self._sections: dict[str, DomainGeometry] = {}
        self._theories: dict[str, Theory] = {}
        self._constitutives: dict[str, ConstitutiveModel] = {}
        self._domains: dict[str, DomainDefinition] = {}
        self._dirichlet: list[ConstraintLike] = []
        self._neumann: list[ConstraintLike] = []
        self._robin: list[ConstraintLike] = []
        self._mpcs: list[object] = []
        self._load_cases: dict[str, LoadCaseLike] = {}

    # ---- registration (collisions raise; nothing silently overrides) ----
    @property
    def mesh(self) -> Mesh:
        """The (immutable) mesh."""
        return self._mesh

    def add_material(self, material: Material) -> None:
        """Register a material by its name."""
        if material.name in self._materials:
            raise ModelError(f"material '{material.name}' already registered")
        self._materials[material.name] = material

    def add_section(self, name: str, section: DomainGeometry) -> None:
        """Register a section/plane geometry by name (a ``CrossSection`` or a ``PlaneGeometry``)."""
        require_identifier(name, "section name")
        if name in self._sections:
            raise ModelError(f"section '{name}' already registered")
        self._sections[name] = section

    def add_theory(self, name: str, theory: Theory) -> None:
        """Register a theory instance by name."""
        require_identifier(name, "theory name")
        if name in self._theories:
            raise ModelError(f"theory '{name}' already registered")
        self._theories[name] = theory

    def add_constitutive(self, name: str, constitutive: ConstitutiveModel) -> None:
        """Register a constitutive-law instance by name (only families that need one)."""
        require_identifier(name, "constitutive name")
        if name in self._constitutives:
            raise ModelError(f"constitutive '{name}' already registered")
        self._constitutives[name] = constitutive

    def add_domain(self, domain: DomainDefinition) -> None:
        """Attach a domain definition (region -> theory/material binding)."""
        if domain.name in self._domains:
            raise ModelError(f"domain '{domain.name}' already registered")
        self._domains[domain.name] = domain

    def add_dirichlet(self, bc: ConstraintLike) -> None:
        """Attach a Dirichlet BC description."""
        self._dirichlet.append(bc)

    def add_neumann(self, bc: ConstraintLike) -> None:
        """Attach a Neumann BC description."""
        self._neumann.append(bc)

    def add_robin(self, bc: ConstraintLike) -> None:
        """Attach a Robin BC description."""
        self._robin.append(bc)

    def add_constraint(self, mpc: object) -> None:
        """Attach a multipoint constraint description."""
        self._mpcs.append(mpc)

    def add_load_case(self, case: LoadCaseLike) -> None:
        """Register a load case by its name."""
        if case.name in self._load_cases:
            raise ModelError(f"load case '{case.name}' already registered")
        self._load_cases[case.name] = case

    @property
    def theories(self) -> dict[str, Theory]:
        """Registered theories (insertion order)."""
        return dict(self._theories)

    @property
    def domains(self) -> dict[str, DomainDefinition]:
        """Registered domain definitions (insertion order)."""
        return dict(self._domains)

    @property
    def materials(self) -> dict[str, Material]:
        """Registered materials (insertion order)."""
        return dict(self._materials)

    @property
    def sections(self) -> dict[str, DomainGeometry]:
        """Registered section/plane geometries (insertion order)."""
        return dict(self._sections)

    @property
    def constitutives(self) -> dict[str, ConstitutiveModel]:
        """Registered constitutive-law instances (insertion order)."""
        return dict(self._constitutives)

    @property
    def dirichlet_bcs(self) -> tuple[ConstraintLike, ...]:
        """Attached Dirichlet BC descriptions, in attachment order."""
        return tuple(self._dirichlet)

    def load_case(self, name: str) -> LoadCaseLike:
        """Registered load case by name; unknown names list what exists."""
        try:
            return self._load_cases[name]
        except KeyError:
            known = ", ".join(sorted(self._load_cases)) or "(none)"
            raise ModelError(f"unknown load case '{name}'; registered: {known}") from None

    @property
    def load_case_names(self) -> tuple[str, ...]:
        """Registered load case names."""
        return tuple(self._load_cases)

    # ---- field materialization (dev note N-4) ----
    def field_specs(self) -> tuple[FieldSpec, ...]:
        """Union of theory field requirements as FieldSpec instances.

        Conflicting component counts for one field name raise; continuity is
        the maximum requirement across theories.
        """
        merged: dict[str, tuple[int, Continuity]] = {}
        component_overrides: dict[str, tuple[str, ...]] = {}
        for dom in self._domains.values():
            theory = self._theories.get(dom.theory)
            if theory is None:
                raise ModelError(f"domain '{dom.name}': theory '{dom.theory}' is not registered")
            cont = theory.continuity_requirements()
            overrides = theory.field_component_names()
            for fname, ncomp in theory.fields():
                required = cont.get(fname, Continuity.C0)
                if fname in merged:
                    n0, c0 = merged[fname]
                    if n0 != ncomp:
                        raise ModelError(
                            f"field '{fname}' declared with {n0} and {ncomp} components "
                            f"by different theories"
                        )
                    required = max(c0, required, key=lambda c: c.value)
                merged[fname] = (ncomp, required)
                if fname in overrides:
                    names = overrides[fname]
                    if len(names) != ncomp:
                        raise ModelError(
                            f"theory '{dom.theory}': field '{fname}' names {len(names)} "
                            f"components but declares {ncomp}"
                        )
                    if fname in component_overrides and component_overrides[fname] != names:
                        raise ModelError(
                            f"field '{fname}' component names disagree across theories: "
                            f"{component_overrides[fname]} vs {names}"
                        )
                    component_overrides[fname] = names
        return tuple(
            FieldSpec(name, component_overrides.get(name, component_names(n)), continuity=c)
            for name, (n, c) in merged.items()
        )

    # ---- validation (fail fast, with mechanics context) ----
    def validate(self) -> None:
        """Raise the first inconsistency; a validated model can attempt analysis."""
        if not self._domains:
            raise ModelError("model defines no domains (region -> theory/material bindings)")
        for dom in self._domains.values():
            region = self._mesh.region(dom.region)
            if region.dimension != self._mesh.dimension:
                raise ModelError(f"domain '{dom.name}': region '{dom.region}' is not a cell region")
            theory = self._theories.get(dom.theory)
            if theory is None:
                raise ModelError(f"domain '{dom.name}': theory '{dom.theory}' is not registered")
            material = self._materials.get(dom.material)
            if material is None:
                raise ModelError(
                    f"domain '{dom.name}': material '{dom.material}' is not registered"
                )
            for key in theory.required_properties():
                if not material.defines(key):
                    raise MissingPropertyError(
                        f"theory '{dom.theory}' requires property '{key}'; "
                        f"material '{dom.material}' does not define it"
                    )
            if dom.geometry is not None and dom.geometry not in self._sections:
                raise ModelError(f"domain '{dom.name}': section '{dom.geometry}' is not registered")
            if dom.constitutive is not None:
                constitutive = self._constitutives.get(dom.constitutive)
                if constitutive is None:
                    raise ModelError(
                        f"domain '{dom.name}': constitutive '{dom.constitutive}' is not registered"
                    )
                for key in constitutive.required_properties():
                    if not material.defines(key):
                        raise MissingPropertyError(
                            f"constitutive '{dom.constitutive}' requires property '{key}'; "
                            f"material '{dom.material}' does not define it"
                        )

        specs = {s.name: s for s in self.field_specs()}
        for kind, bcs in (
            ("Dirichlet", self._dirichlet),
            ("Neumann", self._neumann),
            ("Robin", self._robin),
        ):
            for bc in bcs:
                self._mesh.region(bc.region)  # unknown region raises with context
                if bc.field not in specs:
                    raise ModelError(
                        f"{kind} BC targets field '{bc.field}' which no theory declares "
                        f"(declared: {sorted(specs)})"
                    )
                components = getattr(bc, "components", None)
                if components is not None:
                    bad = [c for c in components if c not in specs[bc.field].components]
                    if bad:
                        raise ModelError(
                            f"{kind} BC on field '{bc.field}': unknown components {bad} "
                            f"(field has {list(specs[bc.field].components)})"
                        )

    def build_dof_handler(self) -> DofHandler:
        """Validate, then number DOFs deterministically over the materialized fields."""
        self.validate()
        return DofHandler.generate(self._mesh, self.field_specs())

    # ---- serialization / fingerprint ----
    def to_manifest(self) -> dict[str, object]:
        """JSON-safe declarative record of the whole model."""
        return {
            "schema": "nanofem-model/1",
            "mesh": self._mesh.to_dict(),
            "materials": {n: m.to_dict() for n, m in sorted(self._materials.items())},
            "sections": sorted(self._sections),
            "theories": sorted(self._theories),
            "constitutives": sorted(self._constitutives),
            "domains": {
                n: {
                    "region": d.region,
                    "theory": d.theory,
                    "material": d.material,
                    "geometry": d.geometry,
                    "constitutive": d.constitutive,
                }
                for n, d in sorted(self._domains.items())
            },
            "load_cases": sorted(self._load_cases),
            "counts": {
                "dirichlet": len(self._dirichlet),
                "neumann": len(self._neumann),
                "robin": len(self._robin),
                "mpc": len(self._mpcs),
            },
        }

    def fingerprint(self) -> str:
        """SHA-256 of the canonical manifest (restart contract, SDS Section 7)."""
        payload = json.dumps(self.to_manifest(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ConstraintLike",
    "DomainDefinition",
    "LoadCaseLike",
    "Model",
]
