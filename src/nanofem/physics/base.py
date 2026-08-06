"""Theory, ConstitutiveModel, KinematicOperator contracts + declaration records.

Discretization-free by rule R2: theories receive *evaluated* kinematic
batches and never import interpolation, quadrature, mapping, or elements.
Field requirements are plain (name, n_components) data (dev note N-4).

Phase 1 adds the declaration half of SDS Section 4 as first-class data:
TheoryDeclaration captures fields, continuity, properties, state, operators,
kinds, roles, and locality; DeclaredTheory is the concrete, instantiable,
zero-mathematics theory the object model builds with until phase-2 physics
arrives. The constitutive contract remains batched by default (SDS C-8).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, unique

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.assembly.contributions import ContributionKind, OperatorRole
from nanofem.numerics.operators.base import OPERATOR_CATALOG, Continuity
from nanofem.state.layout import StateLayout
from nanofem.utils.exceptions import PhysicsError
from nanofem.utils.validation import require_identifier


@unique
class Locality(Enum):
    """Whether a theory produces per-element or per-element-pair blocks (SDS Section 4)."""

    LOCAL = "local"
    PAIRWISE = "pairwise"


class KinematicOperator(ABC):
    """Generalized strain definition: a theory statement, not element machinery."""

    @abstractmethod
    def required_derivative_order(self) -> int:
        """Return the highest shape derivative consumed (drives derived continuity)."""


class ConstitutiveModel(ABC):
    """Batched map: generalized strain + properties + state -> stress + tangent (SDS Section 5)."""

    @abstractmethod
    def required_properties(self) -> tuple[str, ...]:
        """Return material keys this law will query (validated at model build)."""

    @abstractmethod
    def state_layout(self) -> StateLayout:
        """Return declared state variables (empty for elastic laws)."""

    @abstractmethod
    def response_components(self) -> int:
        """Length n of the generalized stress vector; the tangent is (n, n)."""

    @abstractmethod
    def respond_batch(
        self,
        strains: NDArray[np.float64],
        properties: dict[str, NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (stress, tangent) batches; tangent MUST be the consistent tangent."""


@dataclass(frozen=True)
class ConstitutiveDeclaration:
    """Declaration half of the constitutive contract (interfaces only, phase 1)."""

    name: str
    required_properties: tuple[str, ...]
    state_layout: StateLayout
    response_components: int

    def __post_init__(self) -> None:
        require_identifier(self.name, "constitutive name")
        if self.response_components < 1:
            raise PhysicsError(f"constitutive '{self.name}': response_components must be >= 1")


class Theory(ABC):
    """Discrete statement of governing equations, independent of discretization (SDS Section 4)."""

    @abstractmethod
    def fields(self) -> tuple[tuple[str, int], ...]:
        """Return required fields as (name, n_components) pairs."""

    @abstractmethod
    def continuity_requirements(self) -> dict[str, Continuity]:
        """Return per-field continuity, derived from composed operators (SDS Section 8)."""

    @abstractmethod
    def required_properties(self) -> tuple[str, ...]:
        """Return the union of material keys this theory and its laws query."""

    @abstractmethod
    def required_state(self) -> StateLayout:
        """Return the state variables the theory's laws declare (SDS Section 7)."""

    @abstractmethod
    def operators_used(self) -> tuple[str, ...]:
        """Return SDS Section 8 catalog names this theory composes."""

    @abstractmethod
    def operator_roles(self) -> tuple[OperatorRole, ...]:
        """Return the global-operator roles this theory can produce."""

    @abstractmethod
    def contribution_kinds(self) -> tuple[ContributionKind, ...]:
        """Return the kinds emitted (CELL; FACET for surface; PAIR for nonlocal)."""

    @abstractmethod
    def locality(self) -> Locality:
        """Return LOCAL or PAIRWISE (PAIRWISE implies a kernel + horizon)."""

    def field_component_names(self) -> dict[str, tuple[str, ...]]:
        """Optional explicit component names for a field, keyed by field name.

        A field not listed here defaults to positional axis names
        (``core.fields.component_names`` - ``x``, ``y``, ``z``, ...), correct
        whenever a field's components genuinely *are* spatial axes in that
        order, as ``IsotropicElasticity``'s displacement field ``u`` is. It is
        wrong for a field whose single (or few) components mean something
        else - a beam bending theory's transverse displacement is a
        1-component field named ``u`` that is, by SDS C-2's own worked
        convention, the *y*-component, not positionally the first (``x``).
        Overriding this method is how such a theory states that explicitly,
        without every other theory needing to.
        """
        return {}


@dataclass(frozen=True)
class TheoryDeclaration:
    """First-class declaration data for one theory (SDS Section 4 table, as a record).

    Cross-checks at construction: operators must exist in the SDS Section 8
    catalog; continuity keys must name declared fields; PAIRWISE locality and
    the PAIR kind imply each other.
    """

    name: str
    field_requirements: tuple[tuple[str, int], ...]
    continuity: tuple[tuple[str, Continuity], ...]
    required_properties: tuple[str, ...] = ()
    required_state: StateLayout = field(default_factory=lambda: StateLayout(()))
    operators: tuple[str, ...] = ()
    roles: tuple[OperatorRole, ...] = (OperatorRole.STIFFNESS, OperatorRole.FORCE)
    kinds: tuple[ContributionKind, ...] = (ContributionKind.CELL,)
    locality: Locality = Locality.LOCAL

    def __post_init__(self) -> None:
        require_identifier(self.name, "theory name")
        if not self.field_requirements:
            raise PhysicsError(f"theory '{self.name}' declares no fields")
        names = [n for n, _ in self.field_requirements]
        if len(set(names)) != len(names):
            raise PhysicsError(f"theory '{self.name}': duplicate field names {names}")
        for fname, ncomp in self.field_requirements:
            require_identifier(fname, f"theory '{self.name}' field name")
            if ncomp < 1:
                raise PhysicsError(f"theory '{self.name}': field '{fname}' needs >= 1 component")
        for fname, _ in self.continuity:
            if fname not in names:
                raise PhysicsError(
                    f"theory '{self.name}': continuity declared for unknown field '{fname}'"
                )
        unknown = sorted(set(self.operators) - OPERATOR_CATALOG)
        if unknown:
            raise PhysicsError(
                f"theory '{self.name}': operators {unknown} not in the SDS Section 8 catalog "
                f"{sorted(OPERATOR_CATALOG)}"
            )
        if not self.roles or not self.kinds:
            raise PhysicsError(f"theory '{self.name}' must declare roles and kinds")
        pairwise = self.locality is Locality.PAIRWISE
        has_pair = ContributionKind.PAIR in self.kinds
        if pairwise != has_pair:
            raise PhysicsError(
                f"theory '{self.name}': PAIRWISE locality and the PAIR kind imply each other"
            )


class DeclaredTheory(Theory):
    """Concrete, data-driven theory: pure declarations, zero mathematics (phase 1)."""

    def __init__(self, declaration: TheoryDeclaration) -> None:
        self._d = declaration

    @property
    def declaration(self) -> TheoryDeclaration:
        """The underlying declaration record."""
        return self._d

    def fields(self) -> tuple[tuple[str, int], ...]:
        """Declared (name, n_components) pairs."""
        return self._d.field_requirements

    def continuity_requirements(self) -> dict[str, Continuity]:
        """Declared per-field continuity (unlisted fields default to C0 downstream)."""
        return dict(self._d.continuity)

    def required_properties(self) -> tuple[str, ...]:
        """Declared material keys."""
        return self._d.required_properties

    def required_state(self) -> StateLayout:
        """Declared state layout."""
        return self._d.required_state

    def operators_used(self) -> tuple[str, ...]:
        """Declared operator catalog names."""
        return self._d.operators

    def operator_roles(self) -> tuple[OperatorRole, ...]:
        """Declared roles."""
        return self._d.roles

    def contribution_kinds(self) -> tuple[ContributionKind, ...]:
        """Declared kinds."""
        return self._d.kinds

    def locality(self) -> Locality:
        """Declared locality."""
        return self._d.locality
