"""Field declarations: name, components, variable type, continuity (SDS stage 2).

Theories declare requirements as plain (name, n_components) data (dev note
N-4); the Model materializes them into FieldSpec instances. Factory functions
provide the canonical examples: displacement, rotation, temperature, electric
potential, pressure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique

from nanofem.numerics.operators.base import Continuity
from nanofem.utils.exceptions import InputValidationError
from nanofem.utils.validation import require_identifier


@unique
class VariableType(Enum):
    """Physical interpretation of a field (metadata; never drives numerics)."""

    DISPLACEMENT = "displacement"
    ROTATION = "rotation"
    TEMPERATURE = "temperature"
    ELECTRIC_POTENTIAL = "electric_potential"
    PRESSURE = "pressure"
    GENERIC = "generic"


@dataclass(frozen=True)
class FieldSpec:
    """One physical field: components in SDS C-2 order, required continuity."""

    name: str
    components: tuple[str, ...]
    variable_type: VariableType = VariableType.GENERIC
    continuity: Continuity = Continuity.C0

    def __post_init__(self) -> None:
        require_identifier(self.name, "field name")
        if not self.components:
            raise InputValidationError(f"field '{self.name}' must declare at least one component")
        for c in self.components:
            require_identifier(c, f"component of field '{self.name}'")
        if len(set(self.components)) != len(self.components):
            raise InputValidationError(f"field '{self.name}' has duplicate components")

    @property
    def dofs_per_node(self) -> int:
        """DOFs this field attaches to each node (= number of components)."""
        return len(self.components)

    def to_dict(self) -> dict[str, object]:
        """JSON-safe record."""
        return {
            "name": self.name,
            "components": list(self.components),
            "variable_type": self.variable_type.value,
            "continuity": self.continuity.name,
        }


_AXES: tuple[str, ...] = ("x", "y", "z")


def component_names(n: int) -> tuple[str, ...]:
    """Canonical component names: x, y, z for n <= 3, else c0..c(n-1)."""
    if n <= 3:
        return _AXES[:n]
    return tuple(f"c{i}" for i in range(n))


def displacement(dimension: int) -> FieldSpec:
    """Displacement field u with one component per spatial dimension."""
    return FieldSpec("u", component_names(dimension), VariableType.DISPLACEMENT)


def rotation_z() -> FieldSpec:
    """In-plane rotation field (beams/frames), C1 territory later."""
    return FieldSpec("r", ("z",), VariableType.ROTATION)


def temperature() -> FieldSpec:
    """Scalar temperature field."""
    return FieldSpec("T", ("t",), VariableType.TEMPERATURE)


def electric_potential() -> FieldSpec:
    """Scalar electric potential (piezoelectricity, SDS 4.7)."""
    return FieldSpec("phi", ("p",), VariableType.ELECTRIC_POTENTIAL)


def pressure() -> FieldSpec:
    """Scalar pressure field (future mixed formulations)."""
    return FieldSpec("p", ("p",), VariableType.PRESSURE)
