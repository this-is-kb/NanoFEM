"""Dirichlet boundary condition descriptions (SDS 2.14): data, never application."""

from __future__ import annotations

from dataclasses import dataclass

from nanofem.utils.exceptions import InputValidationError
from nanofem.utils.validation import require_finite, require_identifier


@dataclass(frozen=True)
class DirichletBC:
    """Prescribed value for (region, field, components); constant in phase 1."""

    region: str
    field: str
    components: tuple[str, ...]
    value: float

    def __post_init__(self) -> None:
        require_identifier(self.region, "Dirichlet region")
        require_identifier(self.field, "Dirichlet field")
        if not self.components:
            raise InputValidationError("Dirichlet BC must name at least one component")
        require_finite(self.value, "Dirichlet value")
