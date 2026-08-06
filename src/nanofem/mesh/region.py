"""Region: a named set of mesh entities of one dimension (SDS 2.1).

dimension = 0 tags nodes; dimension = mesh dimension tags cells. Facet/edge
regions ((cell, local index) records) arrive with the phase-2 orientation
machinery (SDS C-3) and are refused until then rather than half-supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nanofem.utils.exceptions import InputValidationError
from nanofem.utils.validation import require_identifier


@dataclass(frozen=True)
class Region:
    """Named entity set: physical-group analogue preserved from import."""

    name: str
    dimension: int
    entity_ids: tuple[int, ...]
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_identifier(self.name, "region name")
        if not isinstance(self.dimension, int) or not 0 <= self.dimension <= 3:
            raise InputValidationError(f"region '{self.name}': dimension must be 0..3")
        if not self.entity_ids:
            raise InputValidationError(f"region '{self.name}' must reference at least one entity")
        if any((not isinstance(i, int)) or isinstance(i, bool) or i < 0 for i in self.entity_ids):
            raise InputValidationError(
                f"region '{self.name}': entity ids must be non-negative ints"
            )
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise InputValidationError(f"region '{self.name}' has duplicate entity ids")

    def to_dict(self) -> dict[str, object]:
        """JSON-safe record."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "entity_ids": list(self.entity_ids),
            "metadata": dict(self.metadata),
        }
