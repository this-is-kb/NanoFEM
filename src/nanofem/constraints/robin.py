"""Robin boundary condition description: q = h (u - u_ambient) data (SDS 2.14).

Assembly-side it becomes a FACET STIFFNESS + FORCE pair in phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass

from nanofem.utils.validation import require_finite, require_identifier, require_non_negative


@dataclass(frozen=True)
class RobinBC:
    """Robin/elastic-foundation data: transfer coefficient and ambient value."""

    region: str
    field: str
    coefficient: float
    ambient_value: float

    def __post_init__(self) -> None:
        require_identifier(self.region, "Robin region")
        require_identifier(self.field, "Robin field")
        require_non_negative(self.coefficient, "Robin coefficient")
        require_finite(self.ambient_value, "Robin ambient value")
