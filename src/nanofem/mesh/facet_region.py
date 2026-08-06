"""FacetRegion: a named set of boundary facets, identified as (cell, local facet index) pairs.

``Region`` (SDS 2.1) only ever supported node-dimension (0) and cell-dimension
regions; facet/edge regions were explicitly deferred at that time ("arrive
with the phase-2 orientation machinery"). A facet has no single global id of
its own the way a node or a cell does - two cells can share one interior
facet - but every *boundary* facet (the only kind a traction load or a
Neumann flux needs) belongs to exactly one cell, so a ``(cell_id,
local_facet_index)`` pair is a sufficient, unambiguous identity, with
``local_facet_index`` resolved against that cell's own
``ReferenceElement.facet_vertex_indices`` (``numerics.reference``, complete
since v0.2.0). This is therefore a new, small, additive record - not a
variant of ``Region``, since facet identity genuinely needs two integers
where node/cell identity needs one.
"""

from __future__ import annotations

from dataclasses import dataclass

from nanofem.utils.exceptions import InputValidationError
from nanofem.utils.validation import require_identifier


@dataclass(frozen=True)
class FacetRegion:
    """Named set of ``(cell_id, local_facet_index)`` boundary-facet identifiers."""

    name: str
    facets: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        require_identifier(self.name, "facet region name")
        if not self.facets:
            raise InputValidationError(
                f"facet region '{self.name}' must reference at least one facet"
            )
        if len(set(self.facets)) != len(self.facets):
            raise InputValidationError(f"facet region '{self.name}' has duplicate facets")
        for cell_id, local_index in self.facets:
            if cell_id < 0 or local_index < 0:
                raise InputValidationError(
                    f"facet region '{self.name}': (cell_id, local_facet_index) must be "
                    f"non-negative, got ({cell_id}, {local_index})"
                )

    def to_dict(self) -> dict[str, object]:
        """JSON-safe record."""
        return {"name": self.name, "facets": [list(f) for f in self.facets]}
