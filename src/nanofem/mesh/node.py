"""Node record: identity, coordinates, tags, metadata (SDS 2.1).

Authoritative node storage is the Mesh coordinate array (structure-of-arrays,
SDS C-8); Node is the per-entity *record view* the object model exposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from nanofem.utils.exceptions import InputValidationError
from nanofem.utils.validation import require_finite_array


@dataclass(frozen=True, eq=False)
class Node:
    """One node: id, coordinates (d,), user tags, string metadata."""

    node_id: int
    coordinates: NDArray[np.float64]
    tags: frozenset[str] = frozenset()
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, int) or isinstance(self.node_id, bool) or self.node_id < 0:
            raise InputValidationError(f"node_id must be a non-negative int, got {self.node_id!r}")
        coords = np.asarray(self.coordinates, dtype=np.float64)
        if coords.ndim != 1 or not 1 <= coords.size <= 3:
            raise InputValidationError(
                f"node {self.node_id}: coordinates must be 1-D of length 1..3, "
                f"got shape {coords.shape}"
            )
        require_finite_array(coords, f"node {self.node_id} coordinates")
        coords.setflags(write=False)
        object.__setattr__(self, "coordinates", coords)

    @property
    def dimension(self) -> int:
        """Spatial dimension of this node."""
        return int(self.coordinates.size)

    def to_dict(self) -> dict[str, object]:
        """JSON-safe record."""
        return {
            "node_id": self.node_id,
            "coordinates": [float(c) for c in self.coordinates],
            "tags": sorted(self.tags),
            "metadata": dict(self.metadata),
        }
