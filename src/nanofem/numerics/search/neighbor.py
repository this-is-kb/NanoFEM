"""KD-tree horizon queries for nonlocal pairs (moved from mesh, ADR-007; SDS Section 10).

Returns unordered pairs within a kernel support radius, self-pairs included;
one numerical integration per unordered pair is contract (PAIR symmetry exact
by construction).
"""

from __future__ import annotations


class NeighborSearch:
    """Element-centroid proximity queries backed by scipy.spatial (lazy import)."""

    def pairs_within(self, radius: float) -> tuple[tuple[int, int], ...]:
        """Return unordered (e, e') pairs with centroid distance <= radius."""
        raise NotImplementedError("TODO(phase-4b): KD-tree pair enumeration")
