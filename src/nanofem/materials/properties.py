"""Canonical property keys and the spatial-variation placeholder (SDS Section 6)."""

from __future__ import annotations

from nanofem.materials.grading import GradingLaw
from nanofem.materials.material import _BOUNDS
from nanofem.utils.validation import require_identifier

#: Derived from ``material._BOUNDS`` (the load-bearing definition) rather than
#: hardcoded a second time, so the two vocabularies cannot drift apart.
CANONICAL_KEYS: tuple[str, ...] = tuple(sorted(_BOUNDS))


class SpatialProperty:
    """A property key bound to a grading law: position -> value (the FGM story).

    Evaluation is phase-2 work; phase 1 stores the validated binding.
    """

    def __init__(self, key: str, law: GradingLaw) -> None:
        require_identifier(key, "property key")
        self._key = key
        self._law = law

    @property
    def key(self) -> str:
        """Bound property key."""
        return self._key

    @property
    def law(self) -> GradingLaw:
        """Bound grading law."""
        return self._law
