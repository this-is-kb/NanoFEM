"""Plane-continuum geometry: thickness carrier for 2-D elements (SDS 2.2)."""

from __future__ import annotations

from dataclasses import dataclass

from nanofem.utils.validation import require_positive


@dataclass(frozen=True)
class PlaneGeometry:
    """Thickness t of a plane-stress/plane-strain continuum."""

    thickness: float

    def __post_init__(self) -> None:
        require_positive(self.thickness, "thickness")
