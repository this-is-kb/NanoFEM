"""Neumann boundary condition description: prescribed flux/traction data (SDS 2.14).

Phase-1 addition per the object-model requirements; assembly-side it becomes
a FACET FORCE contribution in phase 2 (SDS Section 10).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.utils.exceptions import InputValidationError
from nanofem.utils.validation import require_finite_array, require_identifier


@dataclass(frozen=True, eq=False)
class NeumannBC:
    """Prescribed flux per component of a field on a boundary region."""

    region: str
    field: str
    flux: NDArray[np.float64]

    def __post_init__(self) -> None:
        require_identifier(self.region, "Neumann region")
        require_identifier(self.field, "Neumann field")
        flux = np.asarray(self.flux, dtype=np.float64)
        if flux.ndim != 1 or flux.size < 1:
            raise InputValidationError(f"Neumann flux must be 1-D per-component, got {flux.shape}")
        require_finite_array(flux, "Neumann flux")
        flux.setflags(write=False)
        object.__setattr__(self, "flux", flux)
