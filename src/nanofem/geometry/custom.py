"""CustomSection: user-supplied property table; lookups are storage, not math (SDS 2.2)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nanofem.geometry.base import CrossSection
from nanofem.utils.exceptions import MissingSectionError
from nanofem.utils.validation import require_finite, require_identifier

_KNOWN = (
    "area",
    "second_moment_y",
    "second_moment_z",
    "polar_moment",
    "torsion_constant",
    "shear_correction",
    "warping_constant",
)


class CustomSection(CrossSection):
    """Wrap measured/user-provided section properties; misses raise, never default."""

    def __init__(self, name: str, **properties: float) -> None:
        require_identifier(name, "custom section name")
        self._name = name
        for key, value in properties.items():
            require_finite(value, f"custom section '{name}': {key}")
        self._properties = {k: float(v) for k, v in properties.items()}

    def _get(self, key: str) -> float:
        try:
            return self._properties[key]
        except KeyError:
            raise MissingSectionError(
                f"custom section '{self._name}' does not supply '{key}'; "
                f"supplied: {sorted(self._properties)}; known keys: {list(_KNOWN)}"
            ) from None

    def area(self) -> float:
        """Stored area."""
        return self._get("area")

    def second_moment_y(self) -> float:
        """Stored I_y."""
        return self._get("second_moment_y")

    def second_moment_z(self) -> float:
        """Stored I_z."""
        return self._get("second_moment_z")

    def polar_moment(self) -> float:
        """Stored I_p (distinct from the torsion constant, SDS 2.2)."""
        return self._get("polar_moment")

    def torsion_constant(self) -> float:
        """Stored Saint-Venant torsion constant J_t."""
        return self._get("torsion_constant")

    def shear_correction(self, nu: float) -> float:
        """Stored constant kappa (Poisson dependence documented as unsupported here)."""
        return self._get("shear_correction")

    def warping_constant(self) -> float:
        """Stored warping constant C_w."""
        return self._get("warping_constant")

    def centroid(self) -> NDArray[np.float64]:
        """Stored centroid, defaulting to the section origin."""
        return np.array(
            [self._properties.get("centroid_y", 0.0), self._properties.get("centroid_z", 0.0)]
        )

    def shear_center(self) -> NDArray[np.float64]:
        """Stored shear center, defaulting to the section origin."""
        return np.array(
            [
                self._properties.get("shear_center_y", 0.0),
                self._properties.get("shear_center_z", 0.0),
            ]
        )
