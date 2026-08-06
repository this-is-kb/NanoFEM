"""CrossSection contract: section properties for structural members (SDS 2.2).

Torsion constant and polar moment are distinct quantities (they coincide only
for circular sections); the API names them separately by design. Poisson's
ratio enters shear_correction as a plain float so this package never imports
nanofem.materials.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class CrossSection(ABC):
    """Abstract cross-section: every quantity in centroidal principal axes."""

    @abstractmethod
    def area(self) -> float:
        """Return A = integral of dA."""

    @abstractmethod
    def second_moment_y(self) -> float:
        """Return I_y = integral of z^2 dA."""

    @abstractmethod
    def second_moment_z(self) -> float:
        """Return I_z = integral of y^2 dA."""

    @abstractmethod
    def polar_moment(self) -> float:
        """Return I_p = I_y + I_z (NOT the torsion constant in general)."""

    @abstractmethod
    def torsion_constant(self) -> float:
        """Return the Saint-Venant torsion constant J_t."""

    @abstractmethod
    def shear_correction(self, nu: float) -> float:
        """Return kappa(nu), e.g. Cowper's formulas (SDS 2.2)."""

    @abstractmethod
    def warping_constant(self) -> float:
        """Return the warping constant C_w."""

    @abstractmethod
    def centroid(self) -> NDArray[np.float64]:
        """Return the centroid in section axes, shape (2,)."""

    @abstractmethod
    def shear_center(self) -> NDArray[np.float64]:
        """Return the shear center in section axes, shape (2,)."""
