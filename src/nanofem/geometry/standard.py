"""Standard cross-sections: validated dimension storage, closed-form properties (SDS 2.2).

Property *computation* for most shapes remains phase-2 work; every computed
accessor still raises until then, and _StoredSection centralizes those
stubs. `CircularSection` is the one exception, made real end to end for the
walking skeleton: every one of its nine properties has an *exact* closed
form (SDS 2.2) - including the circular-only identity that the torsion
constant equals the polar moment, which is precisely the case the
CrossSection contract's own docstring calls out as the classic frame-code
bug this API makes structurally impossible. RectangularSection and the
hollow/I variants stay deferred: their torsion constants have no equally
clean exact closed form (a shape-factor series, not a formula), and nothing
in this phase's Bar element needs them.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from nanofem.geometry.base import CrossSection
from nanofem.utils.exceptions import InputValidationError
from nanofem.utils.validation import require_positive


class _StoredSection(CrossSection):
    """Shared phase-1 base: stores dimensions; computed properties raise."""

    def area(self) -> float:
        """TODO(phase-2)."""
        raise NotImplementedError("TODO(phase-2): section property computation")

    def second_moment_y(self) -> float:
        """TODO(phase-2)."""
        raise NotImplementedError("TODO(phase-2): section property computation")

    def second_moment_z(self) -> float:
        """TODO(phase-2)."""
        raise NotImplementedError("TODO(phase-2): section property computation")

    def polar_moment(self) -> float:
        """TODO(phase-2)."""
        raise NotImplementedError("TODO(phase-2): section property computation")

    def torsion_constant(self) -> float:
        """TODO(phase-2)."""
        raise NotImplementedError("TODO(phase-2): section property computation")

    def shear_correction(self, nu: float) -> float:
        """TODO(phase-2)."""
        raise NotImplementedError("TODO(phase-2): section property computation")

    def warping_constant(self) -> float:
        """TODO(phase-2)."""
        raise NotImplementedError("TODO(phase-2): section property computation")

    def centroid(self) -> NDArray[np.float64]:
        """TODO(phase-2)."""
        raise NotImplementedError("TODO(phase-2): section property computation")

    def shear_center(self) -> NDArray[np.float64]:
        """TODO(phase-2)."""
        raise NotImplementedError("TODO(phase-2): section property computation")


class RectangularSection(_StoredSection):
    """Solid rectangle width x height."""

    def __init__(self, width: float, height: float) -> None:
        require_positive(width, "width")
        require_positive(height, "height")
        self.width = width
        self.height = height


class CircularSection(CrossSection):
    """Solid circle of given radius - every property an exact SDS 2.2 closed form.

    Subclasses ``CrossSection`` directly (not ``_StoredSection``): all nine
    abstract methods are overridden with real formulas, so none of the
    phase-2 stubs are inherited.
    """

    def __init__(self, radius: float) -> None:
        require_positive(radius, "radius")
        self.radius = radius

    def area(self) -> float:
        """``A = pi r^2``."""
        return math.pi * self.radius**2

    def second_moment_y(self) -> float:
        """``I_y = pi r^4 / 4`` (symmetric about both centroidal axes)."""
        return math.pi * self.radius**4 / 4.0

    def second_moment_z(self) -> float:
        """``I_z = pi r^4 / 4`` (symmetric about both centroidal axes)."""
        return math.pi * self.radius**4 / 4.0

    def polar_moment(self) -> float:
        """``I_p = I_y + I_z = pi r^4 / 2``."""
        return self.second_moment_y() + self.second_moment_z()

    def torsion_constant(self) -> float:
        """``J_t = I_p`` - the circular exception where torsion constant and polar
        moment coincide exactly (SDS 2.2); this is *not* true in general, which is
        why ``CrossSection`` names the two separately.
        """
        return self.polar_moment()

    def shear_correction(self, nu: float) -> float:
        """Cowper's shear correction for a solid circle: ``kappa = 6(1+nu)/(7+6nu)``."""
        return 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)

    def warping_constant(self) -> float:
        """``C_w = 0`` exactly: Saint-Venant torsion of a solid circle produces no warping."""
        return 0.0

    def centroid(self) -> NDArray[np.float64]:
        """``(0, 0)`` exactly, by symmetry, in centroidal principal axes."""
        return np.zeros(2, dtype=np.float64)

    def shear_center(self) -> NDArray[np.float64]:
        """``(0, 0)`` exactly, by symmetry - coincides with the centroid."""
        return np.zeros(2, dtype=np.float64)


class HollowCircularSection(_StoredSection):
    """Annulus with 0 < inner_radius < outer_radius."""

    def __init__(self, outer_radius: float, inner_radius: float) -> None:
        require_positive(outer_radius, "outer_radius")
        require_positive(inner_radius, "inner_radius")
        if inner_radius >= outer_radius:
            raise InputValidationError(
                f"inner_radius {inner_radius} must be smaller than outer_radius {outer_radius}"
            )
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius


class HollowRectangularSection(_StoredSection):
    """Rectangular tube with wall thickness below half the smaller side."""

    def __init__(self, width: float, height: float, thickness: float) -> None:
        require_positive(width, "width")
        require_positive(height, "height")
        require_positive(thickness, "thickness")
        if thickness >= min(width, height) / 2.0:
            raise InputValidationError(f"thickness {thickness} must be below half the smaller side")
        self.width = width
        self.height = height
        self.thickness = thickness


class ISection(_StoredSection):
    """I/H shape: flange and web parameters."""

    def __init__(
        self,
        flange_width: float,
        flange_thickness: float,
        web_height: float,
        web_thickness: float,
    ) -> None:
        require_positive(flange_width, "flange_width")
        require_positive(flange_thickness, "flange_thickness")
        require_positive(web_height, "web_height")
        require_positive(web_thickness, "web_thickness")
        self.flange_width = flange_width
        self.flange_thickness = flange_thickness
        self.web_height = web_height
        self.web_thickness = web_thickness
