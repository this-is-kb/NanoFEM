"""Frame rotation utilities shared across numerics (SDS Section 9 transformation row)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rotation_matrix_2d(angle: float) -> NDArray[np.float64]:
    """Return the 2x2 rotation matrix for a counterclockwise angle in radians."""
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin], [sin, cos]], dtype=np.float64)
