"""FUTURE (fenced by ADR): FEM over the section for arbitrary J_t, warping, shear center.

Computing these for arbitrary polygons requires solving Saint-Venant problems
over a meshed cross-section - a small FEM project in itself (SDS 2.2). Until
then, CustomSection covers the gap with user-supplied values.
"""

from __future__ import annotations


class SectionPropertySolver:
    """Placeholder for the arbitrary-section property solver (future ADR)."""
