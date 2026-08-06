"""Boundary-truncation policies for kernel normalization (SDS 4.3).

The normalization integral fails near boundaries when the horizon is
truncated; whether to renormalize or let the two-phase model absorb the
deficit is a modeling decision - an explicit policy object, never a hidden
default.
"""

from __future__ import annotations


class NormalizationPolicy:
    """Placeholder: renormalize vs. two-phase absorption, declared per model."""
