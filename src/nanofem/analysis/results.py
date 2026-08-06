"""Immutable, analysis-specific result containers consumed by postprocess (SDS 2.18).

Fields are pinned when each analysis defines what a result actually carries;
declaring guesses now would fossilize them. ``StaticResult`` is real as of
the walking skeleton (v0.8.0) but lives in ``analysis/static.py`` next to
``LinearStaticAnalysis`` itself, not here - this module only holds
placeholders for analyses that don't exist yet.
"""

from __future__ import annotations


class ModalResult:
    """Frequencies and mass-normalized mode shapes. TODO(phase-3)."""


class BucklingResult:
    """Load factors and buckling shapes. TODO(phase-3)."""


class TransientResult:
    """Time histories, streamed. TODO(phase-3)."""
