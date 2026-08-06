"""Run configuration carrier (no module-level mutable state, ever)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Configuration:
    """Library-wide options: strict promotes warnings to errors (SDS Section 13);
    debug_checks enables C-7 symmetry assertions and similar witnesses."""

    strict: bool = False
    debug_checks: bool = False
