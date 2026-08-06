"""Namespaced logger factory. No ``print`` anywhere in the library.

INFO = pipeline milestones, DEBUG = per-element detail (SDS Section 13).
Handlers/formatters are the application's business; the library only names
loggers under the ``nanofem`` namespace.
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a NanoFEM module, namespaced under ``nanofem``."""
    qualified = name if name.startswith("nanofem") else f"nanofem.{name}"
    return logging.getLogger(qualified)
