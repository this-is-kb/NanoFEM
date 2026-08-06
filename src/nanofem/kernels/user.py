"""User-defined kernel wrapper with validation (SDS Section 12 plugin path)."""

from __future__ import annotations

from nanofem.kernels.base import Kernel


class UserKernel(Kernel):
    """Validated callable wrapper; conformance kit checks normalization honesty."""
