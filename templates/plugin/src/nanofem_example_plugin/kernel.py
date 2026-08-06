"""Template kernel plugin: implement the SDS kernel contract, register, certify."""

from __future__ import annotations

from nanofem.kernels.base import Kernel


class ExampleConeKernel(Kernel):
    """Compact-support cone kernel placeholder.

    TODO(template user): implement evaluate() and support_radius() per the
    SDS kernel contract, then run the kernel conformance kit.
    """
