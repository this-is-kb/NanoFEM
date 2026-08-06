"""Standard kernels: Helmholtz family, Gaussian, bi-exponential, polynomial cone.

TODO(phase-4b): implementations with dimension-aware normalization.
"""

from __future__ import annotations

from nanofem.kernels.base import Kernel


class HelmholtzKernel(Kernel):
    """Green's function of the Helmholtz operator; per-dimension realizations."""


class BiExponentialKernel(Kernel):
    """exp(-|r|/l)/(2l): the 1-D Helmholtz realization, named for the literature."""


class GaussianKernel(Kernel):
    """Gaussian attenuation with dimension-aware normalization."""


class PolynomialConeKernel(Kernel):
    """Compact-support polynomial (cone) kernel."""
