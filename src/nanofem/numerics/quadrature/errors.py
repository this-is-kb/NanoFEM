"""Error types for the quadrature layer.

Rooted at :class:`nanofem.utils.exceptions.NanoFEMError` so one catch clause
still covers every NanoFEM failure.
"""

from __future__ import annotations

from nanofem.utils.exceptions import NanoFEMError


class QuadratureError(NanoFEMError):
    """Base class for quadrature failures."""


class ExactnessError(QuadratureError):
    """A rule does not integrate the polynomials it claims to integrate exactly.

    Raised when the numerical integral of a monomial within the declared
    exactness degree disagrees with its closed-form value, or when a rule is
    exact beyond the degree it reports (which would understate it).
    """


class WeightError(QuadratureError):
    """A rule's weights are inconsistent with its domain or its declaration.

    Raised when the weights do not sum to the reference measure, or when a rule
    declaring positive weights has one that is not.
    """


class SymmetryError(QuadratureError):
    """A rule declaring symmetry is not invariant under the element's symmetries."""
