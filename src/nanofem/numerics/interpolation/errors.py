"""Error types for the interpolation framework.

These root at :class:`nanofem.utils.exceptions.NanoFEMError` so one catch
clause still covers every NanoFEM failure. They signal an *internal*
inconsistency in an interpolation definition (a library bug, surfaced by
:meth:`Interpolation.validate`), as distinct from ``InputValidationError``,
which reports a bad argument supplied by a caller (an unsupported order, a
point of the wrong dimension).
"""

from __future__ import annotations

from nanofem.utils.exceptions import NanoFEMError


class InterpolationError(NanoFEMError):
    """Base class for interpolation-framework failures."""


class PolynomialSpaceError(InterpolationError):
    """A polynomial space is inconsistent (degree, completeness, dimension)."""


class UnisolvenceError(InterpolationError):
    """The degrees of freedom do not uniquely determine a member of the space.

    Raised when the generalized Vandermonde ``M[k, j] = l_k(m_j)`` is not
    square or is rank deficient: the nodal basis dual to these functionals
    then does not exist (or is not unique), so no shape functions can be
    built from this definition.
    """


class NodeOrderingError(InterpolationError):
    """Interpolation nodes violate the numbering or entity-association rules."""
