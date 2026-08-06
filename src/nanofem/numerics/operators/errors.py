"""Error types for the discrete operator library (SDS Section 8).

Rooted at :class:`nanofem.utils.exceptions.NanoFEMError` so one catch clause
still covers every NanoFEM failure. Routine shape/value mistakes continue to
raise :class:`~nanofem.utils.exceptions.InputValidationError`; the types here
are reserved for failures specific to operator recipes.
"""

from __future__ import annotations

from nanofem.utils.exceptions import NanoFEMError


class OperatorError(NanoFEMError):
    """Base class for discrete operator failures."""


class OperatorShapeError(OperatorError):
    """A tabulated batch does not have the shape an operator recipe requires.

    Raised when, for example, physical gradients are not rank-3
    ``(n_qp, n_fun, dim)``, or a facet normal batch does not match the number
    of quadrature points it is paired with.
    """


class UnsupportedDimensionError(OperatorError):
    """An operator was asked to act in a spatial dimension it does not support.

    Raised by :mod:`curl`, whose scalar (2-D) and vector (3-D) forms are
    genuinely different operators - there is no 1-D curl.
    """
