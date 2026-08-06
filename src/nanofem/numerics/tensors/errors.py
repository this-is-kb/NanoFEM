"""Error types for the tensor algebra layer (SDS Section 9).

Rooted at :class:`nanofem.utils.exceptions.NanoFEMError` so one catch clause
still covers every NanoFEM failure. Routine shape/dtype mistakes continue to
raise :class:`~nanofem.utils.exceptions.InputValidationError`; the types here
are reserved for failures specific to tensor representation and rotation.
"""

from __future__ import annotations

from nanofem.utils.exceptions import NanoFEMError


class TensorError(NanoFEMError):
    """Base class for tensor algebra failures."""


class RepresentationError(TensorError):
    """A Voigt/Mandel/full-tensor round trip failed, or a dimension is unsupported.

    Raised when a converter is asked to operate on a spatial dimension outside
    ``{1, 2, 3}`` (SDS C-1 only defines Voigt ordering for those), or when an
    internal consistency check on a representation bridge fails.
    """


class NotRotationError(TensorError):
    """A matrix supplied as a frame rotation is not in ``SO(d)``.

    Raised when ``Q^T Q != I`` or ``det(Q) != +1`` beyond tolerance - a
    reflection or a non-orthogonal matrix cannot be used to rotate strain or
    stress without violating the transformation's own consistency (SDS
    Section 9, "Rotation").
    """


class TensorSymmetryError(TensorError):
    """A fourth-order tensor lacks a symmetry it was asserted to have.

    Raised when major symmetry (``C_ijkl == C_klij``) or minor symmetry
    (``C_ijkl == C_jikl == C_ijlk``) is checked and found absent.
    """
