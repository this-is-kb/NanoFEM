"""Error types for the reference element library.

These root at :class:`nanofem.utils.exceptions.NanoFEMError` so a user can
still catch every NanoFEM failure with one clause, while giving the reference
layer a precise vocabulary. ``TopologyError`` and ``OrientationError`` signal
an *internal* inconsistency in a reference element definition (a bug in the
library, surfaced by :meth:`ReferenceElement.validate`), as distinct from
``InputValidationError`` which reports a bad *argument* passed by a caller
(for example a query point of the wrong dimension).
"""

from __future__ import annotations

from nanofem.utils.exceptions import NanoFEMError


class ReferenceElementError(NanoFEMError):
    """Base class for reference-geometry failures."""


class TopologyError(ReferenceElementError):
    """A reference element's topology is inconsistent (counts, incidence, uniqueness)."""


class OrientationError(ReferenceElementError):
    """A reference element's orientation is inconsistent (a normal points inward)."""
