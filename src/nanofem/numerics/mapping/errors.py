"""Error types for the geometric mapping layer.

``DegenerateCellError`` is deliberately reused rather than redefined: the
exception tree already declares it as "non-positive Jacobian or zero-measure
cell (SDS 2.6)", and SDS 2.6 *is* this layer. It lives under ``MeshError``
because a degenerate map is almost always a bad mesh, and reporting it as such
puts the user at the real cause.

This layer raises with what it knows - the reference point, the determinant, the
node coordinates. It does not know element ids or region names; the element
layer above it adds those when it re-raises, per the SDS policy that every error
carries mechanics context.
"""

from __future__ import annotations

from nanofem.utils.exceptions import NanoFEMError


class MappingError(NanoFEMError):
    """Base class for geometric mapping failures."""


class NonAffineError(MappingError):
    """An affine map was requested for geometry that is not affinely mappable.

    Raised when the physical vertices cannot be reproduced by any single
    ``x = A xi + b``. The common case is a quadrilateral that is not a
    parallelogram: four corners impose eight conditions on six unknowns, so only
    a parallelogram admits an exact affine fit. Use an isoparametric map.
    """


class InverseMapError(MappingError):
    """The physical-to-reference map could not be evaluated.

    Either the Newton iteration failed to converge, or the point does not lie on
    the element's affine subspace (which can happen only for an embedded map,
    where the physical space has more dimensions than the element).
    """


class EmbeddedMappingError(MappingError):
    """An operation was requested that is undefined for an embedded element.

    An element whose embedding dimension exceeds its topological dimension - a
    bar in a plane, a shell in space - has a non-square Jacobian. Signed
    determinants and orientation do not exist for it; the measure scaling is the
    Gram determinant ``sqrt(det(J^T J))`` instead.
    """
