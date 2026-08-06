"""Strongly typed enumeration of geometric mapping kinds.

Names the *character* of the map from reference to physical space, which is what
downstream code branches on: an affine map has a constant Jacobian and a
vanishing mapping Hessian, so second-derivative transformations simplify and the
inverse is closed-form. Everything else must pay the general price.
"""

from __future__ import annotations

from enum import Enum, unique


@unique
class MappingType(Enum):
    """The kind of geometric mapping from the reference to the physical element."""

    IDENTITY = "identity"
    AFFINE = "affine"
    ISOPARAMETRIC = "isoparametric"
    CURVILINEAR = "curvilinear"
    NURBS = "nurbs"
    HIGH_ORDER = "high_order"

    @property
    def is_implemented(self) -> bool:
        """Whether a concrete mapping exists for this kind in this phase."""
        return self in _IMPLEMENTED

    @property
    def has_constant_jacobian(self) -> bool:
        """Whether this kind guarantees a constant Jacobian over the element.

        The guarantee is a property of the *kind*: identity and affine maps have
        it by definition. An isoparametric map may happen to be affine for a
        particular geometry (a parallelogram quadrilateral), which is why
        :pyattr:`GeometricMapping.is_affine` is the runtime question and this is
        only the declaration.
        """
        return self in {MappingType.IDENTITY, MappingType.AFFINE}


_IMPLEMENTED: frozenset[MappingType] = frozenset(
    {MappingType.IDENTITY, MappingType.AFFINE, MappingType.ISOPARAMETRIC}
)
