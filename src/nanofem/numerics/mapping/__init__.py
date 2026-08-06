"""nanofem.numerics.mapping.

Geometric mappings between the reference element and the physical element
(SDS 2.6): reference coordinates -> physical coordinates -> Jacobian -> inverse
Jacobian -> metric tensor -> physical derivatives.

Purely geometric. There is no numerical integration, no quadrature rule, no
element matrix, no assembly, and no constitutive model in this package. It
composes the two layers below it - the reference domain from
``numerics.reference`` and the geometry basis from ``numerics.interpolation`` -
and re-derives neither.

Responsibilities
----------------
- Reference-physical transformation in both directions, exact for affine maps
  and by Newton iteration otherwise
- Jacobian, pseudo-inverse, determinant and Gram measure scaling, metric tensor,
  covariant and contravariant bases
- Gradient push-forward and pull-back; the physical Hessian, including the
  mapping-curvature correction that is mandatory unless the map is affine
- Element geometry: centroid, bounding box, edge chords, characteristic length,
  aspect ratio, scaled-Jacobian quality
- Degenerate-cell detection: coincident nodes, zero and negative Jacobians,
  inverted orientation, near-singular maps
- Caching of every derived batch per (quantity, point set)

TODO
----
- TODO(phase-6): facet mappings and outward normals in physical space
- TODO(phase-6+): the curvilinear, NURBS, and high-order maps in ``future.py``
"""

from __future__ import annotations

from nanofem.numerics.mapping.affine import AffineMapping, IdentityMapping
from nanofem.numerics.mapping.base import GeometricMapping
from nanofem.numerics.mapping.enums import MappingType
from nanofem.numerics.mapping.errors import (
    EmbeddedMappingError,
    InverseMapError,
    MappingError,
    NonAffineError,
)
from nanofem.numerics.mapping.future import (
    CurvilinearMapping,
    HighOrderMapping,
    NURBSMapping,
)
from nanofem.numerics.mapping.isoparametric import IsoparametricMapping

__all__ = [
    # enumeration
    "MappingType",
    # errors
    "MappingError",
    "NonAffineError",
    "InverseMapError",
    "EmbeddedMappingError",
    # contract
    "GeometricMapping",
    # implemented mappings
    "IdentityMapping",
    "AffineMapping",
    "IsoparametricMapping",
    # declared placeholders
    "CurvilinearMapping",
    "NURBSMapping",
    "HighOrderMapping",
]
