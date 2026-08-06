"""nanofem.numerics.reference.

The reference geometry layer: the canonical reference domains of finite
elements and their topological lattices. This package is purely geometric and
topological - it contains no shape functions, no quadrature, no Jacobians, no
interpolation, and no mapping. It is the permanent geometric foundation on
which every future finite element is built, and it depends on nothing in
NanoFEM beyond the shared exception base.

Two complementary abstractions live here:

- ``ReferenceCell`` (``cell.py``): a lightweight record keyed by a mesh
  cell-type name (``"tri6"``), carrying node/entity counts and a measure; used
  by the mesh, quadrature, and interpolation layers as a quick lookup.
- ``ReferenceElement`` (``element.py`` and the concrete shapes): the full
  reference domain and topology (vertices, edges, faces, facets, orientation,
  normals, containment), independent of interpolation order.

Responsibilities
----------------
- Reference domains, canonical vertex/edge/face/facet numbering, orientation
  and outward-normal conventions (SDS C-3, 2.3)
- Topological and geometric queries, validation, and serialization for the
  line, triangle, and quadrilateral; declared placeholders for the 3-D shapes

TODO
----
- TODO(phase-2b): implement the 3-D reference elements (tet, hex, prism,
  pyramid) sketched in ``future.py``
"""

from __future__ import annotations

from nanofem.numerics.reference.cell import REFERENCE_CELLS, ReferenceCell, reference_cell
from nanofem.numerics.reference.element import ReferenceElement
from nanofem.numerics.reference.enums import (
    CellType,
    Dimension,
    EntityType,
    FacetType,
    Orientation,
)
from nanofem.numerics.reference.errors import (
    OrientationError,
    ReferenceElementError,
    TopologyError,
)
from nanofem.numerics.reference.future import (
    ReferenceHexahedron,
    ReferencePrism,
    ReferencePyramid,
    ReferenceTetrahedron,
)
from nanofem.numerics.reference.line import ReferenceLine
from nanofem.numerics.reference.quadrilateral import ReferenceQuadrilateral
from nanofem.numerics.reference.registry import (
    REFERENCE_ELEMENTS,
    cell_type_of_name,
    reference_element,
    reference_element_for_name,
    reference_element_from_dict,
)
from nanofem.numerics.reference.triangle import ReferenceTriangle

__all__ = [
    # enumerations
    "CellType",
    "Dimension",
    "EntityType",
    "FacetType",
    "Orientation",
    # errors
    "ReferenceElementError",
    "TopologyError",
    "OrientationError",
    # reference element base and concrete shapes
    "ReferenceElement",
    "ReferenceLine",
    "ReferenceTriangle",
    "ReferenceQuadrilateral",
    # 3-D placeholders
    "ReferenceTetrahedron",
    "ReferenceHexahedron",
    "ReferencePrism",
    "ReferencePyramid",
    # registry and factories
    "REFERENCE_ELEMENTS",
    "reference_element",
    "reference_element_for_name",
    "reference_element_from_dict",
    "cell_type_of_name",
    # legacy lightweight cell registry
    "ReferenceCell",
    "REFERENCE_CELLS",
    "reference_cell",
]
