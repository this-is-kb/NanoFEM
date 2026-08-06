"""Registry and factories for reference elements.

Maps :class:`~nanofem.numerics.reference.enums.CellType` to concrete
:class:`~nanofem.numerics.reference.element.ReferenceElement` classes, provides
a string/enum factory, a bridge from the mesh cell-type *names* used by the
existing :class:`~nanofem.numerics.reference.cell.ReferenceCell` registry (for
example ``"tri6"``) to the topological ``CellType``, and reconstruction from a
serialized dictionary.

Relationship to ``ReferenceCell``
---------------------------------
``ReferenceCell`` (in ``cell.py``) is a lightweight record keyed by a mesh
cell-type *name* carrying a node count and basic counts; it is what the mesh,
quadrature, and interpolation layers look up. ``ReferenceElement`` is the full
geometric/topological reference *domain*, independent of interpolation order.
The two are related by :func:`cell_type_of_name`, which strips the order suffix
(``"tri3"`` and ``"tri6"`` both map to :pyattr:`CellType.TRIANGLE`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nanofem.numerics.reference.element import ReferenceElement
from nanofem.numerics.reference.enums import CellType
from nanofem.numerics.reference.line import ReferenceLine
from nanofem.numerics.reference.quadrilateral import ReferenceQuadrilateral
from nanofem.numerics.reference.triangle import ReferenceTriangle
from nanofem.utils.exceptions import InputValidationError
from nanofem.utils.validation import resolve_enum_member

#: The reference elements implemented in this phase, keyed by shape.
REFERENCE_ELEMENTS: dict[CellType, type[ReferenceElement]] = {
    CellType.LINE: ReferenceLine,
    CellType.TRIANGLE: ReferenceTriangle,
    CellType.QUADRILATERAL: ReferenceQuadrilateral,
}

#: Mesh cell-type name -> topological cell type (order suffix stripped).
_NAME_TO_CELL_TYPE: dict[str, CellType] = {
    "line2": CellType.LINE,
    "line3": CellType.LINE,
    "line4": CellType.LINE,
    "tri3": CellType.TRIANGLE,
    "tri6": CellType.TRIANGLE,
    "tri10": CellType.TRIANGLE,
    "quad4": CellType.QUADRILATERAL,
    "quad8": CellType.QUADRILATERAL,
    "quad9": CellType.QUADRILATERAL,
    "quad16": CellType.QUADRILATERAL,
}


def reference_element(cell_type: CellType | str) -> ReferenceElement:
    """Return a reference element instance for ``cell_type``.

    Accepts a :class:`CellType` or its string value. Unimplemented shapes
    (the 3-D placeholders) raise ``NotImplementedError``; unknown strings raise
    ``InputValidationError`` listing the implemented shapes.
    """
    resolved = _resolve_cell_type(cell_type)
    element_class = REFERENCE_ELEMENTS.get(resolved)
    if element_class is None:
        implemented = ", ".join(sorted(ct.value for ct in REFERENCE_ELEMENTS))
        raise NotImplementedError(
            f"reference element for {resolved.value!r} is not implemented; "
            f"implemented shapes: {implemented}"
        )
    return element_class()


def cell_type_of_name(name: str) -> CellType:
    """Map a mesh cell-type name (for example ``"tri6"``) to its topological cell type."""
    try:
        return _NAME_TO_CELL_TYPE[name]
    except KeyError:
        known = ", ".join(sorted(_NAME_TO_CELL_TYPE))
        raise InputValidationError(f"unknown cell name {name!r}; known: {known}") from None


def reference_element_for_name(name: str) -> ReferenceElement:
    """Return a reference element for a mesh cell-type name (order suffix ignored)."""
    return reference_element(cell_type_of_name(name))


def reference_element_from_dict(data: dict[str, Any]) -> ReferenceElement:
    """Reconstruct a reference element from a :meth:`ReferenceElement.to_dict` payload.

    Since a reference element is canonical per shape, only ``cell_type`` is
    required to rebuild it; if ``vertex_coordinates`` are present they are
    checked against the canonical geometry so tampering is caught.
    """
    try:
        cell_value = data["cell_type"]
    except KeyError:
        raise InputValidationError("reference-element payload missing 'cell_type'") from None
    element = reference_element(str(cell_value))
    vertices = data.get("vertex_coordinates")
    if vertices is not None:
        supplied = np.asarray(vertices, dtype=np.float64)
        if supplied.shape != element.vertex_coordinates.shape or not np.allclose(
            supplied, element.vertex_coordinates
        ):
            raise InputValidationError(
                f"payload vertices do not match the canonical {element.cell_type.value} geometry"
            )
    return element


def _resolve_cell_type(cell_type: CellType | str) -> CellType:
    return resolve_enum_member(cell_type, CellType, "cell type")
