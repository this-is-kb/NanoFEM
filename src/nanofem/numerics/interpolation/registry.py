"""Registry and factories for interpolation families.

Maps ``(family, cell_type, order)`` to concrete
:class:`~nanofem.numerics.interpolation.base.Interpolation` classes, provides
a factory accepting enums or strings, reports what is available, and
reconstructs an element from a serialized dictionary.

The registry is local to this package rather than reusing
``nanofem.core.Registry``: ``core`` sits above ``numerics`` in the layer
order, so importing it here would invert the dependency rule. Same reasoning,
same shape, as the reference-element registry.
"""

from __future__ import annotations

from typing import Any, Protocol

from nanofem.numerics.interpolation.base import Interpolation
from nanofem.numerics.interpolation.enums import InterpolationFamily
from nanofem.numerics.interpolation.future import (
    HierarchicalInterpolation,
    SerendipityInterpolation,
    SpectralInterpolation,
)
from nanofem.numerics.interpolation.hermite import HermiteInterpolation
from nanofem.numerics.interpolation.lagrange import LagrangeInterpolation
from nanofem.numerics.reference.enums import CellType
from nanofem.utils.exceptions import InputValidationError
from nanofem.utils.validation import resolve_enum_member


class InterpolationFactory(Protocol):
    """The constructor signature every interpolation family satisfies.

    ``type[Interpolation]`` cannot express this: the ABC declares no
    ``__init__``, so mypy would reject constructing a class drawn from the
    registry. The protocol states what the registry actually relies on.
    """

    def __call__(self, cell_type: CellType | str, order: int) -> Interpolation:
        """Build the interpolation for a cell shape and order."""


#: Concrete class per family, including the declared placeholders.
INTERPOLATION_FAMILIES: dict[InterpolationFamily, InterpolationFactory] = {
    InterpolationFamily.LAGRANGE: LagrangeInterpolation,
    InterpolationFamily.HERMITE: HermiteInterpolation,
    InterpolationFamily.SERENDIPITY: SerendipityInterpolation,
    InterpolationFamily.HIERARCHICAL: HierarchicalInterpolation,
    InterpolationFamily.SPECTRAL: SpectralInterpolation,
}

#: Every (family, cell, order) combination implemented in this phase.
AVAILABLE_INTERPOLATIONS: tuple[tuple[InterpolationFamily, CellType, int], ...] = (
    (InterpolationFamily.LAGRANGE, CellType.LINE, 1),
    (InterpolationFamily.LAGRANGE, CellType.LINE, 2),
    (InterpolationFamily.LAGRANGE, CellType.LINE, 3),
    (InterpolationFamily.LAGRANGE, CellType.TRIANGLE, 1),
    (InterpolationFamily.LAGRANGE, CellType.TRIANGLE, 2),
    (InterpolationFamily.LAGRANGE, CellType.TRIANGLE, 3),
    (InterpolationFamily.LAGRANGE, CellType.QUADRILATERAL, 1),
    (InterpolationFamily.LAGRANGE, CellType.QUADRILATERAL, 2),
    (InterpolationFamily.LAGRANGE, CellType.QUADRILATERAL, 3),
    (InterpolationFamily.HERMITE, CellType.LINE, 3),
    (InterpolationFamily.HERMITE, CellType.QUADRILATERAL, 3),
)


def interpolation(
    family: InterpolationFamily | str,
    cell_type: CellType | str,
    order: int,
) -> Interpolation:
    """Return the interpolation for ``(family, cell_type, order)``.

    Accepts enums or their string values. Unimplemented families raise
    ``NotImplementedError`` naming what they still need; unsupported
    cell/order combinations raise ``InputValidationError``.
    """
    resolved_family = _resolve_family(family)
    resolved_cell = _resolve_cell_type(cell_type)
    element_class = INTERPOLATION_FAMILIES[resolved_family]
    return element_class(resolved_cell, order)


def available_interpolations() -> tuple[tuple[InterpolationFamily, CellType, int], ...]:
    """Every ``(family, cell, order)`` combination this phase implements."""
    return AVAILABLE_INTERPOLATIONS


def interpolation_from_dict(data: dict[str, Any]) -> Interpolation:
    """Reconstruct an interpolation from an :meth:`Interpolation.to_dict` payload.

    An interpolation is canonical per ``(family, cell_type, order)``, so those
    three keys rebuild it. If the payload also carries ``num_dofs``, it is
    checked against the reconstruction so tampering is caught.
    """
    for key in ("family", "cell_type", "order"):
        if key not in data:
            raise InputValidationError(f"interpolation payload missing {key!r}")
    element = interpolation(str(data["family"]), str(data["cell_type"]), int(data["order"]))
    declared = data.get("num_dofs")
    if declared is not None and int(declared) != element.num_dofs:
        raise InputValidationError(
            f"payload declares {declared} dofs but canonical "
            f"{element.family.value} order {element.order} on {element.cell_type.value} "
            f"has {element.num_dofs}"
        )
    return element


def _resolve_family(family: InterpolationFamily | str) -> InterpolationFamily:
    return resolve_enum_member(family, InterpolationFamily, "interpolation family")


def _resolve_cell_type(cell_type: CellType | str) -> CellType:
    return resolve_enum_member(cell_type, CellType, "cell type")
