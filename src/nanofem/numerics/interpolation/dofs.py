"""Degrees of freedom as functionals, and interpolation nodes as located points.

A finite element is a triple (domain, polynomial space, degrees of freedom),
where each degree of freedom is a linear *functional* on the space. This
module describes those functionals and the points they act at. It does not
build the basis dual to them - that basis is the shape functions, and it
belongs to a later phase.

The one computation here is :meth:`DofFunctional.apply_to_monomial`, which
applies a functional to a single monomial. For a point value that is
``x^alpha``; for a derivative functional it is the derivative of a *monomial*,
elementary calculus on an exponent tuple. This is not a shape function
derivative: no nodal basis exists at this layer to differentiate. The results
fill the generalized Vandermonde whose invertibility is what
:meth:`Interpolation.verify_linear_independence` checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from nanofem.numerics.interpolation.enums import DofKind
from nanofem.numerics.reference.enums import EntityType
from nanofem.utils.exceptions import InputValidationError

#: Short tag of each entity role, used to build readable identifiers.
_ENTITY_TAG: dict[EntityType, str] = {
    EntityType.VERTEX: "v",
    EntityType.EDGE: "e",
    EntityType.FACE: "f",
    EntityType.CELL: "c",
}


def _entity_identifier(entity: EntityType, entity_index: int, local_index: int) -> str:
    """Identifier of a point by its entity association, e.g. ``"v2"`` or ``"e0.1"``.

    A vertex hosts exactly one point, so its tag carries no local index; an
    edge, face, or cell may host several, which the local index distinguishes.
    """
    tag = f"{_ENTITY_TAG[entity]}{entity_index}"
    if entity is EntityType.VERTEX:
        return tag
    return f"{tag}.{local_index}"


@dataclass(frozen=True)
class InterpolationNode:
    """A located point of an interpolation, with its entity association.

    A node is a *geometric* location; one or more degrees of freedom may act
    at it (Lagrange attaches one, Hermite attaches a value and derivatives).
    ``entity``/``entity_index`` record which sub-entity of the reference
    element owns the node - the information that later decides which DOFs two
    neighbouring cells share, and which validation checks geometrically.
    """

    index: int
    coordinates: tuple[float, ...]
    entity: EntityType
    entity_index: int
    local_index: int = 0

    def __post_init__(self) -> None:
        """Validate the node's index, coordinates, and entity association."""
        if self.index < 0:
            raise InputValidationError(f"node index must be >= 0, got {self.index}")
        if not 1 <= len(self.coordinates) <= 3:
            raise InputValidationError(
                f"node {self.index}: coordinates must have 1..3 entries, "
                f"got {len(self.coordinates)}"
            )
        if any(not math.isfinite(x) for x in self.coordinates):
            raise InputValidationError(f"node {self.index}: coordinates are not all finite")
        if self.entity_index < 0 or self.local_index < 0:
            raise InputValidationError(f"node {self.index}: entity indices must be >= 0")

    @property
    def identifier(self) -> str:
        """Readable identifier from the entity association, e.g. ``"v2"``, ``"e0.1"``."""
        return _entity_identifier(self.entity, self.entity_index, self.local_index)

    @property
    def dimension(self) -> int:
        """Spatial dimension of the node's coordinates."""
        return len(self.coordinates)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible record of the node."""
        return {
            "index": self.index,
            "identifier": self.identifier,
            "coordinates": list(self.coordinates),
            "entity": self.entity.value,
            "entity_index": self.entity_index,
            "local_index": self.local_index,
        }


@dataclass(frozen=True)
class DofFunctional:
    """One degree of freedom, expressed as a linear functional on the space.

    ``derivative`` is a multi-index: all zeros means the point value
    ``u -> u(x)``; ``(1, 0)`` means ``u -> du/dxi (x)``; ``(1, 1)`` means the
    cross derivative ``u -> d2u/dxi deta (x)`` used by the bicubic Hermite
    quadrilateral.
    """

    index: int
    node_index: int
    point: tuple[float, ...]
    derivative: tuple[int, ...]
    entity: EntityType
    entity_index: int
    local_index: int = 0

    def __post_init__(self) -> None:
        """Validate the functional's indices and multi-index."""
        if self.index < 0 or self.node_index < 0:
            raise InputValidationError(f"dof {self.index}: indices must be >= 0")
        if len(self.derivative) != len(self.point):
            raise InputValidationError(
                f"dof {self.index}: derivative multi-index has {len(self.derivative)} entries "
                f"but the point has {len(self.point)}"
            )
        if any(order < 0 for order in self.derivative):
            raise InputValidationError(
                f"dof {self.index}: derivative multi-index {self.derivative} has a negative order"
            )
        if any(not math.isfinite(x) for x in self.point):
            raise InputValidationError(f"dof {self.index}: point is not all finite")

    @property
    def kind(self) -> DofKind:
        """Whether this functional is a point value or a point derivative."""
        return DofKind.POINT_VALUE if self.order == 0 else DofKind.POINT_DERIVATIVE

    @property
    def order(self) -> int:
        """Total differentiation order of the functional (0 for a point value)."""
        return sum(self.derivative)

    @property
    def identifier(self) -> str:
        """Readable identifier, e.g. ``"v0"`` (value) or ``"v1_d10"`` (d/dxi at vertex 1)."""
        base = _entity_identifier(self.entity, self.entity_index, self.local_index)
        if self.order == 0:
            return base
        suffix = "".join(str(order) for order in self.derivative)
        return f"{base}_d{suffix}"

    def apply_to_monomial(self, exponents: tuple[int, ...]) -> float:
        """Apply this functional to the monomial ``x^exponents``.

        Elementary calculus on an exponent tuple:
        ``D^alpha x^beta = prod_i [beta_i! / (beta_i - alpha_i)!] x^(beta - alpha)``,
        which vanishes when any ``beta_i < alpha_i``. For a point value
        (``alpha = 0``) this reduces to evaluating the monomial.

        This differentiates a *monomial*, not a shape function: no nodal basis
        exists at this layer. The values fill the generalized Vandermonde used
        as the unisolvence oracle.
        """
        if len(exponents) != len(self.point):
            raise InputValidationError(
                f"dof {self.index}: monomial has {len(exponents)} exponents, "
                f"expected {len(self.point)}"
            )
        result = 1.0
        for power, order, coordinate in zip(exponents, self.derivative, self.point, strict=True):
            if power < order:
                return 0.0
            result *= float(math.perm(power, order)) * float(coordinate) ** (power - order)
        return result

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible record of the functional."""
        return {
            "index": self.index,
            "identifier": self.identifier,
            "kind": self.kind.value,
            "node_index": self.node_index,
            "point": list(self.point),
            "derivative": list(self.derivative),
            "order": self.order,
            "entity": self.entity.value,
            "entity_index": self.entity_index,
        }
