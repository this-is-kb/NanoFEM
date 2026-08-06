"""DOF records and deterministic global numbering (SDS C-2, C-5).

Numbering is a pure function of (mesh, field specs): node-major, then field
declaration order, then component order. Phase 1 attaches every field to
every node; element-signature-driven attachment (variable DOFs per node)
arrives with elements in phase 2 - the data model already permits it because
the index is keyed per (node, field, component), never by a uniform stride.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nanofem.core.fields import FieldSpec, VariableType
from nanofem.numerics.operators.base import Continuity
from nanofem.utils.exceptions import DofMappingError, InputValidationError

if TYPE_CHECKING:
    from nanofem.mesh.mesh import Mesh


@dataclass(frozen=True)
class Dof:
    """One global degree of freedom: identity, owner, and constraint status."""

    global_id: int
    node_id: int
    field: str
    component: str
    constrained: bool = False


class DofHandler:
    """Deterministic (node, field, component) -> global equation number map."""

    def __init__(self, fields: tuple[FieldSpec, ...], dofs: tuple[Dof, ...]) -> None:
        if not fields:
            raise InputValidationError("DofHandler requires at least one field")
        names = [f.name for f in fields]
        if len(set(names)) != len(names):
            raise InputValidationError(f"duplicate field names: {names}")
        self._fields = fields
        self._dofs = dofs
        self._index: dict[tuple[int, str, str], int] = {
            (d.node_id, d.field, d.component): d.global_id for d in dofs
        }
        self._by_node: dict[int, list[Dof]] = {}
        for d in dofs:
            self._by_node.setdefault(d.node_id, []).append(d)

    @classmethod
    def generate(cls, mesh: Mesh, fields: tuple[FieldSpec, ...]) -> DofHandler:
        """Number every (node, field, component) in SDS C-2 order."""
        dofs: list[Dof] = []
        gid = 0
        for node_id in range(mesh.num_nodes):
            for spec in fields:
                for comp in spec.components:
                    dofs.append(Dof(gid, node_id, spec.name, comp))
                    gid += 1
        return cls(tuple(fields), tuple(dofs))

    # ---- queries ----
    @property
    def num_dofs(self) -> int:
        """Total global DOFs."""
        return len(self._dofs)

    @property
    def fields(self) -> tuple[FieldSpec, ...]:
        """Field specs in declaration order."""
        return self._fields

    def global_dof(self, node_id: int, field: str, component: str) -> int:
        """Equation number of one nodal unknown; misses name the miss."""
        try:
            return self._index[(node_id, field, component)]
        except KeyError:
            raise DofMappingError(
                f"no DOF for node {node_id}, field '{field}', component '{component}'"
            ) from None

    def dof(self, global_id: int) -> Dof:
        """DOF record by global id."""
        if not 0 <= global_id < len(self._dofs):
            raise DofMappingError(f"global DOF id {global_id} out of range [0, {len(self._dofs)})")
        return self._dofs[global_id]

    def dofs_of_node(self, node_id: int) -> tuple[Dof, ...]:
        """All DOFs attached to a node (may vary per node from phase 2 on)."""
        return tuple(self._by_node.get(node_id, ()))

    # ---- export / import / determinism witness ----
    def export_numbering(self) -> dict[str, object]:
        """JSON-safe numbering record (restart contract, SDS Section 7)."""
        return {
            "fields": [f.to_dict() for f in self._fields],
            "dofs": [[d.global_id, d.node_id, d.field, d.component] for d in self._dofs],
        }

    @classmethod
    def import_numbering(cls, data: dict[str, object]) -> DofHandler:
        """Rebuild from an exported record; malformed records raise."""
        raw_fields = data["fields"]
        raw_dofs = data["dofs"]
        if not isinstance(raw_fields, list) or not isinstance(raw_dofs, list):
            raise DofMappingError("malformed numbering record: 'fields'/'dofs' must be lists")
        fields = tuple(
            FieldSpec(
                str(f["name"]),
                tuple(str(c) for c in f["components"]),
                VariableType(str(f["variable_type"])),
                Continuity[str(f["continuity"])],
            )
            for f in raw_fields
        )
        dofs = tuple(Dof(int(g), int(n), str(fl), str(c)) for g, n, fl, c in raw_dofs)
        expected = list(range(len(dofs)))
        if [d.global_id for d in dofs] != expected:
            raise DofMappingError("imported numbering is not contiguous from zero")
        return cls(fields, dofs)

    def fingerprint(self) -> str:
        """SHA-256 of the canonical numbering (SDS C-5 witness)."""
        payload = json.dumps(self.export_numbering(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()
