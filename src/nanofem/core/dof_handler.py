"""DOF records and deterministic global numbering (SDS C-2, C-5).

Numbering is a pure function of (mesh, field specs): node-major, then field
declaration order, then component order. Phase 1 attaches every field to
every node; element-signature-driven attachment (variable DOFs per node)
arrives with elements in phase 2 - the data model already permits it because
the index is keyed per (node, field, component), never by a uniform stride.

A field declared ``Continuity.DISCONTINUOUS`` is not shared through nodes: each
cell owns its own copy of the field at each of its nodes, keyed per (cell,
node, field, component). Cell-owned DOFs are numbered after every nodal DOF,
so a model without discontinuous fields keeps exactly the numbering (and
fingerprint) it had before the capability existed.
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
    """One global degree of freedom: identity, owner, and constraint status.

    A nodal DOF (``cell_id`` None) is shared by every cell around its node. A
    cell-owned DOF belongs to one cell only; ``node_id`` then records the node
    it sits at, so a discontinuous field has one DOF per (cell, node).
    """

    global_id: int
    node_id: int
    field: str
    component: str
    constrained: bool = False
    cell_id: int | None = None


class DofHandler:
    """Deterministic (node | cell+node, field, component) -> global equation number map."""

    def __init__(self, fields: tuple[FieldSpec, ...], dofs: tuple[Dof, ...]) -> None:
        if not fields:
            raise InputValidationError("DofHandler requires at least one field")
        names = [f.name for f in fields]
        if len(set(names)) != len(names):
            raise InputValidationError(f"duplicate field names: {names}")
        self._fields = fields
        self._dofs = dofs
        self._discontinuous = frozenset(
            f.name for f in fields if f.continuity is Continuity.DISCONTINUOUS
        )
        self._index: dict[tuple[int, str, str], int] = {}
        self._cell_index: dict[tuple[int, int, str, str], int] = {}
        self._by_node: dict[int, list[Dof]] = {}
        self._by_cell: dict[int, list[Dof]] = {}
        for d in dofs:
            if d.cell_id is None:
                self._index[(d.node_id, d.field, d.component)] = d.global_id
                self._by_node.setdefault(d.node_id, []).append(d)
            else:
                self._cell_index[(d.cell_id, d.node_id, d.field, d.component)] = d.global_id
                self._by_cell.setdefault(d.cell_id, []).append(d)

    @classmethod
    def generate(cls, mesh: Mesh, fields: tuple[FieldSpec, ...]) -> DofHandler:
        """Number every DOF in SDS C-2 order.

        Nodal fields first: node-major, then field declaration order, then
        component order. Discontinuous fields follow: cell-major, then the
        cell's own node order, then field, then component.
        """
        nodal = [f for f in fields if f.continuity is not Continuity.DISCONTINUOUS]
        broken = [f for f in fields if f.continuity is Continuity.DISCONTINUOUS]
        dofs: list[Dof] = []
        gid = 0
        for node_id in range(mesh.num_nodes):
            for spec in nodal:
                for comp in spec.components:
                    dofs.append(Dof(gid, node_id, spec.name, comp))
                    gid += 1
        if broken:
            for cell_id in range(mesh.num_cells):
                for node_id in mesh.cell(cell_id).connectivity:
                    for spec in broken:
                        for comp in spec.components:
                            dofs.append(Dof(gid, node_id, spec.name, comp, cell_id=cell_id))
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
        if field in self._discontinuous:
            raise DofMappingError(
                f"field '{field}' is discontinuous: its DOFs are owned by cells, not nodes; "
                f"use cell_dof(cell_id, node_id, field, component)"
            )
        try:
            return self._index[(node_id, field, component)]
        except KeyError:
            raise DofMappingError(
                f"no DOF for node {node_id}, field '{field}', component '{component}'"
            ) from None

    def cell_dof(self, cell_id: int, node_id: int, field: str, component: str) -> int:
        """Equation number of one cell-owned unknown of a discontinuous field."""
        try:
            return self._cell_index[(cell_id, node_id, field, component)]
        except KeyError:
            raise DofMappingError(
                f"no cell-owned DOF for cell {cell_id}, node {node_id}, field '{field}', "
                f"component '{component}'"
            ) from None

    def dof(self, global_id: int) -> Dof:
        """DOF record by global id."""
        if not 0 <= global_id < len(self._dofs):
            raise DofMappingError(f"global DOF id {global_id} out of range [0, {len(self._dofs)})")
        return self._dofs[global_id]

    def dofs_of_node(self, node_id: int) -> tuple[Dof, ...]:
        """Nodal DOFs attached to a node (cell-owned DOFs located there: dofs_of_cell)."""
        return tuple(self._by_node.get(node_id, ()))

    def dofs_of_cell(self, cell_id: int) -> tuple[Dof, ...]:
        """Cell-owned DOFs of one cell, in numbering order (empty without broken fields)."""
        return tuple(self._by_cell.get(cell_id, ()))

    # ---- export / import / determinism witness ----
    def export_numbering(self) -> dict[str, object]:
        """JSON-safe numbering record (restart contract, SDS Section 7).

        Nodal DOFs export as ``[id, node, field, component]``; cell-owned DOFs
        append their cell id as a fifth entry, so records of models without
        discontinuous fields are unchanged.
        """
        return {
            "fields": [f.to_dict() for f in self._fields],
            "dofs": [
                [d.global_id, d.node_id, d.field, d.component]
                + ([] if d.cell_id is None else [d.cell_id])
                for d in self._dofs
            ],
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
        dofs = tuple(_dof_from_record(row) for row in raw_dofs)
        expected = list(range(len(dofs)))
        if [d.global_id for d in dofs] != expected:
            raise DofMappingError("imported numbering is not contiguous from zero")
        return cls(fields, dofs)

    def fingerprint(self) -> str:
        """SHA-256 of the canonical numbering (SDS C-5 witness)."""
        payload = json.dumps(self.export_numbering(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


def _dof_from_record(row: object) -> Dof:
    """One exported DOF row: 4 entries (nodal) or 5 (cell-owned); anything else raises."""
    if not isinstance(row, list) or len(row) not in (4, 5):
        raise DofMappingError(f"malformed DOF record {row!r}: expected 4 or 5 entries")
    cell_id = int(row[4]) if len(row) == 5 else None
    return Dof(int(row[0]), int(row[1]), str(row[2]), str(row[3]), cell_id=cell_id)
