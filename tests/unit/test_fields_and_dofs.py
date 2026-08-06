"""Unit tests for FieldSpec, Dof, and DofHandler (requirements 5-7)."""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.core.dof_handler import DofHandler
from nanofem.core.fields import (
    FieldSpec,
    VariableType,
    component_names,
    displacement,
    electric_potential,
    pressure,
    rotation_z,
    temperature,
)
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.numerics.operators.base import Continuity
from nanofem.utils.exceptions import InputValidationError


def line_mesh() -> Mesh:
    """Three collinear nodes, two line2 cells."""
    coords = np.array([[0.0], [1.0], [2.0]])
    block = CellBlock("line2", np.array([[0, 1], [1, 2]], dtype=np.int64), "bar")
    return Mesh(coords, (block,))


def test_fieldspec_validation_and_metadata() -> None:
    """Components are non-empty, unique; dofs_per_node = component count."""
    u = displacement(2)
    assert u.components == ("x", "y") and u.dofs_per_node == 2
    assert u.variable_type is VariableType.DISPLACEMENT
    with pytest.raises(InputValidationError):
        FieldSpec("u", ())
    with pytest.raises(InputValidationError):
        FieldSpec("u", ("x", "x"))
    assert component_names(4) == ("c0", "c1", "c2", "c3")


def test_dof_numbering_is_deterministic_and_c2_ordered() -> None:
    """Node-major, field declaration order, component order (SDS C-2, C-5)."""
    mesh = line_mesh()
    fields = (displacement(1), FieldSpec("T", ("t",), VariableType.TEMPERATURE))
    dh1 = DofHandler.generate(mesh, fields)
    dh2 = DofHandler.generate(mesh, fields)
    assert dh1.num_dofs == 6  # 3 nodes x (1 + 1)
    assert dh1.fingerprint() == dh2.fingerprint()
    # C-2: node 0 owns gids 0..1 in field order, node 1 owns 2..3, ...
    assert dh1.global_dof(0, "u", "x") == 0
    assert dh1.global_dof(0, "T", "t") == 1
    assert dh1.global_dof(1, "u", "x") == 2
    d = dh1.dof(3)
    assert (d.node_id, d.field, d.component, d.constrained) == (1, "T", "t", False)
    assert tuple(x.global_id for x in dh1.dofs_of_node(2)) == (4, 5)


def test_dof_export_import_round_trip() -> None:
    """Export/import preserves numbering exactly (restart contract, SDS C-5)."""
    dh = DofHandler.generate(line_mesh(), (displacement(1),))
    dh2 = DofHandler.import_numbering(dh.export_numbering())
    assert dh2.num_dofs == dh.num_dofs
    assert dh2.fingerprint() == dh.fingerprint()
    assert dh2.global_dof(2, "u", "x") == dh.global_dof(2, "u", "x")


def test_unknown_dof_lookup_raises_with_context() -> None:
    """Unknown (node, field, component) raises rather than guessing."""
    dh = DofHandler.generate(line_mesh(), (displacement(1),))
    with pytest.raises(Exception, match="u|node|component"):
        dh.global_dof(0, "u", "z")


def test_continuity_lives_on_fieldspec() -> None:
    """FieldSpec records the continuity the discretization must provide."""
    w = FieldSpec("w", ("w",), continuity=Continuity.C1)
    assert w.continuity is Continuity.C1
    assert w.to_dict()["continuity"] == "C1"


def test_remaining_field_factories() -> None:
    """rotation_z/temperature/electric_potential/pressure: single-component scalar fields."""
    r = rotation_z()
    assert r.name == "r" and r.components == ("z",) and r.variable_type is VariableType.ROTATION

    t = temperature()
    assert t.name == "T" and t.components == ("t",) and t.variable_type is VariableType.TEMPERATURE

    phi = electric_potential()
    assert (
        phi.name == "phi"
        and phi.components == ("p",)
        and phi.variable_type is VariableType.ELECTRIC_POTENTIAL
    )

    p = pressure()
    assert p.name == "p" and p.components == ("p",) and p.variable_type is VariableType.PRESSURE
