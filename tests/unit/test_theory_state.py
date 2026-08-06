"""Unit tests for theory/constitutive declarations and the state lifecycle (10-12)."""

from __future__ import annotations

import pytest

from nanofem.numerics.assembly.contributions import ContributionKind, OperatorRole
from nanofem.numerics.operators.base import (
    OPERATOR_CATALOG,
    Continuity,
    derived_continuity,
)
from nanofem.physics.base import (
    ConstitutiveDeclaration,
    DeclaredTheory,
    Locality,
    TheoryDeclaration,
)
from nanofem.state.history import StateHistory
from nanofem.state.layout import StateLayout
from nanofem.state.model_state import ModelState
from nanofem.state.quadrature_state import QuadraturePointState
from nanofem.utils.exceptions import PhysicsError, StateError


def local_elasticity() -> TheoryDeclaration:
    """The SDS 4.1 declaration used across these tests."""
    return TheoryDeclaration(
        name="local_elasticity",
        field_requirements=(("u", 2),),
        continuity=(("u", Continuity.C0),),
        required_properties=("E", "nu", "rho"),
        operators=("symmetric_gradient", "voigt_map"),
        roles=(
            OperatorRole.STIFFNESS,
            OperatorRole.MASS,
            OperatorRole.GEOMETRIC_STIFFNESS,
            OperatorRole.FORCE,
        ),
        kinds=(ContributionKind.CELL,),
    )


def test_theory_declaration_cross_checks() -> None:
    """Unknown operators, duplicate fields, and PAIRWISE/PAIR mismatch all raise."""
    local_elasticity()  # valid
    with pytest.raises(PhysicsError, match="catalog"):
        TheoryDeclaration("t", (("u", 2),), (("u", Continuity.C0),), operators=("b_matrix",))
    with pytest.raises(PhysicsError, match="duplicate"):
        TheoryDeclaration("t", (("u", 2), ("u", 1)), ())
    with pytest.raises(PhysicsError, match="imply each other"):
        TheoryDeclaration("t", (("u", 2),), (), locality=Locality.PAIRWISE)


def test_pairwise_theory_declares_pair_kind() -> None:
    """Eringen-integral shape: PAIRWISE locality with CELL + PAIR kinds."""
    d = TheoryDeclaration(
        name="eringen_integral",
        field_requirements=(("u", 1),),
        continuity=(("u", Continuity.C0),),
        required_properties=("E", "rho", "e0a"),
        operators=("symmetric_gradient", "nonlocal_integral"),
        kinds=(ContributionKind.CELL, ContributionKind.PAIR),
        locality=Locality.PAIRWISE,
    )
    theory = DeclaredTheory(d)
    assert theory.locality() is Locality.PAIRWISE
    assert ContributionKind.PAIR in theory.contribution_kinds()


def test_derived_continuity_rule() -> None:
    """SDS Section 8: any second-derivative operator forces C1."""
    assert derived_continuity(("symmetric_gradient",)) is Continuity.C0
    assert derived_continuity(("symmetric_gradient", "second_gradient")) is Continuity.C1
    assert "second_gradient" in OPERATOR_CATALOG


def test_constitutive_declaration_validation() -> None:
    """Response components must be >= 1; layout travels with the declaration."""
    layout = StateLayout((("eps_p", 3),))
    d = ConstitutiveDeclaration("j2", ("E", "nu"), layout, response_components=3)
    assert d.state_layout.variables == (("eps_p", 3),)
    with pytest.raises(PhysicsError):
        ConstitutiveDeclaration("bad", (), StateLayout(()), response_components=0)


def test_state_lifecycle_commit_revert() -> None:
    """Trial/committed banks: writes stay trial until commit; revert discards."""
    s = ModelState()
    s.allocate(StateLayout((("stress", 3),)), n_cells=2, n_qp=2)
    assert s.is_allocated and s.variables == ("stress",)
    s.view("stress")[0, 0, :] = 7.0
    assert float(s.committed_view("stress")[0, 0, 0]) == 0.0
    s.commit()
    assert float(s.committed_view("stress")[0, 0, 0]) == 7.0
    s.view("stress")[0, 0, :] = 99.0
    s.revert()
    assert float(s.view("stress")[0, 0, 0]) == 7.0


def test_empty_layout_allocates_nothing() -> None:
    """Elastic laws declare no state and pay zero bytes (SDS Section 7)."""
    s = ModelState()
    s.allocate(StateLayout(()), n_cells=10, n_qp=4)
    assert s.variables == ()
    with pytest.raises(StateError):
        s.view("stress")


def test_history_snapshot_restore_and_qp_view() -> None:
    """Bounded history restores committed banks; QP views index without copying."""
    s = ModelState()
    s.allocate(StateLayout((("d", 1),)), n_cells=1, n_qp=1)
    h = StateHistory(s, max_depth=2)
    s.view("d")[0, 0, 0] = 1.0
    s.commit()
    h.snapshot()
    s.view("d")[0, 0, 0] = 2.0
    s.commit()
    h.restore()
    assert float(s.committed_view("d")[0, 0, 0]) == 1.0
    qp = QuadraturePointState(s, cell=0, qp=0)
    assert qp.committed("d").shape == (1,)
    with pytest.raises(NotImplementedError):
        h.checkpoint("/tmp/never.json")
