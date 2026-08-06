"""Assembler + SparsityPattern: a patch-test-style collinear bar chain (SDS 2.13)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from nanofem.numerics.assembly.assembler import Assembler
from nanofem.numerics.assembly.contributions import (
    Contribution,
    ContributionKind,
    ContributionProvider,
    OperatorRole,
)
from nanofem.numerics.assembly.sparsity import SparsityPattern
from nanofem.utils.exceptions import AssemblyError


class _FakeBar:
    """A minimal STIFFNESS-only provider: one 2x2 axial block on two global DOFs."""

    def __init__(self, dof_a: int, dof_b: int, stiffness: float) -> None:
        self._dofs = np.array([dof_a, dof_b], dtype=np.int64)
        self._block = stiffness * np.array([[1.0, -1.0], [-1.0, 1.0]])

    def contributions(self, role: OperatorRole) -> Iterator[Contribution]:
        if role is OperatorRole.STIFFNESS:
            yield Contribution(ContributionKind.CELL, role, self._dofs, self._dofs, self._block)


def _chain(num_dofs: int = 4) -> list[_FakeBar]:
    """A collinear chain of unit-stiffness bars: dof i to dof i+1."""
    return [_FakeBar(i, i + 1, 1.0) for i in range(num_dofs - 1)]


def test_sparsity_pattern_hand_count() -> None:
    """3 bars over 4 shared DOFs touch 10 distinct (row, col) pairs by hand count."""
    providers = _chain()
    pattern = SparsityPattern.from_providers(providers, OperatorRole.STIFFNESS, num_dofs=4)
    assert pattern.num_nonzeros() == 10
    assert pattern.contains(1, 2)
    assert not pattern.contains(0, 3)


def test_assembled_stiffness_is_symmetric_and_matches_hand_matrix() -> None:
    providers = _chain()
    pattern = SparsityPattern.from_providers(providers, OperatorRole.STIFFNESS, num_dofs=4)
    k = Assembler(pattern).assemble(providers, OperatorRole.STIFFNESS)
    assert isinstance(k, csr_matrix)
    dense = k.toarray()
    expected = np.array(
        [
            [1.0, -1.0, 0.0, 0.0],
            [-1.0, 2.0, -1.0, 0.0],
            [0.0, -1.0, 2.0, -1.0],
            [0.0, 0.0, -1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(dense, expected)
    np.testing.assert_allclose(dense, dense.T)


def test_rigid_body_translation_produces_zero_net_force() -> None:
    providers = _chain()
    pattern = SparsityPattern.from_providers(providers, OperatorRole.STIFFNESS, num_dofs=4)
    k = Assembler(pattern).assemble(providers, OperatorRole.STIFFNESS)
    residual = k @ np.ones(4)
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)


def test_pattern_violation_raises_assembly_error() -> None:
    provider = _FakeBar(0, 3, 1.0)
    pattern = SparsityPattern.from_providers(
        [_FakeBar(0, 1, 1.0)], OperatorRole.STIFFNESS, num_dofs=4
    )
    with pytest.raises(AssemblyError, match="outside the precomputed sparsity pattern"):
        Assembler(pattern).assemble([provider], OperatorRole.STIFFNESS)


def test_mixed_matrix_and_vector_contributions_raise() -> None:
    class _VectorProvider:
        def contributions(self, role: OperatorRole) -> Iterator[Contribution]:
            if role is OperatorRole.STIFFNESS:
                yield Contribution(
                    ContributionKind.VERTEX,
                    role,
                    np.array([0], dtype=np.int64),
                    None,
                    np.array([1.0]),
                )

    providers: list[ContributionProvider] = [_FakeBar(0, 1, 1.0), _VectorProvider()]
    pattern = SparsityPattern.from_providers(
        [_FakeBar(0, 1, 1.0)], OperatorRole.STIFFNESS, num_dofs=4
    )
    with pytest.raises(AssemblyError, match="after another provider emitted"):
        Assembler(pattern).assemble(providers, OperatorRole.STIFFNESS)


def test_asymmetric_block_violates_c7_and_raises() -> None:
    class _AsymmetricProvider:
        def contributions(self, role: OperatorRole) -> Iterator[Contribution]:
            if role is OperatorRole.STIFFNESS:
                dofs = np.array([0, 1], dtype=np.int64)
                yield Contribution(
                    ContributionKind.CELL, role, dofs, dofs, np.array([[1.0, -2.0], [-1.0, 1.0]])
                )

    provider = _AsymmetricProvider()
    pattern = SparsityPattern.from_providers([provider], OperatorRole.STIFFNESS, num_dofs=4)
    with pytest.raises(AssemblyError, match="C-7"):
        Assembler(pattern).assemble([provider], OperatorRole.STIFFNESS)
