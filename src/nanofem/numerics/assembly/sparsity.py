"""Sparsity pattern precomputation from provider DOF maps (SDS 2.13).

Pattern identity across repeated assembly is contract (factorization reuse);
pattern and emission derive from one source, so a violation names the
offending provider.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.assembly.contributions import ContributionProvider, OperatorRole


class SparsityPattern:
    """Nonzero structure computed once, reused across dynamics/Newton assembly."""

    def __init__(
        self,
        num_dofs: int,
        row_indices: NDArray[np.int64],
        col_indices: NDArray[np.int64],
    ) -> None:
        self.num_dofs = num_dofs
        self._pairs: frozenset[tuple[int, int]] = frozenset(
            zip(
                (int(r) for r in row_indices),
                (int(c) for c in col_indices),
                strict=True,
            )
        )

    @classmethod
    def from_providers(
        cls,
        providers: Iterable[ContributionProvider],
        role: OperatorRole,
        num_dofs: int,
    ) -> SparsityPattern:
        """Union the (row, col) pairs every provider touches for one role.

        Vector contributions (``col_dofs is None``) carry no matrix structure
        and are skipped.
        """
        rows: list[int] = []
        cols: list[int] = []
        for provider in providers:
            for contribution in provider.contributions(role):
                if contribution.col_dofs is None:
                    continue
                for row in contribution.row_dofs:
                    for col in contribution.col_dofs:
                        rows.append(int(row))
                        cols.append(int(col))
        return cls(
            num_dofs,
            np.asarray(rows, dtype=np.int64),
            np.asarray(cols, dtype=np.int64),
        )

    def contains(self, row: int, col: int) -> bool:
        """Return whether ``(row, col)`` lies inside the precomputed pattern."""
        return (row, col) in self._pairs

    def num_nonzeros(self) -> int:
        """Return nnz of the pattern (reported by diagnostics, SDS Section 13)."""
        return len(self._pairs)
