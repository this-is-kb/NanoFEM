"""Scatter of contribution streams into global CSR operators (SDS 2.13)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from nanofem.numerics.assembly.contributions import ContributionProvider, OperatorRole
from nanofem.numerics.assembly.sparsity import SparsityPattern
from nanofem.utils.exceptions import AssemblyError

_SYMMETRY_ATOL = 1e-9


class Assembler:
    """COO-triplet scatter with a precomputed pattern; duplicate (i, j) entries sum."""

    def __init__(self, pattern: SparsityPattern) -> None:
        self._pattern = pattern

    def assemble(
        self, providers: Iterable[ContributionProvider], role: OperatorRole
    ) -> csr_matrix | NDArray[np.float64]:
        """Return the CSR operator (matrix role) or dense vector for one role (SDS Section 11).

        A role's providers must be exclusively matrix or exclusively vector
        emitters; mixing the two for one role is a modeling error, not a
        legitimate composition, and raises ``AssemblyError``.
        """
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        vector = np.zeros(self._pattern.num_dofs, dtype=np.float64)
        seen_matrix = False
        seen_vector = False

        for provider in providers:
            for contribution in provider.contributions(role):
                block = contribution.block
                row_dofs = contribution.row_dofs

                if contribution.col_dofs is None:
                    if seen_matrix:
                        raise AssemblyError(
                            f"{provider!r} emits a vector contribution for role {role} after "
                            "another provider emitted a matrix contribution for the same role"
                        )
                    seen_vector = True
                    if block.shape != (len(row_dofs),):
                        raise AssemblyError(
                            f"{provider!r} vector contribution block shape {block.shape} does "
                            f"not match (len(rows),) = {(len(row_dofs),)}"
                        )
                    np.add.at(vector, row_dofs, block)
                    continue

                if seen_vector:
                    raise AssemblyError(
                        f"{provider!r} emits a matrix contribution for role {role} after "
                        "another provider emitted a vector contribution for the same role"
                    )
                seen_matrix = True
                col_dofs = contribution.col_dofs
                expected_shape = (len(row_dofs), len(col_dofs))
                if block.shape != expected_shape:
                    raise AssemblyError(
                        f"{provider!r} contribution block shape {block.shape} does not match "
                        f"(len(rows), len(cols)) = {expected_shape}"
                    )
                if (
                    __debug__
                    and block.shape[0] == block.shape[1]
                    and not np.allclose(block, block.T, atol=_SYMMETRY_ATOL)
                ):
                    raise AssemblyError(
                        f"{provider!r} contribution for role {role} violates C-7 declared symmetry"
                    )
                for i, row in enumerate(row_dofs):
                    for j, col in enumerate(col_dofs):
                        if not self._pattern.contains(int(row), int(col)):
                            raise AssemblyError(
                                f"{provider!r} touches ({row}, {col}) outside the precomputed "
                                "sparsity pattern"
                            )
                        rows.append(int(row))
                        cols.append(int(col))
                        data.append(float(block[i, j]))

        if seen_vector:
            return vector
        n = self._pattern.num_dofs
        return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
