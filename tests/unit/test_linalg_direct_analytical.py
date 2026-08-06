"""SparseDirectSolver: a hand-solved 2x2 SPD system, and the singular failure mode (SDS 2.15)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from nanofem.numerics.linalg.linear import SparseDirectSolver
from nanofem.utils.exceptions import SingularMatrixError


def test_solves_hand_computed_spd_system() -> None:
    """[[4,1],[1,3]] x = [1,2] -> x = [1/11, 7/11] (Cramer's rule, by hand)."""
    a = csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    b = np.array([1.0, 2.0])
    x = SparseDirectSolver().solve(a, b)
    np.testing.assert_allclose(x, [1.0 / 11.0, 7.0 / 11.0])
    np.testing.assert_allclose(a @ x, b)


def test_singular_operator_raises() -> None:
    """A rank-deficient operator (rigid-body mode, no constraints) is caught."""
    a = csr_matrix(np.array([[1.0, -1.0], [-1.0, 1.0]]))
    b = np.array([1.0, 0.0])
    with pytest.raises(SingularMatrixError):
        SparseDirectSolver().solve(a, b)
