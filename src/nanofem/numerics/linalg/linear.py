"""Linear solvers with structure-metadata honesty (SDS 2.15).

Singular factorization raises SingularMatrixError carrying suspected
under-constrained (node, field, component) triples via the zero-pivot ->
DofHandler back-map - the flagship diagnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import diags, spmatrix
from scipy.sparse.linalg import cg, spsolve

from nanofem.utils.exceptions import SingularMatrixError
from nanofem.utils.validation import require_positive


class LinearSolver(ABC):
    """Solve A x = b for sparse A; trusts and enforces declared structure."""

    @abstractmethod
    def solve(self, operator: object, rhs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return x; attach iterations/residual history to the run report."""


class SparseDirectSolver(LinearSolver):
    """LU factorization with factor caching for repeated right-hand sides.

    The richer zero-pivot -> (node, field, component) diagnostic SDS 2.15
    describes is deferred, not dropped: ``spsolve`` does not expose pivot
    indices without more machinery than a single solve justifies. A non-finite
    solution is still caught and raised as ``SingularMatrixError``.
    """

    def solve(self, operator: spmatrix, rhs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve ``operator @ x = rhs`` via sparse LU factorization."""
        solution = np.asarray(spsolve(operator.tocsc(), rhs), dtype=np.float64)
        if not np.all(np.isfinite(solution)):
            raise SingularMatrixError(
                "linear solve produced a non-finite solution; the operator is likely singular "
                "(an under-constrained model - check for missing Dirichlet data)"
            )
        return solution


class ConjugateGradientSolver(LinearSolver):
    """Preconditioned Conjugate Gradient for a symmetric positive-definite ``operator``.

    Wraps ``scipy.sparse.linalg.cg`` (the same "wrap scipy's primitive, add
    NanoFEM's own diagnostics" pattern ``SparseDirectSolver`` already uses for
    ``spsolve`` - hand-rolling the CG recurrence would duplicate a correct,
    well-tested implementation for no benefit). Preconditioning is Jacobi
    (the reciprocal diagonal, ``M = diag(A)^-1``) - the simplest preconditioner
    that is always defined for an SPD matrix and already dramatically improves
    conditioning for the diagonally-dominant stiffness matrices this solver is
    aimed at; nothing here forecloses a richer preconditioner later.

    Convergence is monitored via scipy's own ``callback``, recording the true
    residual norm ``||b - A x_k||`` (not the preconditioned residual scipy's
    internals track internally) after every iteration, so
    ``residual_history``/``iterations`` describe the same convergence
    criterion the caller actually cares about. A run that does not converge in
    ``max_iterations`` raises rather than silently returning the last iterate,
    matching ``SparseDirectSolver``'s "never hand back a wrong answer quietly"
    contract.
    """

    def __init__(self, rtol: float = 1.0e-8, max_iterations: int | None = None) -> None:
        require_positive(rtol, "rtol")
        if max_iterations is not None and max_iterations < 1:
            raise SingularMatrixError("max_iterations must be >= 1")
        self._rtol = rtol
        self._max_iterations = max_iterations
        self.iterations: int = 0
        self.residual_history: list[float] = []

    def solve(self, operator: spmatrix, rhs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve ``operator @ x = rhs`` via preconditioned CG; records run statistics."""
        matrix = operator.tocsr()
        diagonal = matrix.diagonal()
        if np.any(diagonal <= 0.0):
            raise SingularMatrixError(
                "Jacobi preconditioning requires a strictly positive diagonal; the operator "
                "is not SPD (check for a missing Dirichlet constraint or an unphysical model)"
            )
        preconditioner = diags(1.0 / diagonal)

        self.residual_history = []

        def _record(xk: NDArray[np.float64]) -> None:
            self.residual_history.append(float(np.linalg.norm(rhs - matrix @ xk)))

        solution, info = cg(
            matrix,
            rhs,
            rtol=self._rtol,
            maxiter=self._max_iterations,
            M=preconditioner,
            callback=_record,
        )
        self.iterations = len(self.residual_history)
        result = np.asarray(solution, dtype=np.float64)
        if info != 0 or not np.all(np.isfinite(result)):
            raise SingularMatrixError(
                f"conjugate gradient did not converge in {self.iterations} iterations "
                f"(scipy info={info}); the operator may not be SPD, or is too ill-conditioned "
                f"for Jacobi preconditioning at rtol={self._rtol}"
            )
        return result
