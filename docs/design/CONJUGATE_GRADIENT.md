# NanoFEM Conjugate Gradient Solver (v0.17.0)

**Status:** implemented and tested. Companion to `numerics/linalg/linear.py`'s `SparseDirectSolver`
(v0.8.0), whose "wrap scipy's primitive, add NanoFEM's own diagnostics" pattern this reuses.

**Scope discipline.** Fills the `ConjugateGradientSolver` stub SDS 2.15 already declared (docstring
only since v0.0.1) with a real preconditioned CG solve. Wraps `scipy.sparse.linalg.cg` rather than
hand-rolling the recurrence - the same choice `SparseDirectSolver` already made for `spsolve`, and
for the same reason: reimplementing a correct, well-tested numerical algorithm from scratch would
be duplicated logic with no benefit, and the directive's own "avoid unnecessary complexity" applies
exactly as much to solver internals as to element formulations. Jacobi (diagonal) preconditioning
only - "basic preconditioning if necessary," per Step 6's own scope, and always defined for an SPD
matrix, no design choice to justify beyond that it works.

---

## 1. What's new versus the LU solver

`SparseDirectSolver.solve()` returns only the solution vector - a direct method has no
"convergence" to report. `ConjugateGradientSolver.solve()` also records `iterations` and
`residual_history` (the *true* residual `||b - A x_k||` after every iteration, via scipy's own
`callback`, not the preconditioned residual scipy tracks internally for its own stopping test) as
instance attributes after the call returns - "solver statistics"/"convergence monitoring," Step
6's other two named requirements, without changing `LinearSolver`'s one-method ABC (both solvers
stay interchangeable: same `solve(operator, rhs) -> NDArray` signature).

A run that does not converge within `max_iterations` raises `SingularMatrixError` rather than
returning the best iterate found - matching `SparseDirectSolver`'s existing "never hand back a
wrong answer quietly" contract (its own non-finite-solution check exists for the same reason).

## 2. Not wired into `LinearStaticAnalysis`

`analysis/static.py` still hardcodes `SparseDirectSolver()`. Adding solver selection to
`StaticOptions`/`LinearStaticAnalysis` is a real, separate feature (a new constructor parameter, a
default-solver policy, documentation of when to prefer one over the other) that Step 6's own
success criterion - "implement [a CG solver]" - does not itself require, so it is left for a later
increment if a concrete need for it arises, matching the established "one thing at a time, fully
verified" sequencing this project has used since `Bar` -> `EulerBernoulliBeam` -> `TimoshenkoBeam`.
`ConjugateGradientSolver` is fully usable standalone today (`solver.solve(reduced.k_ff,
reduced.f_f)`, exactly like `SparseDirectSolver`), which is what the tests exercise.

---

## 3. Verification

Two independent checks, not one: a synthetic random SPD system (`A = M M^T + n I`, guaranteeing
positive-definiteness by construction, with a known exact solution `x_true` planted before solving)
and the real T3 plate stiffness matrix this session's other tests already build (a genuine,
non-toy FEM system). Both must match `SparseDirectSolver`'s LU answer to the requested `rtol` -
two different solution methods converging on the same unique answer for a nonsingular system is a
real cross-check, not a restatement.

`tests/unit/test_linalg_cg_analytical.py` (6 tests): CG matches both the planted exact solution and
LU on the synthetic system; `iterations`/`residual_history` populate correctly and the final
residual is below the requested tolerance; CG matches LU on the real FEM system; a non-SPD
(negative-diagonal) matrix is rejected before iterating; a deliberately too-small `max_iterations`
raises rather than returning a non-converged iterate; constructor parameter validation
(`rtol > 0`, `max_iterations >= 1`).
