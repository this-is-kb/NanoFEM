# NanoFEM Element Integration Framework (v0.9.0)

**Status:** implemented and tested. Companion to `ARCHITECTURE_v2.md`, ADR-011 ("dimension-generic
continuum elements parameterized by `ReferenceCell`"), SDS Section 3 (clauses E-1, E-5, E-6), and
`docs/design/OPERATORS.md`/`docs/dev/notes.md` N-42, whose "not-yet-built `elements/` layer" this
phase finally is.

**Scope discipline.** One new element (`ContinuumElement`), single-field theories only, affine
(straight-sided) geometry only, verified against the 1-D case only (`IsotropicElasticity(dim=1)`).
Real 2-D/3-D continuum elements (T3, Q4, Tet, Hex) are explicitly deferred until plane/solid
elasticity exists (separate, later work) — this phase's job is proving the *integration machinery*
composes correctly, not shipping a full element catalog. `Bar` (v0.8.0, ADR-002) is untouched;
`ContinuumElement` is not wired into `elements/factory.py`'s dispatch this phase.

**Update, v0.13.0:** `ContinuumElement` no longer requires affine geometry. It still tries
`AffineMapping` first, but falls back to `IsoparametricMapping` when `AffineMapping` raises
`NonAffineError` (a general, non-parallelogram Q4) — see `docs/design/Q4_QUADRILATERAL.md`. The
scope note above is otherwise historical; it records what was true at v0.9.0.

---

## 1. Why this phase exists

Every layer `ContinuumElement` composes already existed, unconnected: `numerics/interpolation`
(shape functions), `numerics/mapping` (geometric mapping), `numerics/quadrature` (integration
rules), `numerics/operators` (the B operator, mass term), and `physics/base` (`Theory`,
`ConstitutiveModel`). The only element that existed, `Bar`, bypasses all of it via the documented
ADR-002 closed-form exception. Nothing in the codebase, before this phase, could compute a local
stiffness matrix by actually integrating shape functions against a constitutive tangent — the
generic `∫ BᵀDB dΩ` pipeline that every textbook FEM course starts with.

`docs/design/OPERATORS.md` and `docs/dev/notes.md` (N-42) named this gap explicitly and on
purpose: `symmetric_gradient_matrix` returns its B operator with the DOF axis deliberately left
unflattened, `(n_qp, n_voigt, n_fun, dim)` rather than the `(rows, n_dof)` shape SDS Section 8's
own notation writes, because flattening requires a DOF-ordering choice that belongs to "the
not-yet-built `elements/` layer" — this phase is that layer arriving.

---

## 2. The DOF-flattening insight: it's a reshape, not an `einsum`

SDS C-2 orders element DOFs node-major-then-component: `(node 1: u_x, u_y; node 2: u_x, u_y; ...)`.
That is exactly what a C-order NumPy `reshape` produces when merging an array's trailing
`(n_fun, dim)` axes into one: element `[a, c]` of the unflattened array lands at flat index
`a * dim + c` — node `a`'s components contiguous, before moving to node `a + 1`. No data movement,
no `einsum`, no loop:

```python
def _flatten_b_matrix(b, n_dof):
    """(n_qp, n_voigt, n_fun, dim) -> (n_qp, n_voigt, n_dof), node-major-then-component."""
    n_qp, n_voigt, n_fun, dim = b.shape
    return b.reshape(n_qp, n_voigt, n_dof)
```

This is the one genuinely new architectural decision this phase makes; everything else is
composition of existing, unmodified pieces.

---

## 3. `ContinuumElement`: what it composes and what it caches

Constructed per cell from: a reference cell type + interpolation order (→
`LagrangeInterpolation` + `shape_functions()`), physical node coordinates (→ `AffineMapping`,
built from the vertex sub-array only — affine geometry, no curved/isoparametric elements this
phase), a quadrature order (default `2 × interpolation_order`, exact for the mass integrand
`N_a N_b`; the stiffness integrand `BᵀDB` is strictly lower-order so the same rule covers it), a
`Theory` + `ConstitutiveModel` + `Material`, a pre-built global DOF array (via the new
`elements/dof_utils.py::build_local_dof_map`), and a plain `section_measure: float` multiplier
(area/thickness/1.0 — the same role `Bar`'s bare `area` float already plays; `IsotropicElasticity`'s
weak form integrates over length/area only, per the existing bar-verification test's own
documented convention, so cross-sectional area is an element-layer concern, never the theory's).

At construction, `ContinuumElement` tabulates once (shape values, physical gradients, quadrature
weights, `mapping.volume_scale` — the unsigned Gram-determinant form, dimensionally consistent
across cell types with no special-casing) and builds the flattened B matrix and the constitutive
tangent `D` immediately. `D` is extracted via the same zero-strain-probe pattern
`test_bar_verification.py` already established for `Bar`'s equivalence proof: for a linear
constitutive law the tangent doesn't depend on the strain value, so probing `respond_batch` at
`strains = 0` is the correct way to pull out `D` for a stiffness matrix, not a special case
invented for this task.

`local_stiffness()` (`K_e = section_measure Σ_q w_q J_q B_qᵀ D_q B_q`) and the shape-function
quadrature sum used by the body-force integral are computed eagerly at construction, since every
required input (`E`, geometry, quadrature) is already mandatory. `local_mass()` looks up `rho`
**lazily, on first call**, and caches the result — a material that never defines `rho` still
builds a perfectly valid element as long as nothing ever asks for `MASS`. `local_mass()`
block-expands the scalar `mass_term(...)` (already existing, `numerics/operators/helmholtz.py`)
via `section_measure * rho * np.kron(mass_scalar, np.eye(dim))`: `kron(A, I_dim)[a·dim+k, b·dim+l]
= A[a,b]` iff `k == l`, which is exactly the node-major-then-component block-diagonal-per-node
structure the flattening convention requires — the same trick, reused, not a second bespoke
ordering scheme.

`contributions(role)` emits `STIFFNESS` unconditionally and `MASS` only if the theory declares
that role (mirroring `Bar`, which also only emits `STIFFNESS` despite its theory declaring more).
`FORCE` is never emitted by `ContinuumElement` itself — a body force is not intrinsic per-element
data; the same element could be covered by zero, one, or several body-force regions, which is an
external, model-level decision.

---

## 4. `ContinuumBodyForceProvider`: why it lives in `elements/`, not `constraints/`

`constraints/loads.py`'s `BodyForce` dataclass existed since v0.1.0 with no provider — unlike
`NodalLoad`, whose `NodalLoadProvider` needed no integration (it just places nodal values), a body
force genuinely needs `∫ N_a(x) b_c dΩ`, which requires the element's own tabulated shape
values/quadrature/mapping data. The import-linter layer contract (`elements > constraints`, higher
may import lower) lets `elements/continuum/continuum.py` import `BodyForce` from
`constraints/loads.py`, but not the reverse — and the provider must hold references to
already-built `ContinuumElement` instances to reuse their cached data, which structurally forces
it to live one layer up. It is co-located in `continuum.py` rather than a new file, since it has
no meaning independent of `ContinuumElement`.

---

## 5. Verification

`tests/unit/test_continuum_element.py`: stiffness checked against `Bar.local_stiffness()`
directly (two independent pieces of *production* code, not the same code called twice); mass
checked against the classical consistent mass matrix `ρAL/6 [[2,1],[1,2]]`; body force checked
against the classical uniform-load consistent vector `bAL/2 [1,1]`; a global-equilibrium check on
`ContinuumBodyForceProvider` across a 2-cell mesh; a lazy-`rho` safety test (a `MASS`-free theory
never requires the material to define `rho`); a multi-field rejection test.

`test_bar_verification.py` is **unchanged**. Its `_composed_stiffness` helper is deliberately
built "from scratch... with no call into `Bar` itself" — refactoring it to call
`ContinuumElement` would make it circular (a bug shared between the two would go undetected) and
destroy its value as an independent ADR-002 equivalence oracle. This phase's stiffness-vs-`Bar`
test is a different, additional statement: it compares two independent pieces of production code,
which the existing test does not do.
