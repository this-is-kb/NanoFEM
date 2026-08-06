# NanoFEM Eringen Differential Nonlocal Elasticity: General (2-D) Continuum Theory (v0.22.0)

**Status:** implemented and tested. Stage 4's second increment - extends v0.20.0's 1-D nonlocal
bar to a genuine, dimension-independent 2-D continuum theory, confirmed with the project owner
before implementation (architecture: reuse the `Theory` ABC as-is, build a new
`NonlocalContinuumElement`; first benchmark: the plate-with-a-hole, reusing Stage 3's Q4
infrastructure).

**Scope discipline.** No change to `Mesh`, shape functions, reference elements, mapping,
quadrature, the assembly pipeline, the linear solver, or `postprocess/recovery.py` - all
explicitly frozen per this phase's directive. `ContinuumElement`/`Bar`/the beam elements are
completely untouched. Everything new lives in two files:
`physics/elasticity/eringen_differential.py` (the theory + constitutive law) and
`elements/continuum/nonlocal_continuum.py` (the one new element family).

---

## 1. Why `Theory` already is `AbstractContinuumTheory`

Audited against the directive's own requirement list before writing anything:

| Requirement | Existing mechanism |
|---|---|
| Field variables | `fields()` |
| Constitutive operators | `operators_used()` |
| Weak formulation / governing equations | `operators_used()` + `contribution_kinds()` + `operator_roles()` |
| Boundary operators | `ContributionKind.FACET` (already exists) |
| Internal variables | `required_state()` (`StateLayout`) |
| Coupled operators | `Locality.LOCAL`/`PAIRWISE` (already exists) |
| Strong formulation | documentation only - nothing in this codebase calls a "strong form" at runtime |

Confirmed with the project owner: no parallel `AbstractContinuumTheory` class was built - it
would duplicate `Theory`, the same reasoning already applied to the Stage-3 "keep the single
`Element` ABC" decision.

---

## 2. The mathematics: why a second field, and why nonlocal *strain*

Eringen's differential nonlocal elasticity replaces the classical pointwise `sigma = C:eps(u)`
with a Helmholtz-regularized relation:

```
e*(x) - mu * laplacian(e*(x)) = eps(u)(x)      mu = (e0*a)^2
sigma*(x) = C : e*(x)
```

**Why nonlocal *strain*, not stress.** Eringen's own papers state the relation on stress
(`sigma - mu*laplacian(sigma) = C:eps(u)`). For spatially-constant `C` (any homogeneous NanoFEM
material), these are exactly equivalent: `(I - mu*laplacian)(C:e*) = C:(I - mu*laplacian)e*`,
since `C` commutes through differentiation. Choosing strain as the auxiliary field reuses the
*exact* Voigt convention, shape functions, and `PlaneStress/StrainConstitutive` matrices
`ContinuumElement` already has - zero new tensor algebra, not an approximation.

**Why a second field, not just a new `ConstitutiveModel`.** The 1-D nonlocal bar (v0.20.0)
eliminated the nonlocal stress in closed form because a 1-D bar's equilibrium is a first-order
ODE with a closed-form integral. General 2-D/3-D equilibrium is a divergence condition on a
tensor field with no such inversion, so `e*` must be a genuine, independently-discretized field
coupled to `u` - a real, additive extension (one new element family), not a redesign.

---

## 3. The weak form, and the sign bug it took two rounds of verification to catch

Testing equilibrium with `delta_u` and the Helmholtz relation with `C:delta_e*` (the
"energy-conjugate" weighting, needed so the two off-diagonal coupling blocks are exact
transposes of each other rather than one carrying an extra `C` factor the other doesn't) gives:

```
K_ue . e* = f                            (equilibrium)
K_eu . u  = K_ee . e*,   K_eu = K_ue^T   (Helmholtz relation)
```

Assembled as one monolithic system, the correct block matrix is:

```
[[ 0,    K_ue ],   [u ]   [f]
 [ K_eu, -K_ee]] . [e*] = [0]
```

the standard KKT/Stokes-type saddle-point sign pattern (zero block on `u`, *negative* block on
`e*`, from moving `K_eu.u` to the same side as `-K_ee.e*` when the equation is rearranged to
`K_eu.u - K_ee.e* = 0`).

**This sign was gotten wrong on the first attempt, and the first round of verification did not
catch it.** Verification proceeded in two stages:

1. *Substitution-based checks* (symmetry of `K_eu`/`K_ue`; the Schur complement
   `K_ue @ K_ee^-1 @ K_eu^T` exactly matching the classical T3 stiffness at `mu=0`; a
   constant-strain field recovering `e* = eps(u)` exactly for any `mu`) - all passed, using
   `+K_ee`. These checks solve for `e*` *given* `u` by direct substitution
   (`e* = K_ee^-1 K_eu u`), which is mathematically insensitive to the sign of `K_ee` in the
   assembled monolithic matrix: substitution never builds that matrix.
2. *A full `Model -> LinearStaticAnalysis` solve* (the same uniaxial-tension plate
   `test_static_t3_plate_analytical.py` already used for T3) reproduced the closed-form
   `P*L/(E*H*t)` tip displacement to the right *magnitude* but the *wrong sign* - with `+K_ee`.
   Re-deriving the residual equations from scratch (`K_eu.u - K_ee.e* = 0`, matched term by
   term against the matrix-row convention `[block(e,u) | block(e,e)][u;e*] = 0`) showed the
   diagonal block must be `-K_ee`. Fixing it and re-running the same full-pipeline solve gave
   the correct sign, exactly.

This is now the strongest evidence yet in this project for why the "verify against a full,
real solve, not just a substitution or an isolated identity" discipline matters: a check that
is mathematically *correct* for what it tests can still be silently blind to a bug the very next
layer up depends on. `test_static_nonlocal_plate.py` keeps this exact scenario as a permanent
regression test (asserting `u_x > 0`, not just `abs(u_x) == expected`).

---

## 4. `NonlocalContinuumElement`: composition, not a redesign

Mirrors `ContinuumElement`'s constructor closely (same interpolation/mapping/quadrature setup),
differing only in building *two* fields' shape data instead of one:

- `B_u`: `symmetric_gradient_matrix`, exactly as `ContinuumElement` already builds it.
- `N_e*`, `grad(e*)`: block-expanded scalar shape values/gradients into per-Voigt-component
  vector shape functions (`_block_expand`, the same `kron(scalar, I)` trick dev note N-51
  already named, generalized from `I_dim` to `I_n_voigt`).
- The `D` matrix: read from the wrapped classical law via the same zero-strain-probe pattern
  `ContinuumElement` uses.
- `e0a`: Eringen's own canonical material property (`materials/material.py`'s `_BOUNDS`,
  present since v0.1.0 - "`0 = local limit, deliberately legal`" - confirmed as the right,
  pre-existing vocabulary to reuse rather than inventing a new property name).

DOF layout is a documented simplification: `global_dofs` concatenates `u`'s node-major block
then `e*`'s node-major block, rather than a single per-node SDS C-2 interleaving of both fields
- `ElementDofSignature`/`dof_signature()` has zero consumers anywhere in the codebase (confirmed
by grep before this class was written), so full interleaving would add real implementation
complexity for no current benefit. Correctness does not depend on the choice: the `Assembler`
only needs `local_stiffness()`'s rows/columns to line up 1:1 with `global_dofs`.

The element is deliberately *not* named after Eringen: the pattern (displacement + one auxiliary
tensor field, Helmholtz-coupled) is generic enough to serve Strain Gradient or similar theories
later without redesigning this class - only a new `Theory`/`ConstitutiveModel` pair would be
needed. What is Eringen-specific is confined to `physics/elasticity/eringen_differential.py`.

---

## 5. Backbone independence, demonstrated not just claimed

`elements/factory.py` gained one more `isinstance` dispatch branch
(`EringenDifferentialTheory -> NonlocalContinuumElement`), following the exact pattern every
prior theory used. `TractionLoadProvider`/`NodalLoadProvider`/`DirichletBC` needed **zero**
changes - they already resolve fields/components generically via `Model.field_specs()`, so a
`TractionLoad` targeting the `"u"` field works identically whether the domain's theory is
`IsotropicElasticity` or `EringenDifferentialTheory`. The plate-with-hole benchmark
(`test_nonlocal_plate_with_hole_benchmark.py`) reuses `test_plate_with_hole_benchmark.py`'s Q4
mesh generator, Dirichlet BCs, and `TractionLoad` byte-for-byte - only the theory/constitutive
law and the (necessarily element-specific) recovery call differ.

---

## 6. Verification

`test_eringen_differential_theory.py` (12 tests): field/continuity/property declarations, Voigt
component naming, dimension rejection, and that `EringenDifferentialMaterial` genuinely
delegates to its wrapped classical law (no duplicated formula).

`test_nonlocal_continuum_element.py` (10 tests): symmetry; the Schur-complement exact match to
classical T3 at `e0a=0`; the constant-strain null-effect for any `e0a` (the 2-D analogue of the
nonlocal bar's Peddieson-paradox finding); stress recovery; DOF signature; measure.

`test_static_nonlocal_plate.py` (3 tests): the full-pipeline sign regression test described
above, plus factory dispatch.

`test_nonlocal_plate_with_hole_benchmark.py` (4 tests): `e0a=0` converges to the classical
Kirsch value under mesh refinement (the end-to-end confirmation of the element-level exact
equivalence); reactions balance the applied traction; peak stress at the hole boundary
*decreases monotonically* as `e0a` increases (2.78 -> 2.22 -> 1.76, at `e0a` = 0, 0.15, 0.35 -
nonlocal elasticity's defining qualitative behavior, stress-concentration regularization);
mesh convergence holds for `e0a > 0` on a genuinely non-constant stress field. No exact published
closed form exists for this problem/model combination (nonlocal Kirsch-type results in the
literature are almost all for the *integral* model), so the benchmark verifies direction and
convergence rather than an exact number - stated honestly rather than overclaimed.

Full gate: black/isort/ruff/mypy strict/import-linter (4 kept, 0 broken)/pytest, all green.

---

## 7. The local limit's exact scope: single-element exact, mesh-wide convergent

**This section exists because the distinction below is easy to get wrong, and doing so would
overclaim the implementation's accuracy.** Every "`e0a=0` reproduces classical elasticity
exactly" statement above and in the tests refers to one of two genuinely different, both true,
claims - conflating them would be a mistake:

1. **Single-element exact** (`test_nonlocal_continuum_element.py`): a *single* T3/Q4 element's
   own local Schur complement `K_ue @ K_ee^-1 @ K_eu^T` matches its classical stiffness to
   floating-point precision, for *any* `e0a` on a constant-strain field, and at `e0a=0` for any
   field a linear element can represent (which, for a T3, is exactly the constant-strain case).
   This is an algebraic identity, proven once and for all, independent of mesh.

2. **Mesh-wide convergent, not exact on a fixed mesh** (discovered while building the 2-D
   cantilever benchmark, `test_nonlocal_cantilever_benchmark.py`): for a *multi-element* mesh
   whose true classical strain field is **not** constant across the domain (bending, a hole,
   any real 2-D problem), the assembled `e0a=0` solution does **not** exactly match a
   directly-computed classical FEM solution on the same mesh - the discrepancy is large on a
   coarse mesh (185% on a 4x2 cantilever mesh) and shrinks monotonically to a few percent under
   refinement (56% -> 15% -> 3.8% at 8x4 -> 16x8 -> 32x16), verified before the cantilever
   benchmark's tolerances were chosen.

**Why, precisely.** `e*` is discretized with the same C0 (nodally-continuous) linear shape
functions as `u`, so it is *structurally incapable* of representing the classical T3's own
strain field exactly whenever that field is not constant everywhere: a T3's classical strain is
piecewise-constant, generally *discontinuous* across element boundaries, while `e*` is forced
continuous by construction. At `e0a=0` the Helmholtz relation degenerates to a pointwise
`e* = eps(u)`, which a continuous shape-function space can only satisfy approximately (an L2-type
projection) unless `eps(u)` is itself continuous - true only for the degenerate constant-strain
case. As elements shrink, neighbouring elements' classical strains converge to each other, so
the "forced continuity" cost vanishes - hence mesh-wide convergence, not mesh-wide exactness.

**This is not a NanoFEM-specific defect - it is the well-known, accepted behavior of this whole
*class* of model.** The Eringen differential formulation built here is mathematically identical
in structure to the "implicit gradient" regularization models widely used in gradient-enhanced
damage/plasticity (Peerlings et al. and the substantial literature following them): a local
field coupled to a continuous auxiliary field through the same Helmholtz relation, with the same
documented property that the discretized local limit is a mesh-convergent statement, not a
per-mesh exact one, for any field the auxiliary discretization cannot represent exactly.

**Practical consequence for every benchmark in this codebase.** Any full-pipeline test claiming
"`e0a=0` recovers the classical solution" on a genuinely non-constant field states this as a
*mesh-convergence* result (checked by refining and watching the error shrink, as
`test_nonlocal_plate_with_hole_benchmark.py` already did correctly before this section was
written, and as `test_nonlocal_cantilever_benchmark.py` does explicitly) - never as an
exact-match assertion on a single, fixed, coarse mesh. The one exception is a field that is
genuinely constant everywhere (`test_static_nonlocal_plate.py`'s uniaxial tension patch test),
where exactness is correct and expected at any mesh density, including one element.

---

## 8. Field-level local-limit recovery, and a clarifying operator-level subtlety

Section 7 quantified the local limit at the level of the *nodal displacement solution*.
`test_nonlocal_local_limit_recovery.py` (Stage 4's closing verification pass) repeats the same
mesh-refinement study directly at the level of **strain**, **stress**, **strain energy**, and
**effective stiffness** (`F / |deflection|`) - the same cantilever benchmark, same three mesh
levels, so the numbers are directly comparable to Section 7's displacement figures:

| mesh | strain error | stress error | energy error | stiffness error |
|---|---|---|---|---|
| 6x3  | 54.0% | 27.1% | 91.6% | 47.8% |
| 10x5 | 42.3% | 19.4% | 36.2% | 26.6% |
| 16x8 | 30.2% | 15.7% | 14.7% | 12.8% |

Every column shrinks monotonically, exactly like the displacement error in Section 7 - the
mesh-convergent local limit is a property of the whole solution, not an artifact isolated to the
nodal displacement.

**A clarifying subtlety, found while designing this table.** It is tempting to expect the
*global effective stiffness operator itself* - the Schur complement `K_eff = K_ue . K_ee^-1 .
K_eu` obtained by eliminating `e*` from the fully assembled, unconstrained system - to converge
to the classical global stiffness matrix `K_classical` under refinement, entrywise, the same way
the solution does. **It does not: the relative Frobenius-norm difference between `K_eff` and
`K_classical` actually grows under refinement** (measured at 49% -> 64% -> 71% -> 74% for
4x2 -> 8x4 -> 16x8 -> 24x12). This is not a contradiction of the displacement/strain/stress/
energy convergence above, and not a bug: `K_ee`'s global assembly couples every element sharing
an `e*` node to its neighbors (Section 7's C0-continuity point again), so its inverse is
generically **dense**, and `K_eff` inherits that density - long-range, small-magnitude "fill-in"
entries between u-DOFs that have no direct element connection in `K_classical` at all. As the
mesh refines, the number of such fill-in entries grows faster than their individual magnitude
shrinks, so the whole-matrix Frobenius norm of the difference grows even as the *response to any
smooth, physically realistic load* - which is what every benchmark in this codebase actually
measures - keeps converging. The lesson: "the operator converges" and "the response to a given
load converges" are different claims for this formulation, and only the second one holds
mesh-wide; every convergence claim in this codebase is, and remains, of the second kind.
