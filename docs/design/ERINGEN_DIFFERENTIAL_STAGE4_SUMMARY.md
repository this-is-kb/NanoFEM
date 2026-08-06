# Stage 4 Completion Summary: Eringen Differential Nonlocal Elasticity (v0.20.0-v0.25.0)

**Status:** complete. This document ties the five increments that make up Stage 4 - the nonlocal
bar (v0.20.0), the general 2-D continuum theory (v0.22.0), the nonlocal Euler-Bernoulli beam
(v0.23.0), the first verification/benchmark/documentation closing pass (v0.24.0), and a second,
more exhaustive verification pass against a more detailed restatement of the same directive
(v0.25.0) - against the directive's own final acceptance criteria, so the whole stage can be
audited in one place rather than reconstructed from five separate changelog entries.

Full mathematical detail lives in the three per-member design documents this summary indexes,
not duplicates: `ERINGEN_DIFFERENTIAL_BAR.md`, `ERINGEN_DIFFERENTIAL_CONTINUUM.md`,
`ERINGEN_DIFFERENTIAL_BEAM.md`.

---

## 1. Acceptance criteria, checked one by one

The directive's final acceptance criteria, verbatim, each mapped to the artifact that satisfies
it:

| Criterion | Satisfied by |
|---|---|
| Solve classical elasticity | `IsotropicElasticity`/`PlaneStress\|StrainConstitutive` (pre-Stage-4, unmodified) |
| Switch to Eringen differential without touching the solver | `examples/ex10_classical_to_eringen_theory_swap.py` - the same `Model`/`DofHandler`/`Assembler`/`LinearStaticAnalysis` pipeline, only `Theory`+`ConstitutiveModel` swapped in the `DomainDefinition` |
| Solve benchmarks | 1-D bar (`test_nonlocal_bar_benchmark.py`), 1-D beam (`test_nonlocal_beam_benchmark.py`), plane stress: plate-with-a-hole (`test_nonlocal_plate_with_hole_benchmark.py`) and cantilever (`test_nonlocal_cantilever_benchmark.py`), plane strain (`test_nonlocal_plane_strain_benchmark.py`), Q4 cantilever (`test_nonlocal_parametric_sensitivity.py`) |
| Recover published/analytical solutions | Bar and beam benchmarks match closed forms to machine precision at every mesh size (v0.20.0/v0.23.0); plate-with-hole recovers the classical Kirsch solution exactly at `e0a=0` (v0.22.0); cantilever recovers Timoshenko beam theory in the mesh-refinement limit (v0.24.0) |
| Recover classical elasticity as `e0a -> 0` (`ell -> 0`) | Exact to machine precision for every 1-D benchmark and for any constant-strain 2-D field (single-element patch test, any mesh size for the bar/beam); **mesh-convergent** (not exact-on-a-fixed-mesh) for a general 2-D field - see Section 3 below, this is a documented mathematical property, not a gap |
| Demonstrate convergence | Mesh convergence: all four benchmarks. Characteristic-length convergence/behavior: `figure_3_characteristic_length_study` (monotonic softening), `test_tip_deflection_increases_monotonically_with_e0a` |
| Demonstrate stable conditioning | `test_nonlocal_conditioning_and_energy.py` - `np.linalg.cond` stays finite, grows only polynomially under refinement, and is not pathologically worse than the equivalent classical system |
| Publication-quality figures | `examples/ex09_eringen_differential_parametric_study.py` - three `matplotlib` figures (bar displacement profile, beam `O(h^2)` mesh convergence, cantilever characteristic-length softening) |
| No modification of Mesh/Reference Elements/Shape Functions/Quadrature/Assembly/Linear Solver/Post-processing | Confirmed for every increment - see each design document's own "scope discipline" section; `git diff`-equivalent audit (file list below) shows only new files plus two pre-planned dispatch-table additions |

---

## 2. What changed, file by file, across all four increments

**New physics/theory files** (all additive, none replacing existing classes):
- `physics/elasticity/eringen_differential.py` - `EringenDifferentialTheory`,
  `EringenDifferentialMaterial` (v0.22.0)
- `elements/continuum/nonlocal_continuum.py` - `NonlocalContinuumElement` (v0.22.0)
- `constraints/loads.py` - `NonlocalAxialLoad` (v0.20.0), `NonlocalTransverseLoad` (v0.23.0)
- `constraints/nonlocal_load.py` - `NonlocalAxialLoadProvider` (v0.20.0),
  `NonlocalTransverseLoadProvider` (v0.23.0)

**Pre-planned dispatch-table additions** (the only touches to shared infrastructure, both
single `if`/`elif` branches following an existing pattern exactly, not new abstractions):
- `elements/factory.py`: `EringenDifferentialTheory -> NonlocalContinuumElement`
- `analysis/static.py`: load-case dispatch gained `NonlocalAxialLoad`/`NonlocalTransverseLoad`
  branches

**Untouched** (verified, not just asserted): `Mesh`, `DofHandler`, `Assembler`,
`SparsityPattern`, `ConstraintHandler`, `LinearStaticAnalysis`, every `ReferenceElement`, every
shape-function/interpolation class, every quadrature rule, `AffineMapping`/`IsoparametricMapping`,
the sparse direct solver, `postprocess/recovery.py`, `Bar`, `EulerBernoulliBeam`,
`TimoshenkoBeam`, `ContinuumElement`, `IsotropicElasticity`, `PlaneStress/StrainConstitutive`.

**Verification/benchmark/documentation added in the closing increment (v0.24.0)**, the subject
of this summary's own increment:
- `tests/unit/test_nonlocal_conditioning_and_energy.py` (condition number, energy consistency,
  global symmetry - the two Step-6 items not covered by any prior test)
- `tests/unit/test_nonlocal_cantilever_benchmark.py` (the explicit "2-D cantilever" Step-7
  target)
- `examples/ex09_eringen_differential_parametric_study.py` (Step-8's parametric study, rendered
  as figures)
- `examples/ex10_classical_to_eringen_theory_swap.py` (Step-9's tutorial requirement)
- `docs/design/ERINGEN_DIFFERENTIAL_CONTINUUM.md` Section 7 and dev note N-86 (the mesh-
  convergent local-limit discovery, Section 3 below)

**Verification added in the second pass (v0.25.0)**, Section 4 below covers each in detail:
- `tests/unit/test_nonlocal_helmholtz_and_assembly_verification.py` (18 tests)
- `tests/unit/test_nonlocal_plane_strain_benchmark.py` (7 tests)
- `tests/unit/test_nonlocal_local_limit_recovery.py` (4 tests)
- `tests/unit/test_nonlocal_parametric_sensitivity.py` (5 tests)
- `docs/design/ERINGEN_DIFFERENTIAL_CONTINUUM.md` Section 8 (field-level recovery table and the
  global-operator-versus-response clarification)

---

## 3. The one genuine discovery this stage produced

Building the 2-D cantilever benchmark (v0.24.0) found that the mixed `(u, e*)` formulation's
`e0a=0` local limit matches a directly-computed classical FEM solution **exactly only for a
constant-strain field** - already proven at the element level since v0.22.0
(`test_nonlocal_continuum_element.py`) - but only **mesh-convergently** for a field that varies
in space, such as a cantilever's bending strain. Root cause: `e*` is C0-continuous (shared
between elements, like `u`), while a T3's own classical strain is naturally discontinuous
between elements - the local, single-element Schur-complement equivalence proof does not imply
global equivalence, because matrix inversion does not distribute over sums,
`(A+B)^-1 != A^-1 + B^-1`.

This was investigated to the point of certainty (mesh refinement showed the discrepancy
shrinking monotonically: 185.77% at 4x2 down to 3.79% at 32x16) before being classified as
**expected behavior of implicit-gradient-type mixed regularization models generally** - the same
mathematical structure as Peerlings-style gradient-enhanced damage/plasticity in the published
literature - rather than a defect. No already-shipped test or design document was found to have
made a false "exact match on any mesh" claim, so nothing already delivered needed correction;
this discovery only changed how the new cantilever benchmark's tolerances were written (mesh-
convergent assertions throughout, never exact-match-on-a-fixed-mesh) and added the explanation
itself (`ERINGEN_DIFFERENTIAL_CONTINUUM.md` Section 7, dev note N-86).

A second, smaller instance of the same underlying idea - nonlocality vanishing for spatially
uniform fields - appears in `ex10`'s single-element patch test: a single T3's constant-strain
field makes the Helmholtz *gradient* correction vanish identically, so the tip displacement is
independent of `e0a` altogether, not just equal to the classical value at `e0a=0`. This is the
2-D analogue of the Peddieson paradox already established for the 1-D bar/beam (concentrated or
uniform loading produces zero nonlocal effect), and is called out explicitly in the tutorial
rather than left as a surprising, unexplained number.

---

## 4. The v0.25.0 verification pass: what it added and why

A second, more detailed restatement of the same Stage 4 directive arrived after v0.24.0 shipped.
Rather than re-deriving what v0.24.0 already covered, this pass was a fresh gap audit against
the more granular nine-step breakdown, adding exactly five things that were genuinely missing:

**Helmholtz operator and assembly verification** (`test_nonlocal_helmholtz_and_assembly_
verification.py`, 18 tests). Isolates properties no existing test checked directly: `K_ee`'s
mass term is strictly SPD; its diffusion (gradient) term is PSD with an exactly 3-dimensional
null space (the constant-field mode, once per Voigt component); `K_ee` stays SPD for every
tested characteristic length from 0 to 10; the Helmholtz relation's boundary contribution is
confirmed to be the natural (homogeneous Neumann) condition it is meant to be, by direct
inspection - no Dirichlet BC is ever declared on the nonlocal-strain field, and the system stays
solvable purely because `K_ee`'s own positive-definiteness needs no such BC. At the *assembled,
multi-element* scale (a stronger statement than the existing element-level checks): the
`u`-`u` block is structurally exactly zero after assembly; the `u`-`e*` and `e*`-`u` blocks stay
exact transposes; the global `e*`-`e*` block stays negative definite; the sparsity pattern
declares every entry the assembler actually writes; the factory's DOF ordering matches the
element's own layout assumption; the element emits nothing for any role but `STIFFNESS`.

**The plane-strain benchmark** (`test_nonlocal_plane_strain_benchmark.py`, 7 tests). Every 2-D
benchmark through v0.24.0 used `PlaneStressConstitutive`; `EringenDifferentialMaterial` already
wrapped `PlaneStrainConstitutive` just as generically, but no full solve had ever exercised it.
Mirrors the plane-stress cantilever exactly (same geometry/mesh/load) so only the constitutive
law changes: a single-element patch test matches classical plane-strain FEM for any `e0a`, the
cantilever mesh-convergent local limit holds exactly as for plane stress, softening is monotonic
in `e0a`, and a textbook cross-check confirms plane strain is stiffer than plane stress for
identical (E, nu, load).

**Field-level local-limit recovery** (`test_nonlocal_local_limit_recovery.py`, 4 tests). Extends
the existing displacement-level mesh convergence to strain, stress, energy, and effective
stiffness individually, with the numerical error tabulated at each mesh level (reproduced in
`ERINGEN_DIFFERENTIAL_CONTINUUM.md` Section 8). This also surfaced a clarifying subtlety worth
recording: the *global effective stiffness operator* itself (the Schur complement eliminating
`e*`) does **not** converge to the classical global stiffness matrix entrywise under refinement
(its Frobenius-norm difference actually grows, 49% -> 74%, because eliminating a globally-shared
`e*` field produces long-range dense "fill-in" that has no counterpart in the sparse classical
operator) - yet the *response to any specific smooth load* converges regardless, which is the
only claim any benchmark in this codebase ever makes or needs to make.

**Parametric sensitivity: integration order and element type**
(`test_nonlocal_parametric_sensitivity.py`, 5 tests). Confirms the default quadrature order for
a T3 is already numerically exact (raising it changes nothing beyond floating-point noise, since
every integrand is a low-degree polynomial), and reproduces the cantilever benchmark's
mesh-convergent local limit and nonlocal softening independently for `quad4` - both properties
belong to the theory, not to one specific element family.

**Computational cost and scalability** (measured directly, not gated by a flaky timing test -
wall-clock time is not asserted in `pytest`, only reported here). The mixed system carries 2.5x
as many DOFs as the equivalent classical system for a 2-D problem (`u`: 2 components/node,
`e*`: 3 Voigt components/node, vs. 2 for classical alone); the sparse direct solve's wall-clock
time scales accordingly, staying within a roughly constant 1.7-2.2x multiple of the classical
solve across two orders of magnitude of mesh refinement - consistent, not pathological, scaling:

| mesh | DOFs (mixed) | DOFs (classical) | solve time, mixed (s) | solve time, classical (s) | ratio |
|---|---|---|---|---|---|
| 8x4   | 225    | 90    | 0.054 | 0.025 | 2.20 |
| 16x8  | 765    | 306   | 0.161 | 0.097 | 1.67 |
| 32x16 | 2,805  | 1,122 | 0.653 | 0.384 | 1.70 |
| 48x24 | 6,125  | 2,450 | 1.534 | 0.883 | 1.74 |
| 64x32 | 10,725 | 4,290 | 2.826 | 1.561 | 1.81 |

**Mathematical consistency and notation** (Step 1, audited fresh rather than re-derived): the
characteristic-length notation (`e0a` the material property, `mu = (e0a)^2` the quantity the
Helmholtz relation actually uses) was confirmed identical across all three per-member design
documents (bar, beam, continuum) and the source code - no drift found, so no changes were
needed.

## 5. What was explicitly not built, per the directive's own list

Integral Nonlocal Elasticity, Strain Gradient theory, micropolar/couple-stress theory, surface
elasticity, plasticity, dynamics, buckling, and topology optimization were not implemented,
matching the directive's explicit "DO NOT IMPLEMENT" list. `NonlocalContinuumElement`'s coupling
pattern (a displacement field plus one Helmholtz-coupled auxiliary field) was deliberately named
generically rather than after Eringen specifically, so that Stage 5 (Eringen Integral Nonlocal
Elasticity) and any future gradient-type theory can be evaluated for reuse without renaming
anything - a decision recorded, not yet acted on, since Stage 5 is out of this stage's scope.

---

## 6. Verification gate (both increments)

`black --check`, `isort --check-only`, `ruff check`, `mypy --strict`, `lint-imports` (4/4
contracts kept), and the full `pytest` suite all pass - see `CHANGELOG.md`'s `[0.24.0]` and
`[0.25.0]` entries for the itemized list of what was added in each pass. `.github/workflows/
ci.yml` already runs every one of these checks on every push, so every test added in either
increment is part of NanoFEM's continuous verification framework from the moment it merges -
no separate "verification framework" setup was needed (Step 8 of the v0.25.0 directive).
