# Stage 4 Completion Summary: Eringen Differential Nonlocal Elasticity (v0.20.0-v0.24.0)

**Status:** complete. This document ties the four increments that make up Stage 4 - the nonlocal
bar (v0.20.0), the general 2-D continuum theory (v0.22.0), the nonlocal Euler-Bernoulli beam
(v0.23.0), and the closing verification/benchmark/documentation increment (v0.24.0) - against
the directive's own final acceptance criteria, so the whole stage can be audited in one place
rather than reconstructed from four separate changelog entries.

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
| Solve benchmarks | 1-D bar (`test_nonlocal_bar_benchmark.py`), 1-D beam (`test_nonlocal_beam_benchmark.py`), plate-with-a-hole (`test_nonlocal_plate_with_hole_benchmark.py`), 2-D cantilever (`test_nonlocal_cantilever_benchmark.py`) |
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

## 4. What was explicitly not built, per the directive's own list

Integral Nonlocal Elasticity, Strain Gradient theory, micropolar/couple-stress theory, surface
elasticity, plasticity, dynamics, buckling, and topology optimization were not implemented,
matching the directive's explicit "DO NOT IMPLEMENT" list. `NonlocalContinuumElement`'s coupling
pattern (a displacement field plus one Helmholtz-coupled auxiliary field) was deliberately named
generically rather than after Eringen specifically, so that Stage 5 (Eringen Integral Nonlocal
Elasticity) and any future gradient-type theory can be evaluated for reuse without renaming
anything - a decision recorded, not yet acted on, since Stage 5 is out of this stage's scope.

---

## 5. Verification gate (this increment)

`black --check`, `isort --check-only`, `ruff check`, `mypy --strict`, `lint-imports` (4/4
contracts kept), and the full `pytest` suite all pass - see `CHANGELOG.md`'s `[0.24.0]` entry
for the itemized list of what was added.
