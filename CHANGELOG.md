# Changelog

All notable changes to NanoFEM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
Semantic Versioning with the documented 0.x policy (minor may break, patch
may not).

## [Unreleased]

## [0.24.0] - 2026-08-06

### Added
- **Eringen Differential Nonlocal Elasticity, verification/benchmark/documentation
  completion** - Stage 4's closing increment. Audited the directive's nine-item verification
  list and Step-7 benchmark suite against what v0.20.0-v0.23.0 already shipped; filled the
  remaining genuine gaps (condition-number analysis, energy consistency at global scale, a 2-D
  cantilever benchmark, publication-quality figures, and a tutorial) without touching any
  already-verified physics or the frozen architecture.
- `tests/unit/test_nonlocal_conditioning_and_energy.py` (5 tests): condition-number analysis
  (`np.linalg.cond`, SVD-based, well-defined for the indefinite saddle-point (u, e*) system) on
  the Dirichlet-reduced global stiffness, confirmed finite and growing only polynomially under
  mesh refinement, and not pathologically worse than the equivalent classical system's own
  conditioning; Clapeyron's-theorem energy consistency (`0.5*u^T*K*u == 0.5*f^T*u`) and global
  symmetry, both checked on the fully assembled coupled system rather than element-by-element.
- `tests/unit/test_nonlocal_cantilever_benchmark.py` (5 tests): the Stage-4 benchmark suite's
  explicit "2-D cantilever" target - a clamped rectangular plate under a tip shear load, T3
  mesh, `NonlocalContinuumElement`/`EringenDifferentialTheory`. Verifies mesh-convergent
  recovery of a directly-computed classical FEM solution at `e0a=0`, convergence of both the
  classical and mixed solutions toward closed-form Timoshenko beam theory, monotonic nonlocal
  softening with `e0a`, reaction equilibrium, and mesh-convergent stability at nonzero `e0a`.
- `examples/ex09_eringen_differential_parametric_study.py` (new): three publication-quality
  figures (`matplotlib`, headless `Agg` backend - the project's first use of this pre-existing
  dependency), each pure post-hoc visualization of already-verified results - the nonlocal bar's
  displacement profile for several `e0a`, the nonlocal beam's `O(h^2)` mesh-convergence curve,
  and the 2-D cantilever's characteristic-length softening study.
- `examples/ex10_classical_to_eringen_theory_swap.py` (new, tutorial): the executable version of
  the Stage 4 directive's final acceptance criterion - a model solved with classical elasticity,
  then re-solved after swapping only the domain's `Theory`/`ConstitutiveModel` to Eringen
  Differential, with the mesh, boundary conditions, load, and solver pipeline completely
  unchanged. A single-element constant-strain patch test matches to machine precision for *any*
  `e0a` (a 2-D analogue of the Peddieson paradox: a spatially uniform strain field has zero
  Helmholtz gradient correction); the 2-D cantilever then demonstrates genuine nonlocal
  softening once the field varies in space.

### Discovered and documented
- **The mixed (u, e*) formulation's local-limit (`e0a=0`) recovery of classical elasticity is
  exact only for a constant-strain field - for a general, spatially varying field it is
  *mesh-convergent*, not exact-on-a-fixed-mesh.** Root cause: the nonlocal strain `e*` is
  C0-continuous (shared between elements, like `u`), while a T3's own classical strain is
  naturally piecewise-constant/discontinuous between elements - the local, single-element
  Schur-complement equivalence proof (already established in `test_nonlocal_continuum_element.py`)
  does not imply global equivalence, since matrix inversion does not distribute over sums.
  Verified via mesh refinement on the 2-D cantilever: relative discrepancy between the mixed
  system at `e0a=0` and a directly-computed classical solution on the same mesh shrank
  monotonically, 185.77% (4x2) -> 55.78% (8x4) -> 14.69% (16x8) -> 3.79% (32x16). This is the
  well-documented, accepted behavior of implicit-gradient-type mixed regularization models
  generally (the same mathematical structure as Peerlings-style gradient-enhanced damage/
  plasticity), not a defect specific to NanoFEM - confirmed no already-shipped test or design
  document had made a false "exact match on any mesh" claim, so no prior work needed correction.
  See `docs/design/ERINGEN_DIFFERENTIAL_CONTINUUM.md` Section 7 and dev note N-86.

### Notes
- `Mesh`, `ReferenceElement`, shape functions, quadrature, the assembly architecture, the linear
  solver, and post-processing remain completely unmodified by this increment, per the Stage 4
  directive's explicit constraint - every addition here is either a test, a documentation
  section, or a pure-visualization example script built on top of the already-frozen v0.20.0-
  v0.23.0 physics.

## [0.23.0] - 2026-08-06

### Added
- **Eringen Differential Nonlocal Elasticity, the nonlocal Euler-Bernoulli beam** - Stage 4's
  third increment, the "nanobeam" target and the richest area of the published nonlocal
  elasticity literature. Mirrors v0.20.0's nonlocal bar one derivative order up: the nonlocal
  effect eliminates to a pure load correction, so `EulerBernoulliBeam`/
  `EulerBernoulliBendingTheory` are completely unmodified.
- `constraints/loads.py`: `NonlocalTransverseLoad` (region, field, nodal `q(x)` samples, the
  nonlocal parameter `mu`), mirroring `NonlocalAxialLoad`'s shape exactly.
- `constraints/nonlocal_load.py`: `NonlocalTransverseLoadProvider`, computing the classical
  consistent load plus `mu*integral(dN_a/dx * q'(x)) dx` for a Hermite `EulerBernoulliBeam`.
- `analysis/static.py`: load-case dispatch gained a fourth branch (`NonlocalTransverseLoad`).
- New test files: `test_nonlocal_transverse_load_provider.py` (6), `test_nonlocal_beam_
  benchmark.py` (7, a simply-supported beam under sinusoidal load, mesh-convergent, exact at
  `mu=0`, and showing the Peddieson-paradox null effect for a uniform load).
- `docs/design/ERINGEN_DIFFERENTIAL_BEAM.md`.

### Fixed
- A genuinely new subtlety, not present in the bar case: a Hermite beam's shape function
  *values* (not only derivatives) need the same reference-to-physical rescaling
  (`_reference_derivative_scale`, N-53) already applied to curvature. Caught by a numerical
  check against the real Hermite/mapping stack that first gave a non-machine-precision `mu=0`
  residual (`~6.7e-5`) where near-exact agreement was expected, given cubic Hermite beam
  elements' own classical nodal-superconvergence property. See
  `docs/design/ERINGEN_DIFFERENTIAL_BEAM.md` Section 2 and `docs/dev/notes.md`.
- Beam equilibrium's sign (`M'' = q`, not the more commonly quoted `M'' = -q`) was derived from
  this codebase's own already-verified weak form (`integral(M*delta_w'') = integral(q*delta_w)`)
  rather than assumed from an independently-recalled shear/moment convention - the first attempt
  using an assumed sign gave a wrong intermediate result, corrected before any production code
  was written.

### Notes
- `NonlocalTransverseLoad.nonlocal_parameter` stores `mu` directly (matching
  `NonlocalAxialLoad`'s existing convention for structural-member load corrections) - a
  different, independently-precedented design point from the 2-D continuum work's
  material-property `e0a` (v0.22.0); the two are not required to match.

## [0.22.0] - 2026-08-06

### Added
- **Eringen Differential Nonlocal Elasticity, general (2-D) continuum theory** - Stage 4's
  second increment, extending v0.20.0's 1-D nonlocal bar to a genuine, dimension-independent
  continuum theory. Confirmed with the project owner before implementation: the existing
  `Theory` ABC already satisfies every requirement of an "AbstractContinuumTheory" (no parallel
  class built); the first benchmark is the plate-with-a-hole, reusing Stage 3's Q4
  infrastructure.
- `physics/elasticity/eringen_differential.py` (new): `EringenDifferentialTheory` (two fields -
  `u`, the displacement, and `e_star`, the nonlocal strain, `VOIGT_LENGTH[dim]` components,
  named by Voigt pair via `field_component_names()`) and `EringenDifferentialMaterial` (wraps a
  classical `PlaneStress/StrainConstitutive` law unchanged - `sigma* = C:e_star` uses the exact
  same `C`, only the strain it acts on differs - no duplicated formula). Uses Eringen's own
  canonical `e0a` material property (`materials/material.py`, present since v0.1.0 - "0 = local
  limit, deliberately legal" - the correct, pre-existing vocabulary, not a new property name).
- `elements/continuum/nonlocal_continuum.py` (new): `NonlocalContinuumElement`, a mixed element
  coupling `u` and `e*` through the symmetric weak form `[[0, K_ue], [K_eu, -K_ee]]` (the
  standard KKT/Stokes-type saddle-point sign pattern). Deliberately not named after Eringen -
  the pattern (displacement + one Helmholtz-coupled auxiliary tensor field) is generic enough
  for Strain Gradient or similar theories later.
- `elements/factory.py`: one more dispatch branch (`EringenDifferentialTheory ->
  NonlocalContinuumElement`), following the existing pattern exactly.
- New test files: `test_eringen_differential_theory.py` (12), `test_nonlocal_continuum_element.py`
  (10), `test_static_nonlocal_plate.py` (3), `test_nonlocal_plate_with_hole_benchmark.py` (4).
- `docs/design/ERINGEN_DIFFERENTIAL_CONTINUUM.md`.

### Fixed
- A genuine sign bug in the mixed formulation's assembled block matrix, caught only by a full
  `Model -> LinearStaticAnalysis` solve, not by symmetry/Schur-complement/constant-strain checks
  (which use a *substitution* `e* = K_ee^-1 K_eu u` insensitive to this exact sign). The correct
  diagonal block is `-K_ee`, not `+K_ee`; the wrong sign gave a displacement with the right
  magnitude but the wrong sign end to end. See `docs/design/ERINGEN_DIFFERENTIAL_CONTINUUM.md`
  Section 3 and `docs/dev/notes.md` for the full account.

### Notes
- `Mesh`, shape functions, reference elements, mapping, quadrature, the assembly pipeline, the
  linear solver, and `postprocess/recovery.py` are all completely unmodified, per this stage's
  explicit "frozen backbone" directive. `TractionLoadProvider`/`NodalLoadProvider`/`DirichletBC`
  needed zero changes to work with the new theory - direct evidence the backbone is genuinely
  theory-independent, not just claimed to be.
- The plate-with-hole benchmark shows the expected qualitative nonlocal behavior (stress
  concentration regularization: peak ratio 2.78 -> 2.22 -> 1.76 as `e0a` goes 0 -> 0.15 -> 0.35)
  and exact recovery of the classical Kirsch solution at `e0a=0`. No exact published closed form
  exists for this problem/model combination, so the benchmark verifies direction and mesh
  convergence rather than an unavailable exact number.

## [0.21.0] - 2026-08-05

### Added
- **Closes the Nonlocal-Ready Classical FEM Backbone's last audit gap: full-pipeline stress
  recovery for `Bar` and `EulerBernoulliBeam`.** An explicit audit (prompted by the project
  owner's request to verify the complete Mesh -> ... -> Stress Recovery pipeline for all four
  minimal elements before declaring Stage 3 done) found that `Bar` and `EulerBernoulliBeam` had
  *no* recovery method at all, and `EulerBernoulliBeam` had never been solved through the real
  `Model -> LinearStaticAnalysis` pipeline - every existing beam test built the element by hand,
  including the v0.14.0 factory-dispatch test, which only compared stiffness matrices. T3/Q4 had
  no such gap (both already had full-pipeline solves and stress recovery since v0.14.0-v0.15.0).
- `Bar.axial_response(local_displacement) -> AxialResponse` (strain/stress/force, exact and
  constant along the member - E-5's closed form).
- `EulerBernoulliBeam.curvature_response(local_displacement) -> BendingResponse`
  (curvature/moment at both end nodes - linear within one cubic-Hermite element, so both nodes
  capture the whole element's field exactly). Verified against the classical cantilever result
  (`M(fixed end) = P*L`) two independent ways (the analytic closed-form Hermite curvature
  formula, and a from-scratch finite-difference curvature of the raw polynomial) before being
  written - this element already caught two sign/scaling bugs in earlier phases (N-53, N-54).
- `tests/unit/test_static_beam_eb_cantilever.py` (new): the first full `Mesh -> Model ->
  LinearStaticAnalysis` solve of a Euler-Bernoulli beam - tip deflection/rotation, reactions, and
  recovered bending moment all checked against classical closed forms.
- `test_static_analytical.py`/`test_bar_analytical.py`/`test_beam_eb_analytical.py`: new tests
  for the two recovery methods, both in isolation and through the full pipeline via
  `build_elements`.

### Notes
- No architecture change - both methods are closed-form, element-local, and follow ADR-002's
  existing pattern exactly (`local_stiffness()`'s sibling, not routed through
  `postprocess/recovery.py`'s tensor-based API, which is correctly scoped to continuum elements'
  full stress tensors - a bar's/beam's generalized stress is a scalar force/moment, a genuinely
  different, simpler concept).
- With this closed, all four Stage-3 minimal elements (Bar, Euler Beam, T3, Q4) now have a
  verified, full `Mesh -> Finite Element -> Shape Functions -> Mapping -> Quadrature -> Local
  Stiffness -> Global Assembly -> Boundary Conditions -> Linear Solver -> Displacements -> Stress
  Recovery` path (Bar/Beam's "shape functions/mapping/quadrature" stages are the ADR-002
  closed-form exception, proven equivalent to the composed path independently, per prior phases).

## [0.20.0] - 2026-08-05

### Added
- **Stage 4 begins: Eringen Differential Nonlocal Elasticity, first target - the fixed-fixed
  nonlocal bar.** `constraints/loads.py`: `NonlocalAxialLoad` (region, field, per-node `q(x)`
  samples, the nonlocal parameter `mu = (e0*a)^2`). `constraints/nonlocal_load.py` (new):
  `NonlocalAxialLoadProvider`, a CELL FORCE `ContributionProvider` computing the classical
  consistent load `integral(N_a*q_h) dx` plus the nonlocal correction
  `mu*integral(dN_a/dx*q_h') dx`, both from the same shape-function/quadrature data.
  `analysis/static.py`'s load-case dispatch gained a third `isinstance` branch alongside
  `NodalLoad`/`TractionLoad`.
- `tests/unit/test_nonlocal_axial_load_provider.py` (6 tests) and
  `tests/unit/test_nonlocal_bar_benchmark.py` (8 tests, parametrized mesh convergence) -
  the latter solving the full `Mesh -> Model -> LinearStaticAnalysis` pipeline for a bar under
  a sinusoidal distributed load, verified against a closed-form solution derived two
  independent symbolic ways and cross-checked numerically before any production code existed.
- `docs/design/ERINGEN_DIFFERENTIAL_BAR.md`: the full strong-form/weak-form derivation, the
  "Peddieson paradox" (uniform loads and point loads show *zero* nonlocal effect in Eringen's
  differential model - a genuine, documented property, not a bug), and why this benchmark
  needed no `Theory`/`ConstitutiveModel` change or the `helmholtz_matrix` operator (the
  nonlocal effect eliminates to a pure load correction for a statically-determinate 1-D bar;
  a general 2-D/3-D nonlocal continuum element will need the mixed/Helmholtz treatment this
  increment's original plan anticipated - separate future work).

### Notes
- `Bar`, `IsotropicElasticity(dim=1)`, `IsotropicElasticConstitutive` are completely
  unmodified - the nonlocal bar reuses the classical bar's stiffness exactly as-is.
- The original plan for this increment (confirmed with the project owner before starting)
  assumed a `NonlocalBarTheory`/`EringenDifferentialConstitutive` pairing built around
  `helmholtz_matrix`. The strong-form derivation showed that assumption doesn't hold for this
  specific, statically-determinate benchmark - documented in full in the design doc as a
  finding, not a shortcut.

## [0.19.0] - 2026-08-04

### Added
- `tests/unit/test_plate_with_hole_benchmark.py`: the classical Kirsch (1898) plate-with-a-hole
  benchmark - a quarter-symmetry, radially-graded Q4 mesh solved through the full `Model ->
  LinearStaticAnalysis` pipeline with a real `TractionLoad` for the remote tension, verified
  against two independent closed-form checkpoints (`sigma_xx = 3S` at the top of the hole,
  `sigma_yy = -S` at the side) and a mesh-convergence sweep showing the recovered peak stress
  error strictly decreasing under refinement. This single benchmark satisfies Stage 3's Step 8
  "infinite plate with a hole" and "mesh convergence" requirements together, by design - a
  constant-strain field (already verified since v0.12.0/v0.13.0) has nothing to converge *to*, so
  a genuinely non-constant field was needed to demonstrate convergence at all.
- `docs/design/PLATE_WITH_HOLE_BENCHMARK.md`.

### Notes
- Two numerically-caught details recorded in full in the design doc: a finite-width plate's true
  stress concentration factor is genuinely higher than the infinite-plate value of 3 (confirmed as
  real physics, not a bug, before widening the plate to `W/a = 10` to keep that correction small);
  uniform radial mesh spacing converges far slower than grading toward the hole for this kind of
  localized-gradient problem (the standard, textbook-documented fix, confirmed by direct
  comparison before choosing it).
- This completes the Stage-3 "Classical FEM Backbone" success criteria as stated: mesh -> elements
  -> shape functions -> Jacobians -> B matrices -> local/global stiffness -> boundary conditions
  -> linear solve -> stress/strain recovery -> VTK export, with every mandatory Step 8 benchmark
  passing.

## [0.18.0] - 2026-08-04

### Added
- `postprocess/export.py`: `VTKExporter` is now real - writes a `Mesh` plus point/cell field data
  to a `.vtu` file via `meshio`. `io/meshio_adapter.py`: `MeshIOAdapter.build_meshio_mesh(points,
  cells)` builds a `meshio.Mesh` from plain geometry primitives (the write path SDS 2.19/rule R3
  already declared; `.read()` - importing an external mesh file - remains a deferred stub, a
  separate, larger undertaking).
- `tests/unit/test_postprocess_export.py` (6 tests) and `docs/design/VTK_EXPORT.md`.
- `pyproject.toml`: a `[[tool.mypy.overrides]]` entry for `meshio.*` (ships no `py.typed` marker,
  same treatment as the existing `scipy.*` override).

### Fixed
- A real layer-contract violation, caught by `lint-imports` before shipping: the first draft put
  the nanofem-`Mesh` -> `meshio.Mesh` conversion inside `MeshIOAdapter` itself, but `io` sits
  *below* `mesh` in the layer contract and cannot import it. Moved the `Mesh`-aware extraction up
  into `VTKExporter` (which legally imports both `mesh` and `io`); `MeshIOAdapter` is now
  genuinely mesh-agnostic, working in plain point/cell-block primitives only.

### Notes
- `TimeSeriesWriter` (XDMF transient export) stays a stub - transient analysis is explicitly out
  of Stage 3's scope.

## [0.17.0] - 2026-08-04

### Added
- `numerics/linalg/linear.py`: `ConjugateGradientSolver` is now real - preconditioned CG (Jacobi/
  diagonal preconditioning) wrapping `scipy.sparse.linalg.cg`, the same "wrap scipy's primitive,
  add NanoFEM's own diagnostics" pattern `SparseDirectSolver` already uses for `spsolve`.
  `iterations`/`residual_history` (the true residual `||b - A x_k||`, recorded via scipy's own
  callback) are exposed as instance attributes after `solve()` returns - convergence monitoring
  and solver statistics without changing `LinearSolver`'s one-method ABC. A run that fails to
  converge within `max_iterations` raises `SingularMatrixError` rather than returning the last
  iterate; a non-positive-diagonal operator (Jacobi preconditioning's own precondition) is
  rejected before iterating.
- `tests/unit/test_linalg_cg_analytical.py` (6 tests) and `docs/design/CONJUGATE_GRADIENT.md`.

### Notes
- Verified against both a synthetic random SPD system with a planted exact solution and the real
  T3 plate stiffness matrix this session's other increments already build - CG matches
  `SparseDirectSolver`'s LU answer on both.
- Not wired into `LinearStaticAnalysis` (still hardcodes `SparseDirectSolver`) - solver selection
  is a separate feature Step 6's own success criterion does not require; `ConjugateGradientSolver`
  is fully usable standalone today.

## [0.16.0] - 2026-08-04

### Added
- `mesh/facet_region.py`: `FacetRegion`, a named set of `(cell_id, local_facet_index)` boundary
  facet identifiers - the minimal facet-identity record a rigorous traction integral needs.
  `Mesh.__init__` gained an optional `facet_regions` parameter (default `()`, every existing
  `Mesh(...)` call unaffected) plus `facet_region`/`facet_region_names`/`facets_in_region`/
  `facet_node_ids`, mirroring the existing node/cell region query style. Closes a gap
  `mesh/region.py`'s own docstring had named since v0.1.0 ("facet/edge regions... refused until
  then rather than half-supported") - confirmed with the project owner before building, since it
  is a genuine architecture extension, not a bug fix.
- `constraints/traction.py`: `TractionLoadProvider`, a real `ContributionProvider` for
  `TractionLoad` (declared since v0.1.0, previously unconsumed) - `integral N_a(s) t_c dS` over a
  facet, mirroring `ContinuumBodyForceProvider`'s cell-level body-force integral one dimension
  down. For the T3/Q4 minimal element library, a facet is always a 2-node line, and a P1/Q1
  element's shape functions restricted to a facet are exactly that line's own linear Lagrange
  basis, so no restriction operator was needed - just a fresh 1-D interpolation + the existing
  embedded `AffineMapping` (v0.5.0's "bar in a plane" case) + the existing line quadrature.
  `analysis/static.py`'s load-case dispatch now accepts `TractionLoad` entries alongside
  `NodalLoad`.
- `tests/unit/test_traction_load_provider.py` (5 tests) and `docs/design/TRACTION_LOADS.md`.

### Fixed
- The same quadrature/interpolation/mapping independence leak v0.14.0 fixed for
  `elements/factory.py` (dev note N-66) recurred in `constraints/traction.py`'s first draft -
  fixed the same way, deferring the three heavy imports into the functions that need them.

### Notes
- `NeumannBC`/`RobinBC` (the separate, `Model`-level flux/BC declarations) remain
  declared-but-unconsumed - this phase's scope was the load-case-level surface load
  (`TractionLoad`, matching `NodalLoad`'s existing pattern), not the whole BC vocabulary.

## [0.15.0] - 2026-08-04

### Added
- `ContinuumElement.quadrature_point_response(local_displacement)` (`elements/continuum/
  continuum.py`): recovers kinematic/kinetic Voigt strain and stress at every quadrature point
  from a solved local displacement vector, reusing the same cached `B` matrix and
  `Constitutive`/`Material` already held for `local_stiffness()`. `ContinuumElement.measure()`:
  the element's own length/area/volume (`section_measure * sum_q w_q J_q`).
- `postprocess/recovery.py` (real, replacing the phase-2 stub): `RecoveryInput`,
  `recover_element_fields` (quadrature-weighted element-average strain/stress, principal values,
  von Mises - the classical *direct* recovery method, not SPR/Zienkiewicz-Zhu, which remain
  future adaptivity work), `recover_nodal_fields` (measure-weighted average across elements
  sharing a node - explicitly per-call, not automatic across a whole model, so a caller respects
  SDS 2.19's "never crosses material interfaces" rule), `strain_energy` (an independent
  `sum_e 0.5 (stress:strain) measure_e` cross-check for `0.5 u^T K u`). Scoped to plane
  stress/strain continuum elements (T3/Q4) only - structural elements' internal
  force/moment/shear is already a direct constitutive-law output, a different concern.
- `tests/unit/test_postprocess_recovery.py` (6 tests) and `docs/design/POSTPROCESSING.md`.

### Notes
- Von Mises/principal values need the *full* 3x3 stress state; a plane reduction's own
  out-of-plane term (`sigma_zz` for plane strain, `eps_zz` for plane stress) is recovered from
  the in-plane state and Poisson's ratio, not silently zeroed - `_out_of_plane_terms` names this
  explicitly, and `RecoveryInput.__post_init__` fails fast for any constitutive law other than
  the two plane laws this reduction is defined for.
- Reaction forces needed no new code - `StaticResult.reactions` has existed since the v0.8.0
  walking skeleton.

## [0.14.0] - 2026-08-04

### Added
- `elements/factory.py`'s `build_elements` dispatch extended from `Bar`-only to all four Stage-3
  element families: `IsotropicElasticity(dim=1)` -> `Bar`, `IsotropicElasticity(dim=2)` ->
  `ContinuumElement` (T3/Q4, dispatched further by mesh cell type), `EulerBernoulliBendingTheory`
  -> `EulerBernoulliBeam`, `TimoshenkoBeamTheory` -> `TimoshenkoBeam`. Every family had existed
  and been verified in isolation since v0.9.0-v0.13.0, but only `Bar` had ever been driven
  through `Model`/`LinearStaticAnalysis` - every other family's tests built elements by hand.
- `core/model.py`: `Model.add_constitutive`/`.constitutives` (mirrors `add_theory`/`.theories`);
  `DomainDefinition.constitutive: str | None` (new optional field); `Model._sections`'s value
  type widened to `DomainGeometry = CrossSection | PlaneGeometry`, since a plane-continuum
  domain's geometry is a `PlaneGeometry` (thickness), not a structural `CrossSection`.
  `Model.validate()` cross-checks a registered constitutive law's `required_properties()` against
  the domain's material, mirroring the existing theory check.
- `physics/base.py`: `Theory.field_component_names()`, a new non-abstract method (default `{}`)
  letting a theory declare explicit per-field component names when the positional default
  (`x`,`y`,`z`,...) is wrong. `EulerBernoulliBendingTheory`/`TimoshenkoBeamTheory` override it to
  `{"u": ("y",), "r": ("z",)}`, matching their own already-pinned `u.y`/`r.z` DOF signature
  convention; `Model.field_specs()` consults it.
- `tests/unit/test_elements_factory.py` (new, 7 tests): each family's dispatch verified against a
  directly-constructed element; error paths for a missing constitutive/geometry and a wrong cell
  type.
- `tests/unit/test_static_t3_plate_analytical.py` (new, 3 tests): the first full `Mesh -> Model ->
  LinearStaticAnalysis` solve of a 2-D continuum domain - a two-triangle plate under a
  consistent-nodal-load uniaxial tension reproduces the classical `P*L/(E*H*t)` bar formula
  exactly, a genuine global patch test through the whole assembly/BC/solve pipeline.
- `docs/design/ELEMENT_FACTORY.md`.

### Fixed
- A real plumbing bug, caught by wiring the beam theories through `Model` for the first time:
  `Model.field_specs()`'s positional component naming gave a bending theory's 1-component `u`
  field the name `u.x`, not the `u.y` its own `dof_signature()` (and two pre-existing, pinned
  regression tests) already required - fixed via `field_component_names()` above, not by
  touching the frozen beam element classes.
- A quadrature-independence leak in five unrelated `numerics` leaf-package tests
  (`test_module_needs_no_quadrature` and four siblings for `operators`/`quadrature`/
  `shape_functions`/`tensors`), caused by `elements/factory.py` importing `ContinuumElement` at
  module scope - `nanofem/__init__.py`'s eager top-level re-exports meant *any* `import
  nanofem.anything` now transitively pulled in `numerics.quadrature`. Fixed by deferring
  `ContinuumElement`/`PlaneGeometry`/`cell_type_of_name`'s imports to inside
  `_build_continuum_elements` itself; `Bar`/`EulerBernoulliBeam`/`TimoshenkoBeam` need no such
  deferral since none of the three imports quadrature.

### Notes
- No new element class, no new top-level package, no change to `Element`'s ABC - confirmed with
  the project owner that the existing single, field-agnostic `Element` ABC already satisfies
  Stage 3's "finite element abstraction" requirement generically.

## [0.13.0] - 2026-07-30

### Added
- **Q4 (4-node quadrilateral)**: `ContinuumElement` (`elements/continuum/continuum.py`) no
  longer requires affine geometry - its mapping construction now tries `AffineMapping` first
  and falls back to `IsoparametricMapping` (built v0.5.0, never previously used by any element)
  whenever `AffineMapping` raises `NonAffineError` for a genuinely non-parallelogram
  quadrilateral. The fallback reuses the same tabulated shape-function basis the field already
  built - no separate geometry interpolation object. `Bar`/T3/parallelogram-Q4 are unaffected
  (they never hit the fallback branch); confirmed via their existing test suites, unmodified,
  still green.
- `tests/unit/test_q4_quadrilateral_verification.py` (new): the same constant-strain patch test
  and rigid-body translation null-space check T3 got in v0.12.0, parametrized over both a
  parallelogram (the `AffineMapping` path) and a genuinely non-affine quadrilateral (the new
  `IsoparametricMapping` path), for both plane stress and plane strain; a pinned
  `NonAffineError` premise check; the DOF signature (4 nodes, `("u.x","u.y")` each).
- `docs/design/Q4_QUADRILATERAL.md`, including the pre-implementation numerical verification
  (patch test + rigid-body checks on a real non-parallelogram quadrilateral, via the real
  `IsoparametricMapping` and `symmetric_gradient_matrix`, before the fallback was written).

### Notes
- No new element class and no new physics - Q4 is delivered exactly as T3 was, as a verified
  use case of the existing `ContinuumElement`.
- The constant-strain patch test passing for a strongly distorted quadrilateral is the
  isoparametric consistency property (partition of unity + linear completeness of the bilinear
  basis), not a coincidence specific to any one shape.
- This closes the classical-FEM-pipeline element library's last named gap
  (Bar/T3/Q4/Euler-Bernoulli beam); Timoshenko beam shipped alongside it in v0.11.0.

## [0.12.0] - 2026-07-30

### Added
- `IsotropicElasticity` (`physics/elasticity/isotropic.py`) extended to accept `dim=2` (plane
  kinematics: a 2-component displacement field, same `symmetric_gradient`/`voigt_map` strain
  measure) alongside the existing `dim=1` axial case - its own docstring already named this gap.
  `required_properties()` returns `("E","nu")` for `dim=2` vs `("E",)` for `dim=1`.
  `IsotropicElasticConstitutive` (the dim=1 axial law) is unchanged - still dim=1-only.
- `PlaneStressConstitutive`/`PlaneStrainConstitutive` (`physics/elasticity/plane.py`, new): the
  two dim=2 constitutive laws (`D = E/(1-nu^2)[[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]]` and
  `D = E/((1+nu)(1-2nu))[[1-nu,nu,0],[nu,1-nu,0],[0,0,(1-2nu)/2]]`), deliberately separate
  classes (matching dev note N-55's precedent) rather than one parametrized law. Pair with
  `IsotropicElasticity(dim=2)`'s kinematics and `geometry/plane.py`'s pre-existing
  `PlaneGeometry(thickness)` via `ContinuumElement`'s existing `section_measure` parameter -
  `symmetric_gradient_matrix` needed zero changes (`VOIGT_ORDER[2]` already existed).
- **T3 (3-node triangle)**: delivered as a verified use case of the existing `ContinuumElement`
  (v0.9.0) - `ContinuumElement(cell_type=TRIANGLE, interpolation_order=1, ...)` - not a new
  element class, since a triangle with linear shape functions is always geometrically affine
  and duplicating `ContinuumElement`'s existing composition would be exactly the kind of
  unnecessary abstraction this phase's directive itself warns against.
- New test files: `test_elasticity_plane_analytical.py`, `test_t3_triangle_verification.py`
  (the constant-strain patch test - both plane stress and plane strain, on two different
  triangles, verified via strain energy since stress/strain recovery doesn't exist yet).
- `docs/design/PLANE_ELASTICITY.md`, including the `NonAffineError`/`IsoparametricMapping`
  reasoning for why Q4 is explicitly deferred to its own later increment.

### Changed
- `test_elasticity_isotropic_analytical.py`: `dim=2` no longer asserted to raise (it's real
  now); only `dim=3` still raises `PhysicsError`, mirroring prior test-loop-exclusion precedent
  (`CircularSection`/`LinearStaticAnalysis` from the walking skeleton).

### Notes
- Q4 needs `ContinuumElement` to support `IsoparametricMapping` (a general, non-parallelogram
  quadrilateral's bilinear map is not affine - confirmed via `AffineMapping`'s own fit-residual
  check, which correctly raises `NonAffineError` rather than giving a silently wrong answer).
  That is a real architecture extension, not attempted this phase.
- Rigid-body null-space checks for T3 use a tolerance scaled to the stiffness matrix's own
  magnitude (`atol = 1e-9 * K.max()`), not the small fixed `atol` the beam elements' tests used
  - floating-point noise on a mathematically-exact-zero product scales with `K`'s magnitude,
  and `E ~ 2e11` Pa makes that noise `~1e-5`, larger than a beam-scale fixed tolerance would
  allow.

## [0.11.0] - 2026-07-30

### Added
- `TimoshenkoBeam` (`elements/structural/beam_timoshenko.py`, new): a closed-form bending+shear
  element (ADR-002) implementing the selective-reduced-integration (SRI) formulation SDS
  clause E-3 names ("Timoshenko declares selective-reduced integration of the shear term") -
  full 2-point Gauss integration of the bending term, reduced 1-point Gauss integration of the
  shear term. 1-D-in-1-D embedding only, mirroring `Bar`/`EulerBernoulliBeam`. DOF signature
  `("u.y","r.z")` per node, same convention as `EulerBernoulliBeam`.
- `TimoshenkoBeamTheory`/`TimoshenkoBeamConstitutive` (`physics/elasticity/timoshenko.py`,
  new): the first genuinely 2-component constitutive law in the codebase -
  `[M_per_I, V_per_As] = diag(E, G) @ [kappa, gamma]`, SDS Section 5's "composed generalized
  strain" pattern applied for the first time. Uses the plain `gradient` operator (not
  `second_gradient`), giving `Continuity.C0` - the precise kinematic consequence of treating
  `w` and `theta` as independent fields (curvature `kappa = d(theta)/dx` is a first derivative
  of the independent rotation, not a second derivative of `w` as in Euler-Bernoulli).
- New test files: `test_timoshenko_beam_analytical.py`, `test_elasticity_timoshenko_
  analytical.py`, `test_timoshenko_beam_verification.py` (the composed-path equivalence proof
  plus a mesh-convergence regression test, 1/2/4/8/16/32-element cantilevers converging
  monotonically toward the classical `PL^3/(3EI)+PL/(GA_s)` solution),
  `test_timoshenko_beam_cantilever_benchmark.py` (an 8-element benchmark at a loose,
  explicitly-justified tolerance).
- `docs/design/TIMOSHENKO_BEAM.md`, including a full record of a wrong initial assumption
  (the widely-known "exact" `Phi`-parametrized Timoshenko formula) caught by numerical
  verification before any production code was written - see `docs/dev/notes.md` N-56.

### Notes
- `Bar`, `EulerBernoulliBeam`, `ContinuumElement`, all of `numerics/`, and the existing
  `isotropic`/`euler_bernoulli` physics files are unmodified - purely additive.
  `TimoshenkoBeam` is not wired into `elements/factory.py`'s dispatch this phase, matching the
  `ContinuumElement`/`EulerBernoulliBeam` precedent.
- A single `TimoshenkoBeam` element is *not* exact for a cantilever (unlike `Bar`/
  `EulerBernoulliBeam`) - this is a genuine, verified property of the selective-reduced-
  integration formulation (convergence under mesh refinement, not single-element exactness,
  is its correctness guarantee), not a defect.

## [0.10.1] - 2026-07-30

### Fixed
- Mandatory Phase 0 repository review (full audit report:
  `docs/dev/PHASE0_REPOSITORY_REVIEW.md`). Removed a dead, name-colliding
  `StaticResult` stub in `analysis/results.py` that shadowed the real one in
  `analysis/static.py`. Consolidated a byte-for-byte-duplicated enum-resolution
  helper (`_resolve_cell_type` x2, `_resolve_family` x1) into a new shared
  `utils.validation.resolve_enum_member()`. Derived `materials/properties.py`'s
  `CANONICAL_KEYS` from `materials/material.py`'s `_BOUNDS` instead of
  hardcoding the same 13 keys twice. Fixed two precision-losing `Any` type
  hints (`ShapeFunctions.reference_element`, `GeometricMapping._cached`).
  Corrected stale "future module"/TODO documentation in `core/__init__.py`,
  `mesh/__init__.py`, `analysis/__init__.py`, and `utils/serialize.py` that
  described already-implemented work as pending or claimed adoption that
  never happened.
- Closed 6 missing-test-coverage gaps found by the audit, purely additively
  (no production code changed to add them): `core/registry.py::Registry`
  (new `test_core_registry.py`), `materials/grading.py`'s grading laws,
  `materials/properties.py::SpatialProperty`, `geometry/standard.py`'s
  `HollowCircularSection`/`HollowRectangularSection`/`ISection`,
  `core/fields.py`'s remaining field factories, and `utils/{config,logging,
  scaling,serialize}.py` (new `test_utils_misc.py`). 1209 tests pass (was
  1193).
- Repository hygiene: added the one `.gitignore` entry it was missing
  (`.import_linter_cache/`) and removed ~58MB of regenerable tool caches from
  the working tree. This project has no `.git` directory - nothing is
  actually under version control to "remove."

## [0.10.0] - 2026-07-30

### Added
- `EulerBernoulliBeam` (`elements/structural/beam_eb.py`, new): a closed-form bending element
  (ADR-002), mirroring `Bar`'s pattern - `K = (EI/L^3)[[12,6L,-12,6L],[6L,4L^2,-6L,2L^2],
  [-12,-6L,12,-6L],[6L,2L^2,-6L,4L^2]]`, 1-D-in-1-D embedding only (pure bending, no
  axial-flexural coupling). DOF signature `("u.y","r.z")` per node - `NanoFEM_SDS.md`'s own
  worked C-2 example, not an invented naming.
- `EulerBernoulliBendingTheory` / `EulerBernoulliBendingConstitutive`
  (`physics/elasticity/euler_bernoulli.py`, new): the first bending `Theory`/`ConstitutiveModel`
  pair - composes the `second_gradient` operator (forcing `Continuity.C1`, satisfied by the
  existing `HermiteInterpolation`), `M_per_I = E * kappa` as its own dedicated constitutive law
  (not a reuse of `IsotropicElasticConstitutive` - see `docs/design/BEAM_ELEMENT.md` §3 for why).
  Includes `_curvature_from_hessian` (the theory-specific Hessian contraction
  `second_gradient_tensor` defers) and `_reference_derivative_scale` (see below).
- The composed-path (ADR-002/SDS E-5) verification test discovered and corrected a real subtlety:
  `HermiteInterpolation`'s "derivative" DOFs are `dw/dxi` (reference-coordinate), not `dw/dx`
  (the physical rotation this element's DOF actually is) - composing the raw Hessian pipeline
  against physically-parametrized rotation DOFs reproduces `K` off by exactly the Jacobian `J` per
  rotation row/column and `J^2` on the rotation-rotation term. `_reference_derivative_scale`
  corrects it; documented in full in `docs/design/BEAM_ELEMENT.md` §4 and `docs/dev/notes.md`.
- New test files: `test_beam_eb_analytical.py` (closed-form + symmetry + correct rigid-body
  null-space checks - `[0,1,L,1]` for rigid rotation about node 1, not `[0,1,0,1]`),
  `test_elasticity_euler_bernoulli_analytical.py`, `test_beam_eb_verification.py` (the composed-
  path equivalence proof), `test_beam_eb_cantilever_benchmark.py` (single-element tip-loaded
  cantilever vs. `PL^3/(3EI)`/`PL^2/(2EI)`, a free preview of the later verification-suite's
  cantilever-beam benchmark).
- `docs/design/BEAM_ELEMENT.md`.

### Notes
- `Bar`, `ContinuumElement`, all of `numerics/`, and `physics/elasticity/isotropic.py` are
  unmodified - purely additive. `EulerBernoulliBeam` is not wired into `elements/factory.py`'s
  dispatch this phase, mirroring the `ContinuumElement` precedent.

## [0.9.0] - 2026-07-30

### Added
- The Element Integration Framework (`elements/continuum/continuum.py`, new): `ContinuumElement`,
  a dimension-generic element combining `numerics.interpolation` + `numerics.mapping` +
  `numerics.quadrature` + `numerics.operators` + a `Theory`/`ConstitutiveModel`/`Material` into
  local stiffness, mass, and body-force matrices - the general composition SDS clause E-5
  describes, as opposed to `Bar`'s closed-form ADR-002 exception. Single-field theories and
  affine (straight-sided) geometry only this phase; verified against the 1-D case (reproduces
  `Bar.local_stiffness()` exactly, plus independent hand-derived consistent-mass-matrix and
  uniform-body-force checks). Flattens operator output `(n_qp, n_voigt, n_fun, dim)` into the
  `(n_voigt, n_dof)` B-matrix SDS Section 8 anticipates via node-major-then-component ordering
  (SDS C-2) - a pure `reshape`, the first place in the codebase this flattening decision is made
  (deferred explicitly since v0.7.0, `docs/dev/notes.md` N-42).
- `ContinuumBodyForceProvider` (same file): the first provider for the `BodyForce` dataclass
  (declared since v0.1.0, previously unconsumed) - a CELL FORCE `ContributionProvider` that
  integrates `∫ N_a b_c dΩ` per node/component using an element's own tabulated shape/quadrature
  data. Lives in `elements/`, not `constraints/`, per the import-linter layer contract.
- `elements/dof_utils.py` (new): `build_local_dof_map()`, a node-major-then-component
  local-to-global DOF array builder, generalizing the by-hand DOF pair `elements/factory.py`
  builds for `Bar`'s 2-node case to arbitrary node/component counts.
- `docs/design/ELEMENT_INTEGRATION.md`; `tests/unit/test_continuum_element.py` (9 tests:
  stiffness-vs-`Bar` equivalence, hand-derived mass matrix, hand-derived body-force vector,
  `ContinuumBodyForceProvider` global-equilibrium check, lazy-`rho` safety, multi-field
  rejection).

### Notes
- `Bar`, `numerics/operators/`, `numerics/assembly/`, and `physics/elasticity/isotropic.py` are
  unmodified - purely additive. `ContinuumElement` is not wired into `elements/factory.py`'s
  dispatch this phase (a separate policy decision: whether 1-D `IsotropicElasticity` domains
  should prefer the closed-form `Bar` or the generic path). `test_bar_verification.py` is
  unmodified for the same reason its own docstring gives: its composed-path oracle must share no
  code with anything it verifies.

## [0.8.0] - 2026-07-29

### Added
- The walking skeleton (ARCHITECTURE_v2.md's named phase-0 milestone): a bar element solved
  end-to-end, `K u = f`, through every seam from `Mesh` to `LinearStaticAnalysis`. See
  `docs/design/WALKING_SKELETON.md` for the full pipeline, the ADR-002 equivalence argument,
  and the SDS-module-to-concrete-class map.
- `geometry/standard.py`: `CircularSection` is now fully real (all 9 `CrossSection` methods,
  each an exact SDS 2.2 closed form, including the circular-only `J_t = I_p` identity and
  exact `C_w = 0`). Every other section stays a deferred stub.
- `physics/elasticity/isotropic.py` (new): `IsotropicElasticity` (`Theory`) and
  `IsotropicElasticConstitutive` (`ConstitutiveModel`), dim=1 (axial) only; `dim != 1` raises
  `PhysicsError` naming the plane-stress/strain reduction gap. The constitutive law is the real
  `sigma = E eps`, `D = [[E]]`, not derived from the tensor layer's verification-oracle
  `isotropic_stiffness`.
- `numerics/assembly/sparsity.py`: real `SparsityPattern` (`from_providers`, `contains`,
  `num_nonzeros`) - the union of `(row, col)` pairs every provider touches for one role.
- `numerics/assembly/assembler.py`: real `Assembler` - COO-triplet scatter with pattern-violation
  and mixed matrix/vector-contribution checks (both `AssemblyError`, naming the offending
  provider), C-7 symmetry assertion per square contribution block under `__debug__`, returning a
  `scipy.sparse.csr_matrix` (matrix role) or dense `NDArray` (vector role).
- `numerics/assembly/system.py`: real `GlobalSystem` (role-keyed operators/vectors) and
  `ReducedSystem` (`from_global`, `recover`, `reactions`) implementing Dirichlet elimination
  (SDS 2.14: `K_ff u_f = f_f - K_fc u_c`, `R = K_cf u_f + K_cc u_c - f_c`). MPC transformation
  remains out of scope - no model here declares one.
- `numerics/linalg/linear.py`: real `SparseDirectSolver` (`scipy.sparse.linalg.spsolve`),
  raising `SingularMatrixError` on a non-finite solution. The richer zero-pivot -> DOF back-map
  diagnostic SDS 2.15 describes is deferred, not dropped.
- `constraints/handler.py`: real `ConstraintHandler.partition()` and a new `DofPartition`
  dataclass - resolves each `DirichletBC`'s region to DOFs, raises `ConstraintConflictError` on
  conflicting values, returns free/constrained DOFs with constrained DOFs sorted for
  determinism (SDS C-5).
- `constraints/loads.py`: `NodalLoadProvider`, a `NodalLoad` as a VERTEX FORCE
  `ContributionProvider` (structural typing, no inheritance) - resolves `field_components`
  against a `DofHandler` to emit one contribution per node in the load's region.
- `elements/structural/bar.py` (new): `Bar`, the closed-form axial element (ADR-002) -
  `K = (EA/L)[[1,-1],[-1,1]]` computed directly, embedded in 1-D space only this phase;
  `transformation_matrix()` is the honestly trivial `np.eye(2)` (E-10), asserted orthonormal.
- `elements/factory.py` (new): `build_elements(model, dof_handler)` - explicit dispatch (not a
  plugin registry) from `Model` domains to `Bar` instances, for `IsotropicElasticity` domains
  over `line2` cells only.
- `analysis/static.py`: real `LinearStaticAnalysis.run()` and a new `StaticResult` dataclass -
  assembles STIFFNESS once and reuses it across load cases; per load case, assembles FORCE,
  reduces, solves, recovers, and computes reactions. Raises `ModelError` on a load-case entry
  using a time function (static analysis has no time axis) or any load type other than
  `NodalLoad`.
- `examples/ex08_bar_under_end_load.py`: builds the model, solves it, and checks the tip
  displacement, reaction, and closed-form-vs-composed-path stiffness against independent hand
  computations.
- New test files: `test_bar_analytical.py`, `test_bar_verification.py` (the ADR-002/SDS E-5
  equivalence proof plus the dim=1 isotropic-oracle cross-check), `test_assembly_analytical.py`
  (a patch-test-style collinear bar chain), `test_static_analytical.py`,
  `test_geometry_circular_analytical.py`, `test_elasticity_isotropic_analytical.py`,
  `test_constraints_partition_analytical.py`, `test_linalg_direct_analytical.py`.
- `docs/design/WALKING_SKELETON.md`; a cross-reference sentence in `OBJECT_MODEL.md`.

### Changed
- `core/model.py`: `Model` gains five read-only accessors - `domains`, `materials`, `sections`
  (each a one-line mirror of the existing `theories` property), `dirichlet_bcs`, and
  `load_case(name)` (dict lookup, `ModelError` naming registered cases on a miss). No existing
  method's signature or behavior changed.
- `test_materials_geometry.py` and `test_model_and_analysis.py`: `CircularSection` and
  `LinearStaticAnalysis` are excluded from their respective blanket
  "everything still raises `NotImplementedError`" loops, since both are now real; each gained
  its own dedicated, passing coverage instead.
- `numerics/math/__init__.py` and `numerics/tensors/__init__.py`: added the missing `TODO`
  section their package docstrings required (a pre-existing v0.7.0 gap caught by the full
  verification gate, not new to this phase).

## [0.7.0] - 2026-07-29

### Added
- Tensor algebra layer (`numerics/tensors/`, SDS Section 9): second-order algebra
  (`second_order.py` - symmetric/skew parts, trace, determinant, inverse, deviator, Frobenius
  norm, outer products, contractions, `is_symmetric`); fourth-order algebra (`fourth_order.py` -
  the identity, symmetrizer, volumetric/deviatoric projectors `J`/`K`, the isotropic oracle
  `C = 3*kappa*J + 2*mu*K`, major/minor symmetry classification); Voigt/Mandel/full converters
  (`voigt.py` - `strain_to_voigt`/`stress_to_voigt` and their inverses per SDS C-1's
  kinematic/kinetic split, `full_to_mandel`/`mandel_to_full`, `voigt_to_mandel`/`mandel_to_voigt`,
  `fourth_order_to_mandel`/`mandel_to_fourth_order` - the only sanctioned representation bridge);
  `SO(d)` rotation and the Bond transformation pair (`rotations.py` - `is_rotation`,
  `rotate_vector`, `rotate_second_order`, `bond_matrix_stress`, `bond_matrix_strain` computed as
  `bond_matrix_stress`'s inverse-transpose rather than re-derived, `rotate_stiffness_voigt`);
  invariants and spectra (`invariants.py` - `I1`/`I2`/`I3`, `J2`/`J3`, von Mises, batched
  `principal_values`/`principal_directions`).
- `numerics/math/rotations.py`'s `rotation_matrix_2d` stub filled (a one-line `[[cos,-sin],
  [sin,cos]]` construction), unblocking `tensors/rotations.py` and `operators/transformation.py`.
- Discrete operator library (`numerics/operators/`, SDS Section 8): `gradient_matrix`,
  `divergence_matrix`, `curl_matrix` (2-D scalar rotation and 3-D vector curl, via the
  Levi-Civita symbol), `symmetric_gradient_matrix` (the Voigt-ordered B operator, keyed off the
  same `VOIGT_ORDER` table `tensors` uses), `laplacian_matrix`, `helmholtz_matrix` (length scale
  passed per call, never constructor state), `surface_gradient_matrix`/`surface_projector`
  (`P = I - n(x)n`), `second_gradient_tensor` (a deliberately thin, validated pass-through of
  `GeometricMapping.physical_hessian`'s output), `identity_vector`/`deviatoric_projector_voigt`/
  `trace_row_voigt`, and `vector_transformation_matrix`/`tensor_transformation_matrix_voigt`
  (dispatching to the tensor layer's Bond matrices). Each recipe pairs a plain function with a
  thin `DiscreteOperator` subclass carrying its own `verify()`/`is_valid()`.
- `numerics/operators/future.py`: `NonlocalIntegralOperator`, a declared placeholder
  (`PROVISIONAL_METADATA` + `BLOCKED_BY`) for the two-phase Eringen pairwise operator - deferred
  because it structurally needs a concrete `kernels.Kernel` and `numerics.search.NeighborSearch`,
  both still stubs, matching the standard already applied to `GaussJacobiQuadrature` and
  `CurvilinearMapping`.
- `operators/registry.py` (`OPERATOR_REGISTRY`, `available_operators()` - every real recipe
  except the future placeholder) and module-level verification suites for both packages
  (`tensors/verification.py`: `verify_tensor_library`/`is_tensor_library_valid`;
  `operators/verification.py`: `verify_operator_library`/`is_operator_library_valid`, plus
  registry self-consistency, continuity-derivation, and cross-operator-consistency checks).
  Verification is module-level for `tensors` (pure functions, no single stateful class to hang
  it off, unlike every prior phase) and per-class for `operators` (real `DiscreteOperator`
  subclasses, so class-based trip testing still applies).
- New local exception hierarchies, matching the `numerics/mapping/errors.py` and
  `numerics/quadrature/errors.py` precedent rather than `utils/exceptions.py`: `tensors/errors.py`
  (`TensorError`, `RepresentationError`, `NotRotationError`, `TensorSymmetryError`) and
  `operators/errors.py` (`OperatorError`, `OperatorShapeError`, `UnsupportedDimensionError`).
- ~90 unit tests across four tensor test files (analytical algebra, Voigt/Mandel conversions
  and the work-conjugacy identity with its factor-of-2 trip test, rotations and invariants, the
  module-level verification suite) and two operator test files (analytical closed forms
  including a patch test and the constant-strain-triangle Laplacian pattern, per-class
  verification with the future-placeholder and independence-proof tests); `docs/design/TENSORS.md`
  and `docs/design/OPERATORS.md`; `examples/ex07_operators_and_tensors.py`.

### Changed
- `numerics/tensors/__init__.py` and `numerics/operators/__init__.py` gain curated public-API
  re-exports (`__all__`), matching the pattern already established by `numerics/quadrature/__init__.py`
  rather than remaining docstring-only stubs.

## [0.6.0] - 2026-07-25

### Added
- Quadrature layer (`numerics/quadrature/`, SDS 2.5): points and weights with declared exactness
  on reference domains. No Jacobian-weighting, physical integration, element matrices, assembly,
  or constitutive models.
- `QuadratureRule` ABC: concrete rules declare `family`, `reference_element`, `order`,
  `exactness_degree`, `points`, and `weights`; integration, moments, measure, centroid,
  verification, serialization, and value semantics are derived and shared.
- `GaussLegendreQuadrature` (exact to 2n-1) and `GaussLobattoQuadrature` (2n-3, endpoints
  included) on the line; `TensorProductQuadrature` on the square, isotropic and anisotropic via
  `from_rules`; `DunavantQuadrature` degrees 1-5 on the triangle.
- `moments.py`: exact monomial integrals by closed form (interval and simplex) and monomial
  evaluation, written locally so the layer borrows nothing from the interpolation tabulator.
- `symmetry.py`: the affine symmetry group of a reference domain, derived generically from vertex
  permutations - six for the triangle, eight for the square, two for the line.
- Verification suite: weight normalization, positivity (reported, not imposed), points in domain,
  exactness against the closed forms, maximality of the declared degree, moment identities, and
  symmetry; plus per-family checks (tensor-product construction, barycentric consistency).
- `quadrature()` factory + `QuadratureFactory` (the phase-0 seam, taking a `ReferenceCell`) with
  the SDS 2.5 policy: full integration by default, implicit reduction prohibited; per-domain
  defaults; memoized immutable rules.
- Declared placeholders `GaussJacobiQuadrature`, `AdaptiveQuadrature`, `SparseGridQuadrature`,
  each with `PROVISIONAL_METADATA` and a `BLOCKED_BY` string.
- ~150 unit tests: closed forms by hand, Gauss/Lobatto/Dunavant against textbook values,
  integration of polynomial, transcendental, and vector fields, the full verification suite, a
  trip test per identity, symmetry-group derivation and orbit recovery, tensor-product
  correctness (isotropic and anisotropic), caching, serialization, regressions, and proof by
  subprocess and source scan that the layer imports no interpolation or mapping code.
- `docs/design/QUADRATURE.md`; `examples/ex06_quadrature.py`.

### Changed
- The phase-0 `QuadratureRule` value object becomes an abstract base carrying a
  `ReferenceElement` (the reference-element layer now owns the domain the `cell_name: str` draft
  predated); `QuadratureFactory` is filled rather than bypassed, as the `ShapeFunctions` seam was
  in phase 4.

## [0.5.0] - 2026-07-17

### Added
- Geometric mapping layer (`numerics/mapping/`, SDS 2.6): reference coordinates -> physical
  coordinates -> Jacobian -> inverse Jacobian -> metric tensor -> physical derivatives. No
  quadrature, integration, element matrices, assembly, or constitutive models.
- `GeometricMapping` ABC: concrete maps declare `map`, `inverse_map`, `jacobian`,
  `mapping_hessian`, and `is_affine`; everything else is derived and shared - transpose,
  pseudo-inverse, determinant, Gram measure scaling, condition number, metric tensor and its
  inverse, covariant and contravariant bases, gradient push-forward and pull-back, physical
  Hessian, centroid, bounding box, edge chords, characteristic length, aspect ratio,
  scaled-Jacobian quality, physical measure, caching, validation, and verification.
- `AffineMapping`: derives `(A, b)` by least squares from the vertex correspondence and
  verifies the fit is exact, so the parallelogram condition for quadrilaterals emerges as a
  residual rather than being hard-coded; closed-form exact inverse; `from_linear_map`.
- `IdentityMapping`: a genuine `AffineMapping` subclass (`A = I`, `b = 0`).
- `IsoparametricMapping`: geometry contracted from a phase-4 `ShapeFunctionFamily` against
  node coordinates; Newton inversion; `is_affine` answered from the mapping Hessian, so a
  parallelogram `Q1` and a straight-sided `P2` are both correctly detected as affine.
- Embedded elements (a bar in a plane, a triangle in space): non-square Jacobians throughout,
  with the Gram determinant for measure and the pseudo-inverse for the contravariant basis;
  `jacobian_determinant` and `physical_hessian` raise `EmbeddedMappingError` with the
  mathematical reason rather than inventing an answer.
- Physical Hessian including the mapping-curvature correction
  `H_x = J^+T (H_xi - grad_x N . K) J^+`, which the phase-0 SDS 2.6 note calls mandatory
  unless the map is affine.
- `MappingType` enum; `MappingError`, `NonAffineError`, `InverseMapError`,
  `EmbeddedMappingError`; `DegenerateCellError` reused for degeneracy per the exception tree.
- Declared placeholders `CurvilinearMapping`, `NURBSMapping`, `HighOrderMapping`, each with
  `PROVISIONAL_METADATA` and a `BLOCKED_BY` string.
- 93 unit tests: analytical and random affine maps, identity, embedded elements, bilinear and
  curved isoparametric geometry, Newton inversion, the Hessian correction checked against a
  physical-space finite difference, every degeneracy detector fired, a trip test per
  verification, caching, scale invariance across five decades, regressions, and a
  no-quadrature-import proof.
- `docs/design/GEOMETRIC_MAPPING.md`; `examples/ex05_geometric_mapping.py`.

### Fixed
- Degeneracy is judged from the singular values of `J`, not from `det J` or `det(J^T J)`. The
  absolute-tolerance version rejected a legitimate nanometre-scale element as degenerate, and
  the Gram determinant squares the condition number so a rank-deficient map left a measure
  scaling near 1e-8 that no sensible threshold caught. The new criterion is dimensionless and
  scale invariant.
- `inverse_jacobian` is computed by SVD pseudo-inverse rather than `(J^T J)^-1 J^T`, which
  squared the conditioning and leaked `LinAlgError` out of a geometry query.
- A diverging Newton iterate that reaches a degenerate point is reported as an inverse that
  could not be continued, not as element degeneracy - the element is sound and the user should
  not be sent to inspect the mesh.

## [0.4.0] - 2026-07-17

### Added
- Shape function library: the nodal basis constructed from the interpolation triple by
  `C = M^-T`, where `M` is the generalized Vandermonde phase 3 built and proved invertible.
  Reference coordinates only - no Jacobian, mapping, physical derivative, B-matrix,
  quadrature, integration, or assembly.
- `ShapeFunctionFamily` ABC implementing the phase-0 SDS 2.4 `ShapeFunctions` contract
  (`cell`, `continuity`, `completeness_degree`, `evaluate`, `derivatives`) and extending it
  with `gradient`, `hessian`, `derivative` (any multi-index), `tabulate`, `interpolate`,
  `coefficients`, `duality_matrix`, caching, and value semantics.
- `LagrangeShapeFunctions` and `HermiteShapeFunctions` for line, triangle, and quadrilateral
  at orders 1-3, plus the `shape_functions()` factory, `SHAPE_FUNCTION_FAMILIES`, and a
  `ShapeFunctionFactory` protocol.
- `numerics/interpolation/tabulation.py`: `monomial_table` (vectorized `D^a x^b` for whole
  batches), the `Tabulation` value object, `as_points`, and the `PointsLike` alias.
- Verification suite: `verify_kronecker_delta` (duality `l_k(N_i) = delta_ki`),
  `verify_partition_of_unity` (values and gradients), `verify_polynomial_reproduction`,
  `verify_derivative_consistency` (central finite differences for gradient and Hessian),
  `verify_symmetry`, `verify_boundary_restriction`, `verify_interpolation_exactness`, plus
  each family's idiomatic check; `verify()` and `is_valid()`.
- Per-instance tabulation cache keyed by (points, multi-index) with read-only shared arrays
  (SDS C-8), `cache_info()`, and `clear_cache()`.
- 199 unit tests: closed-form comparisons against independently written references (Lagrange
  product formula, classical Hermite cubics, barycentric triangle polynomials, tensor-product
  factorization of the quad and BFS bases), the full verification suite, a trip test per
  identity, caching and tabulation, regressions pinning coefficient matrices and nodal values,
  and a subprocess test asserting no quadrature or mapping module is imported.
- `docs/design/SHAPE_FUNCTIONS.md`; `examples/ex04_shape_functions.py`.

### Changed
- `REFERENCE_CELLS` and the cell-name bridge now register `line4`, `tri10`, `quad9`, and
  `quad16`, closing dev note N-21: the entries land with the family that consumes them, and
  `ShapeFunctionFamily.cell()` is that consumer. `quad8` remains registered and unused,
  awaiting the serendipity family.
- The phase-0 stubs `LagrangeLine2` and `HermiteBeamLine2` are superseded by
  `LagrangeShapeFunctions` / `HermiteShapeFunctions` and removed; the `ShapeFunctions` seam
  they declared is now filled rather than bypassed.
- Evaluation entry points are annotated `PointsLike`, matching the documented contract that a
  single point may be passed as a flat sequence.

## [0.3.0] - 2026-07-17

### Added
- Complete interpolation framework (`numerics/interpolation/`), describing finite elements as
  the Ciarlet triple (reference domain, polynomial space, degrees of freedom) with no shape
  function formed, evaluated, or differentiated.
- `PolynomialSpace`: monomial exponent sets for `P_k` (total degree) and `Q_k` (tensor
  product), keeping order, completeness degree, and maximum total degree distinct; graded
  lexicographic ordering; structural queries and `expected_dimension` cross-check.
- `DofFunctional` (point values and derivative multi-indices, with `apply_to_monomial`) and
  `InterpolationNode` (located points with entity associations).
- `Interpolation` ABC: derived metadata (order, family, space, nodes, DOFs, identifiers,
  continuity, support dimension, nodal/simplex/tensor-product flags, `dofs_per_node`,
  `mesh_cell_name`), queries (`interpolation_nodes`, `nodes_on_entity`, `dofs_on_node`,
  `evaluation_points`), the `unisolvence_matrix()` oracle and its condition number,
  serialization, and value semantics by (family, cell, order).
- Verification suite, all derived from the space and functionals alone:
  `verify_polynomial_degree`, `verify_polynomial_completeness`, `verify_linear_independence`,
  `verify_kronecker_delta`, `verify_partition_of_unity`, `verify_node_ordering`, `validate`.
- `LagrangeInterpolation` orders 1-3 on line, triangle, and quadrilateral (line2/3/4,
  tri3/6/10, quad4/9/16), with node placement derived from the reference element's own vertex
  and edge numbering.
- `HermiteInterpolation`: the cubic line (P3, 4 DOFs, C1, classical `(w1, theta1, w2, theta2)`
  beam ordering) and the bicubic Bogner-Fox-Schmit quadrilateral (Q3, 16 DOFs, C1 on
  rectangular meshes). The Hermite triangle raises with its reason (the cubic Hermite triangle
  is only C0; C1 on simplices needs Argyris or HCT).
- Declared placeholders `SerendipityInterpolation`, `HierarchicalInterpolation`, and
  `SpectralInterpolation`, each carrying `PROVISIONAL_METADATA` and a `BLOCKED_BY` string
  naming what it structurally needs.
- Registry and factories: `INTERPOLATION_FAMILIES`, `AVAILABLE_INTERPOLATIONS`,
  `interpolation()`, `available_interpolations()`, `interpolation_from_dict()`, and an
  `InterpolationFactory` protocol expressing the constructor contract.
- Error types `InterpolationError`, `PolynomialSpaceError`, `UnisolvenceError`,
  `NodeOrderingError` (rooted at `NanoFEMError`).
- 106 unit tests across polynomial spaces, families, and registry/serialization, including a
  trip test for every reachable validation rule, convention regressions, and a subprocess test
  proving the layer needs only the reference layer; `examples/ex03_interpolation.py`;
  `docs/design/INTERPOLATION.md`.

### Changed
- `numerics/interpolation/base.py` now documents both sides of the phase boundary: the
  existing `ShapeFunctions` evaluation contract is preserved untouched as the next phase's
  seam, alongside the new `Interpolation` contract.

## [0.2.0] - 2026-07-16

### Added
- Complete reference element library (`numerics/reference/`), the permanent geometric
  foundation: `ReferenceElement` ABC with data-driven subclasses, plus `ReferenceLine`
  (`[-1, 1]`), `ReferenceTriangle` (unit right triangle, facet i opposite vertex i), and
  `ReferenceQuadrilateral` (`[-1, 1]^2`, facets bottom/right/top/left).
- Strongly typed enumerations: `CellType` (with `topological_dimension` and
  `is_implemented`), `EntityType`, `FacetType`, `Orientation` (with `sign`), `Dimension`.
- Topological operations: vertex/edge/face/facet counts, entity and connectivity queries,
  incidence, boundary extraction, facet permutation under orientation.
- Geometric operations: centroid, reference measure, bounding box, edge lengths, diameter
  and characteristic length, local axes, reference tangents, outward facet normals, point
  containment, signed and unsigned distance to the boundary.
- Validation (`validate()` / `is_valid()`): dimension consistency, vertex count and
  uniqueness, incidence validity/arity/uniqueness, outward orientation, unit normals and
  tangents, and declared-vs-computed measure agreement.
- Utilities: `pretty()`, `debug_summary()`, `to_dict()`, `to_json()`, `visualization_data()`
  (plain data, no plotting import), value semantics (`__eq__`/`__hash__` by shape).
- Registry and factories: `REFERENCE_ELEMENTS`, `reference_element()`,
  `reference_element_for_name()` (bridges mesh names like `"tri6"` to `CellType`),
  `reference_element_from_dict()`.
- Declared 3-D placeholders (`ReferenceTetrahedron`, `ReferenceHexahedron`, `ReferencePrism`,
  `ReferencePyramid`) carrying `PROVISIONAL_TOPOLOGY` and raising on construction.
- Error types `ReferenceElementError`, `TopologyError`, `OrientationError` (rooted at
  `NanoFEMError`).
- 82 unit tests across topology, geometry, and serialization, including regression tests
  pinning the frozen conventions and a subprocess test proving the layer stands alone;
  `examples/ex02_reference_elements.py`; `docs/design/REFERENCE_ELEMENTS.md`.

### Fixed
- Package version and `pyproject.toml` version had drifted apart (0.1.0 vs 0.0.1); both are
  now 0.2.0.

## [0.1.0] - 2026-07-16

### Added
- Complete phase-1 object model: `Node`, `CellBlock`/`Cell`, `Mesh` (topology and
  connectivity queries, frozen arrays, fail-fast integrity), `Region`, `FieldSpec` +
  `VariableType`, `Dof`/`DofHandler` (deterministic C-2 numbering, export/import,
  fingerprints), `Material` (bounded canonical keys, namespaced user keys, G-consistency,
  auxetic-friendly nu bounds), section geometry records, `Theory`/`TheoryDeclaration`/
  `DeclaredTheory` + `ConstitutiveDeclaration`, state lifecycle (`StateLayout`, `ModelState`
  trial/committed/commit/revert, bounded `StateHistory`, QP views), boundary-condition and
  load data objects (`DirichletBC`, `NeumannBC`, `RobinBC`, `MultiPointConstraint`, four load
  types, `LoadCase`, `TimeFunction` family), analysis metadata classes with validated frozen
  options, and the `Model` facade (`DomainDefinition` bindings, materialized field specs,
  fail-fast `validate()`, `build_dof_handler()`, versioned manifest, SHA-256 fingerprint).
- `OPERATOR_CATALOG` + `derived_continuity` (SDS Section 8 as data); extended
  `ReferenceCell` registry; implemented validation/logging/serialization utilities.
- Unit test suite for the whole object model (tests/unit, 40 tests); runnable success-
  criterion example `examples/ex01_object_model.py`; design document
  `docs/design/OBJECT_MODEL.md`.

### Fixed
- `DofHandler.import_numbering` now restores per-field `VariableType` and `Continuity`
  (round-trip fingerprints previously diverged); malformed records raise `DofMappingError`
  instead of asserting.
- Analysis constructors no longer instantiate default options in argument defaults (B008).

## [0.0.1] - 2026-07-15

### Added
- Phase-0 architectural skeleton: complete package tree per the frozen SDS,
  placeholder contracts, exception hierarchy, contribution vocabulary
  (kinds, roles, provider protocol).
- Tooling: black, isort, ruff, mypy (strict), pytest + coverage, pre-commit,
  import-linter architecture contracts, GitHub Actions CI.
- Documentation skeleton (Diataxis), validation framework layout, plugin
  template, developer notes.
- Zero numerical implementation, by design (phase-0 success criterion).
