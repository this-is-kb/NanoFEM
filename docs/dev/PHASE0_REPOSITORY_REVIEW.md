# NanoFEM — Phase 0 Repository Review (v0.10.0 → v0.10.1)

Mandatory pre-development audit requested before continuing the Classical FEM
Pipeline. Scope: the entire repository — the "frozen" mathematical foundation
(`mesh/`, `core/`, `numerics/{reference,interpolation,mapping,quadrature,
tensors}/`, `materials/`, `geometry/`, `utils/`) and the classical-pipeline
layers built v0.7.0–v0.10.0 (`numerics/{operators,assembly,linalg}/`,
`physics/`, `elements/`, `constraints/`, `state/`, `analysis/`,
`postprocess/`). Conducted via two independent research passes, cross-checked
against `docs/dev/notes.md` (N-1..N-55) and `docs/design/*` so nothing already
explained and deliberate got flagged as a bug.

## Summary

- **173 source files, 461 import-linter-tracked dependencies, 0 circular
  imports** — the layered architecture contract (4/4 rules) held before and
  after every change below.
- **No accidental placeholder gaps.** Every `NotImplementedError`/TODO found
  across both packages is a documented, phase-fenced decision (3-D reference
  elements, Serendipity/Spectral/Hierarchical interpolation, non-affine
  measure, `GaussJacobiQuadrature`/`AdaptiveQuadrature`, FGM grading
  evaluation, `ConjugateGradientSolver`/`EigenSolver` family, Neumann/Robin
  assembly, `postprocess/*`, all `physics/{eringen,couple_stress,
  strain_gradient,...}` stub packages). None of these were touched.
- **No unnecessary abstractions.** Every ABC/Protocol in scope has ≥2 real
  implementations or a recorded single-consumer justification (`ConstraintLike`
  /`LoadCaseLike`, N-9).
- **9 real, fixable issues found; all fixed.** **6 categories of missing test
  coverage found; all closed additively**, no production code touched to add
  them. **1209 tests pass** (was 1193 before this review), gate green
  (black/isort/ruff/mypy --strict/import-linter 4/4).

## Issues found and fixed

1. **Dead, name-colliding class** (`analysis/results.py`): an empty
   `StaticResult` stub shadowed the real, used `StaticResult` in
   `analysis/static.py` — a wrong import (`from nanofem.analysis.results
   import StaticResult`) would have silently returned the empty class.
   Removed; `ModalResult`/`BucklingResult`/`TransientResult` (legitimate
   placeholders for analyses that don't exist yet) kept.
2. **Duplicated canonical-key vocabulary**: `materials/properties.py`'s
   `CANONICAL_KEYS` hardcoded the same 13 keys `materials/material.py`'s
   `_BOUNDS` already defines and validates against. `CANONICAL_KEYS` now
   derives from `_BOUNDS` — one source of truth, cannot drift apart.
3. **Duplicated enum-resolution helper**: `_resolve_cell_type` was
   byte-for-byte identical in `numerics/reference/registry.py` and
   `numerics/interpolation/registry.py`; `interpolation/registry.py`'s
   `_resolve_family` was the same pattern for a different enum. Extracted one
   shared `resolve_enum_member()` in `utils/validation.py`; all three call
   sites now delegate to it. (`quadrature/factory.py`'s similarly-named
   resolvers were left alone — they have genuinely different behavior,
   default-family fallback and mesh-name coercion, not pure duplication.)
4. **Two precision-losing `Any` type hints**: `ShapeFunctions.reference_element`
   (`numerics/interpolation/shape_functions.py`) returned `Any` instead of the
   already-imported-elsewhere `ReferenceElement`; `GeometricMapping._cached`'s
   `build` parameter (`numerics/mapping/base.py`) was `Any` instead of
   `Callable[[], NDArray[np.float64]]`. Both fixed; mypy --strict stayed clean.
5. **Stale documentation** in three package `__init__.py` files claiming
   unimplemented work that was actually done: `core/__init__.py` (Model
   validation and deterministic DOF numbering, both real since early phases),
   `mesh/__init__.py` (region query API, real since phase 1), `analysis/
   __init__.py` (`static.py`/`results.py` listed as "future" despite `static.py`
   being fully real since v0.8.0). All three corrected to reflect actual state.
6. **A factually wrong module docstring**: `utils/serialize.py` claimed its
   `encode_array`/`decode_array` are "shared by every `to_dict`/`from_dict`" —
   they have zero call sites anywhere; every actual `to_dict`/`from_dict`
   hand-rolls its own array encoding. Docstring corrected to describe their
   actual (currently unused, still correct) status; added round-trip tests
   rather than deleting working, spec-mandated (SDS C-4) code with no current
   caller.
7. **A stale dev note**: N-47 claimed a test loop asserted `HollowCircularSection`
   (among others) still raises `NotImplementedError` — no such test existed in
   the current `test_materials_geometry.py`. Added the real test rather than
   just correcting the note, since the underlying gap (zero direct test
   coverage of `HollowCircularSection`/`HollowRectangularSection`/`ISection`)
   was real.

## Missing test coverage closed (additive only, no production API changes)

`core/registry.py::Registry` (new `test_core_registry.py`, 6 tests: register/
get/keys round trip, sorted-keys determinism, namespaced-key requirement,
reserved-prefix protection, duplicate-registration rejection, unknown-key
listing) · `materials/grading.py` (`PowerLawGrading`/`ExponentialGrading`
validate-but-defer) · `materials/properties.py::SpatialProperty` ·
`geometry/standard.py`'s `HollowCircularSection`/`HollowRectangularSection`/
`ISection` · `core/fields.py`'s `rotation_z`/`temperature`/
`electric_potential`/`pressure` factories · `utils/config.py::Configuration`,
`utils/logging.py::get_logger`, `utils/scaling.py::Nondimensionalizer`,
`utils/serialize.py::encode_array`/`decode_array` (new `test_utils_misc.py`,
5 tests).

## Repository cleanup

This project is **not a git repository** (no `.git` directory) — "remove
generated files from version control" doesn't literally apply since nothing
is tracked by any VCS; noting this rather than silently no-op-ing or
fabricating git actions. `.gitignore` already covered every pattern
requested; added the one missing entry (`.import_linter_cache/`). Deleted
~58MB of regenerable tool caches (`.mypy_cache`, `.pytest_cache`,
`.ruff_cache`, `.import_linter_cache`, all `__pycache__` dirs) — safe, they
rebuild automatically. Left `.venv` (382MB) and `src/nanofem.egg-info` in
place: both are required for this session's own verification gate to keep
running; removing them is a separate, larger decision (full environment
rebuild) not bundled into a routine cleanup pass.

## Remaining technical debt (found, deliberately not fixed, with reasoning)

- **`Bar`/`EulerBernoulliBeam` share ~15 lines of near-identical constructor
  validation and `transformation_matrix()` boilerplate.** Not extracted: only
  two data points exist for the closed-form-structural-element pattern: a
  shared base is premature abstraction until a third element (Timoshenko,
  next in the roadmap) confirms the shape is stable.
- **`elements/dof_utils.py::build_local_dof_map`** has no production caller
  yet — only `test_continuum_element.py` uses it. This is the same
  already-confirmed, documented scoping decision as `ContinuumElement` not
  being wired into `elements/factory.py`'s dispatch (v0.9.0/v0.10.0); not a
  new finding.
- **Package `__init__.py` export style is inconsistent**: `numerics/{reference,
  interpolation,mapping,quadrature,tensors}` re-export their public surface
  via explicit `__all__`; `core`, `mesh`, `materials`, `geometry`, `utils` are
  pure docstrings with no re-exports, forcing direct submodule imports.
  Stylistic, not a bug — left alone since standardizing it would touch many
  files for zero functional gain, and nothing consumes these packages via a
  top-level `import nanofem.core` expecting re-exports today.
- **`IsotropicElasticConstitutive`/`EulerBernoulliBendingConstitutive`** and
  **`mass_term`/`laplacian_matrix`/`elements/continuum/continuum.py`'s
  `_integrate`** both "look like near-duplication" on casual inspection but
  are already deliberately-documented exceptions (N-55, N-50–N-52
  respectively) — re-examined during this review and the reasoning holds up;
  not re-litigated.

## Verification

`black --check`, `isort --check-only`, `ruff check`, `mypy` (strict, 218
files), `import-linter` (4/4 contracts kept, 461 dependencies), `pytest`
(1209 passed, 0 failed) — all run after every fix in this review, not just at
the end.
