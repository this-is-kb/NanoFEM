# NanoFEM Unified Element Factory (v0.14.0)

**Status:** implemented and tested. Companion to `docs/design/WALKING_SKELETON.md` (the original
`Bar`-only factory), `ELEMENT_INTEGRATION.md`/`PLANE_ELASTICITY.md`/`Q4_QUADRILATERAL.md`
(`ContinuumElement`), `BEAM_ELEMENT.md`/`TIMOSHENKO_BEAM.md` (the two closed-form beams).

**Scope discipline.** `elements/factory.py`'s dispatch extended from `Bar`-only to all four
Stage-3 element families; `core/model.py` gained the minimal registration `Model` needed to name
a constitutive law and a plane geometry per domain. No new element class, no new top-level
package, no redesign of `Element`'s ABC (confirmed with the project owner: the existing single,
field-agnostic `Element` ABC already covers scalar and vector fields generically - see the
"Element ABC" decision this phase opened with).

---

## 1. Why this was the real remaining gap, not a new abstraction

Every element family already existed and was already verified in isolation: `Bar` (v0.8.0),
`ContinuumElement`/T3 (v0.9.0/v0.12.0), `EulerBernoulliBeam` (v0.10.0), `TimoshenkoBeam`
(v0.11.0), Q4 (v0.13.0). But `elements/factory.py` - the one seam `Model`/`LinearStaticAnalysis`
use to turn a validated `Model` into concrete elements - dispatched `Bar` only, hardcoded to
`IsotropicElasticity`/`"line2"`. Every other family's own verification test constructed its
element directly, bypassing `Model` entirely. Auditing this (via a read-only research pass)
found the true Stage-3 gap wasn't a missing abstraction - it was this one un-generalized factory,
plus a `Model` object model with no place to register a `ConstitutiveModel` or a `PlaneGeometry`
at all (`DomainDefinition` bound only `region -> theory, material[, section]`).

---

## 2. `Model` extensions: the smallest addition that closes the gap

- `Model.add_constitutive(name, constitutive)` / `.constitutives` - mirrors `add_theory`/
  `.theories` exactly. Only `ContinuumElement`-backed domains need one; `Bar`/`EulerBernoulliBeam`/
  `TimoshenkoBeam` are ADR-002 closed-form and never did.
- `DomainDefinition.constitutive: str | None = None` - a new optional field, so every existing
  `DomainDefinition(...)` call in the codebase (all `Bar`/beam domains) is unaffected.
- `DomainGeometry = CrossSection | PlaneGeometry` - `Model._sections`'s value type widened from
  `CrossSection` alone, since a plane-continuum domain's geometry record is `PlaneGeometry`
  (thickness only), not a structural `CrossSection` (area/second-moment/torsion/...). The two
  geometry records both come from SDS 2.2; nothing about either changed.
- `Model.validate()` now also resolves `domain.constitutive` when set and cross-checks the
  constitutive law's `required_properties()` against the domain's material, exactly mirroring the
  existing theory `required_properties()` check.

## 3. `elements/factory.py`: dispatch by theory type, not a registry

Still explicit `isinstance` branches (see the module docstring for why a registry buys nothing
yet): `IsotropicElasticity(dim=1)` -> `Bar`; `IsotropicElasticity(dim=2)` -> `ContinuumElement`
(cell type further dispatches T3 vs. Q4 - a `dict[str, int]` restricted to `{"tri3": 1, "quad4":
1}`, deliberately excluding T6/Q8/anything higher-order per Stage 3's own "minimal classical
element library" scope); `EulerBernoulliBendingTheory` -> `EulerBernoulliBeam`;
`TimoshenkoBeamTheory` -> `TimoshenkoBeam`. Each family gets its own small `_build_*_elements`
helper (single responsibility, easy to extend per family without touching the others).

### A genuine plumbing bug this surfaced, caught before it shipped

Wiring `EulerBernoulliBeam`/`TimoshenkoBeam` through `Model.build_dof_handler()` for the first
time immediately raised `DofMappingError: no DOF for node 0, field 'u', component 'y'`. The root
cause: `Model.field_specs()` always names a field's components positionally
(`core.fields.component_names(n)` - `x`, `y`, `z`, ...), which is correct for
`IsotropicElasticity`'s displacement field (its components genuinely *are* spatial axes in
order) but wrong for a bending theory's 1-component fields `u`/`r`, which by SDS C-2's own
worked example (and two existing, pinned regression tests) are named `u.y`/`r.z` - the *second*
and *third* axis names, not the positional-first default `u.x`/`r.x`. This mismatch was invisible
before this phase because no beam theory had ever been driven through `DofHandler` - every prior
beam test built `global_dofs` by hand.

Fixed additively, not by touching the frozen beam elements (whose `dof_signature()` already
correctly said `u.y`/`r.z`, pinned by existing tests) or by changing `Theory.fields()`'s
contract (which would touch every `Theory` subclass): a new `Theory.field_component_names()`
method, non-abstract with a `{}` default, lets a theory declare explicit component names per
field when the positional default is wrong. `EulerBernoulliBendingTheory`/`TimoshenkoBeamTheory`
override it to return `{"u": ("y",), "r": ("z",)}`; `IsotropicElasticity` and every
`DeclaredTheory` inherit the default unchanged. `Model.field_specs()` consults it, falling back
to the positional name only when a theory doesn't override.

### A second bug, caught by the existing independence-proof test suite itself

Adding `ContinuumElement` to the factory's import list broke five, unrelated-looking existing
tests: `test_module_needs_no_quadrature` (mapping), and four sibling independence proofs for
`operators`/`quadrature`/`shape_functions`/`tensors`. Cause: `nanofem/__init__.py` eagerly
re-exports `analysis.static`, which imports this factory, which - once it imported
`ContinuumElement` at module scope - transitively pulled `numerics.quadrature` into `sys.modules`
on *any* `import nanofem.anything`, including a bare `from nanofem.numerics.mapping import
AffineMapping`. `Bar`/`EulerBernoulliBeam`/`TimoshenkoBeam` never had this problem (confirmed by
reading their own import lists: none imports quadrature). Fixed by deferring `ContinuumElement`/
`PlaneGeometry`/`cell_type_of_name`'s imports to inside `_build_continuum_elements` itself (a
`TYPE_CHECKING`-only import satisfies mypy for the return-type annotation without an eager
runtime import) - the T3/Q4 path still works identically, but paying its own import cost only
when a domain actually needs it.

---

## 4. Verification

`tests/unit/test_elements_factory.py` (new): each family's dispatch branch produces an element
whose `local_stiffness()` matches one built directly (a plumbing proof, not new element-physics
verification - each family's own physics is already verified in its own test file), plus error
paths (`ContinuumElement` domain missing a constitutive law or geometry; a `Bar` domain pointed
at non-`"line2"` cells).

`tests/unit/test_static_t3_plate_analytical.py` (new): the first full `Mesh -> Model ->
LinearStaticAnalysis` solve of a 2-D continuum domain. A two-triangle rectangular plate under the
*consistent* nodal load for a uniform traction (`P/2` at each right-edge node, exact for a linear
edge) reproduces the classical 1-D bar formula `u_x(L) = P L / (E H t)` exactly - `H*t` is exactly
the cross-sectional area a uniaxial bar under the same total force would have, so this is a
genuine global patch test through the full pipeline (assembly, Dirichlet elimination, solve,
reaction recovery), not a restatement of the single-element patch test v0.12.0 already proved.
Reactions are also checked to balance the applied load.

Full regression: `test_static_analytical.py` (the original `Bar` walking-skeleton test) and
`test_model_and_analysis.py` pass unmodified. Full suite: 1251 -> 1261 tests (10 new), full gate
(black/isort/ruff/mypy strict/import-linter 4-kept-0-broken/pytest) green.
