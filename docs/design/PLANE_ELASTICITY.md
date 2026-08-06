# NanoFEM Plane Stress/Strain Elasticity + T3 Triangle (v0.12.0)

**Status:** implemented and tested. Companion to `docs/design/ELEMENT_INTEGRATION.md`
(`ContinuumElement`, whose first real 2-D exercise this is) and `docs/design/BEAM_ELEMENT.md`/
`TIMOSHENKO_BEAM.md` (the Theory/ConstitutiveModel split precedent this phase reuses).

**Scope discipline.** `IsotropicElasticity`'s existing `dim` parameter extended to accept `2`
(kinematics only); two new constitutive classes, `PlaneStressConstitutive`/
`PlaneStrainConstitutive`; T3 delivered as a verified use case of the existing
`ContinuumElement`, not a new element class. Q4 explicitly deferred — see Section 4.

---

## 1. Why this is a Theory extension, not a new Theory

`IsotropicElasticity` declares *kinematics*: a displacement field of `dim` components, the
`symmetric_gradient`/`voigt_map` strain measure. That kinematic description is **identical**
whether the resulting stress-strain law is plane stress, plane strain, or (later) an Eringen
nonlocal reduction — only the constitutive law differs. This is precisely the modularity the
project's own current directive asks for: *"every component must be modular enough that the
constitutive law can later be replaced... without modifying the solver architecture."*

So the fix was to extend `IsotropicElasticity.__init__`'s existing `dim` check from
`dim != 1` to `dim not in (1, 2)` — its own docstring already named this exact gap
("`dim` is accepted here only to name that gap honestly") — rather than create a parallel
`PlaneElasticityTheory` class. `dim=2`'s `fields()` returns `(("u", 2),)`;
`required_properties()` returns `("E", "nu")` (vs. `("E",)` for `dim=1`, since the plane laws
need `nu`, the axial law doesn't). `IsotropicElasticConstitutive` (the dim=1 axial law) is
untouched — it stays a separate, dim=1-only class, exactly as before.

`symmetric_gradient_matrix` needed **zero changes** for `dim=2`: it already reads
`VOIGT_ORDER[dim]` from `numerics/tensors/conventions.py`, which already has a `dim=2` entry
(`((0,0), (1,1), (0,1))` — `eps_xx, eps_yy, gamma_xy`, the standard kinematic-Voigt engineering-
shear convention). This was confirmed, not assumed, by reading the operator's source before
writing any new code.

---

## 2. The two constitutive laws

**Plane stress** (thin plate, free to strain out-of-plane, `sigma_zz = 0`):
```
D = E/(1-nu^2) * [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]]
```

**Plane strain** (thick/constrained body, no out-of-plane strain, `eps_zz = 0`):
```
D = E/((1+nu)(1-2nu)) * [[1-nu, nu, 0], [nu, 1-nu, 0], [0, 0, (1-2nu)/2]]
```

Both act on the same generalized strain `[eps_xx, eps_yy, gamma_xy]` `symmetric_gradient_matrix`
already produces. They are deliberately **separate classes**, not one class parametrized by a
stress/strain flag — the same reasoning already recorded for
`EulerBernoulliBendingConstitutive` vs. `IsotropicElasticConstitutive` (dev note N-55): plane
stress and plane strain are physically distinct reductions with different formulas, and keeping
each class's name/docstring honest about exactly which one it implements is worth the ~15 lines
of structural similarity between them.

`geometry/plane.py`'s `PlaneGeometry(thickness)` — which **already existed**, not something
built this phase — supplies the thickness multiplier through `ContinuumElement`'s existing
`section_measure` parameter, exactly as `Bar`'s `area`/`EulerBernoulliBeam`'s `second_moment`
are applied: the constitutive law never sees thickness, only the per-unit-thickness stress.

---

## 3. T3: a verified use case, not a new class

A 3-node triangle with linear (order-1) shape functions has an **always-affine** geometric map
— 3 vertices exactly determine the 2×2 linear part plus translation of an affine map in 2-D, a
fact about triangles, not an approximation. `ContinuumElement` (built v0.9.0) is already
hardcoded to `AffineMapping`, so `ContinuumElement(cell_type=TRIANGLE, interpolation_order=1,
theory=IsotropicElasticity(dim=2), constitutive=PlaneStressConstitutive(), ...)` **is** T3,
with zero new element code. Building a separate `Triangle3`/`T3` class would duplicate exactly
what `ContinuumElement` already provides generically — precisely the "avoid unnecessary
abstractions" principle this phase's own directive states.

### Verification: the constant-strain patch test

The classical first check for any new continuum element. Since stress/strain recovery doesn't
exist yet (a later phase), verification is via strain energy: for a prescribed linear
(constant-strain) nodal displacement field, `0.5 u^T K u` must equal the analytical
`0.5 (eps^T D eps) * Area * thickness` to floating-point precision. Verified for both a
normal-strain and a pure-shear generalized strain, for both plane stress and plane strain, on
two different triangles (an axis-aligned unit right triangle and an arbitrary scalene/rotated
one, to rule out a right-triangle-specific coincidence) — all match to `rtol=1e-9`.

### A tolerance subtlety: rigid-body checks need a scale-relative tolerance here

`Bar`/`EulerBernoulliBeam`/`TimoshenkoBeam`'s rigid-body null-space tests used a small fixed
`atol` (`1e-6`). For a plane-stress T3 with `E ~ 2e11` Pa, `K`'s own entries are of order `1e11`
— floating-point noise on a mathematically-exact-zero product is itself of order
`1e11 * 1e-16 ~ 1e-5`, which a fixed `atol=1e-6` would fail spuriously. This phase's rigid-body
tests instead use `atol = 1e-9 * K.max()`, scaled to the matrix's own magnitude — the correct
general form, which the earlier, smaller-magnitude beam matrices happened not to need.

---

## 4. Q4: deferred here, delivered in v0.13.0

A general (non-parallelogram) quadrilateral's bilinear map is **not** affine — confirmed by
reading `AffineMapping`'s own fit-residual check (`numerics/mapping/affine.py`), which
correctly *raises* `NonAffineError` (not a silently wrong answer) for a real Q4, naming
`IsoparametricMapping` as the fix. `ContinuumElement` did not support `IsoparametricMapping`
this phase — adding that support was a genuine architecture extension (not just a physics
extension like this phase's plane elasticity), deferred to its own increment, matching the
established "one thing at a time, fully verified" sequencing from `Bar` → `EulerBernoulliBeam`
→ `TimoshenkoBeam`. See `docs/design/Q4_QUADRILATERAL.md` for the v0.13.0 delivery.

---

## 5. Verification summary

`test_elasticity_isotropic_analytical.py` (edited): `IsotropicElasticity(dim=2)` declarations;
`dim=2` no longer raises, only `dim=3` does; `IsotropicElasticConstitutive` still dim=1-only,
unaffected.

`test_elasticity_plane_analytical.py` (new): both laws' `D` matrices against independently
hand-built matrices across several `(E, nu)` pairs; the two laws are confirmed genuinely
different (not the same matrix in disguise); tangent-vs-finite-difference per Voigt component
(SDS Section 5 condition 2); tangent symmetry (condition 1); null response (condition 3).

`test_t3_triangle_verification.py` (new): the constant-strain patch test (Section 3); symmetry
and rigid-body null space with the scale-relative tolerance; DOF signature
(`("u.x","u.y")` per node, matching `core.fields.component_names(2)`).
