# NanoFEM Q4 Quadrilateral via IsoparametricMapping (v0.13.0)

**Status:** implemented and tested. Companion to `docs/design/ELEMENT_INTEGRATION.md`
(`ContinuumElement`) and `docs/design/PLANE_ELASTICITY.md` (T3, plane stress/strain — this
phase's kinematics/constitutive law, unchanged) and `docs/design/GEOMETRIC_MAPPING.md`
(`IsoparametricMapping`, built and verified in v0.5.0, unused by any element until now).

**Scope discipline.** One change to `ContinuumElement`'s mapping construction (a fallback, not a
rewrite); no new element class, no new physics. Q4 is delivered exactly as T3 was — as a verified
use case of the existing `ContinuumElement`.

---

## 1. Why Q4 needed real architecture work and T3 didn't

A 3-node triangle's affine map is exactly determined by 3 vertices (a fact about simplices).
A 4-node quadrilateral's bilinear map is **not**, in general, affine: 4 corners impose 8
conditions on the 6 unknowns of `x = A xi + b`, so a solution exists only when the corners
satisfy the parallelogram condition `x2 = x1 + x3 - x0`. `AffineMapping`'s own construction
(`numerics/mapping/affine.py`) proves this rather than asserting it: it solves the fit by least
squares and then checks the residual is exactly zero, raising `NonAffineError` — naming
`IsoparametricMapping` as the fix — the moment it isn't. This was read and confirmed (not
assumed) while planning v0.12.0's T3 delivery, which is why Q4 was deferred rather than shipped
alongside it.

`IsoparametricMapping` (`numerics/mapping/isoparametric.py`) already existed, built and verified
in v0.5.0: `x(xi) = sum_a N_a(xi) x_a`, the geometry basis contracted against physical node
coordinates, with Newton inversion and the curvature-corrected physical Hessian for the
non-affine case. It had simply never been used by any element — `ContinuumElement` was
hardcoded to `AffineMapping` alone since v0.9.0.

---

## 2. The change: a fallback, not a new code path

`ContinuumElement.__init__` (`elements/continuum/continuum.py`) already built a shape-function
basis (`basis = shape_functions(interpolation)`) for the field itself. That same basis *is* a
valid geometry basis — nodal, matching the element's own node coordinates — so no separate
geometry interpolation object is needed. The mapping construction became:

```python
try:
    mapping = AffineMapping(cell_type, coords[:num_vertices])
except NonAffineError:
    mapping = IsoparametricMapping(basis, coords)
```

Every quantity `ContinuumElement` reads afterward — `physical_gradient`, `volume_scale` — goes
through `GeometricMapping`'s shared abstract interface, which both concrete mappings implement
identically from the caller's point of view (per-quadrature-point arrays; a constant Jacobian
for the affine case is simply a mapping-internal fact, not a different shape). No other line in
`ContinuumElement` changed. `Bar`, `T3` (a parallelogram-or-simplex geometry always takes the
`AffineMapping` branch, so both are unaffected regressions-wise — confirmed by re-running their
existing test suites unmodified after this change, all still passing).

---

## 3. Verification

**Numerically verified before writing the fallback**, using the real `IsoparametricMapping` and
`symmetric_gradient_matrix` (not mocks) on a genuinely non-parallelogram quadrilateral
(vertices `(0,0), (4,0.5), (3.2,3), (-0.3,2.4)`):
- `AffineMapping` on those vertices does raise `NonAffineError`, confirming the premise.
- A constant-strain patch test (both a normal-strain and a pure-shear case) matches the
  analytical `0.5 (eps^T D eps) Area thickness` to machine precision (`rel err ~1e-16` to
  exactly `0.0`) — the isoparametric consistency property (partition of unity + linear
  completeness of the bilinear basis reproduces any linear displacement field exactly,
  regardless of how distorted the quadrilateral is), not a coincidence.
- Rigid-body translation and rotation residuals are at floating-point-noise scale relative to
  `K`'s own magnitude (`~1e-16` relative, matching T3's finding in v0.12.0), confirming the
  scale-relative tolerance convention applies here too.
- `K` is symmetric.

`tests/unit/test_q4_quadrilateral_verification.py` (new) turns the above into pytest-parametrized
tests, covering **both** the parallelogram case (still the `AffineMapping` path — a regression
guard) and the genuinely non-affine case (the new `IsoparametricMapping` path), for both plane
stress and plane strain: the constant-strain patch test, symmetry + rigid-body translation null
space (scale-relative tolerance), a pinned `NonAffineError` premise check, and the DOF signature
(`("u.x","u.y")` per node, 4 nodes).

`tests/unit/test_continuum_element.py`, `test_t3_triangle_verification.py`, and
`test_bar_verification.py` are unmodified and confirmed still green — the fallback changes
nothing for geometries that were already affine.

---

## 4. What's still out of scope

Rigid-body *rotation* is verified only in this document's pre-implementation numerical check,
not asserted in the checked-in test file — matching T3's own test file, which likewise checks
only translation. `IsoparametricMapping`'s Newton-iteration `inverse_map` and curvature-corrected
Hessian are exercised implicitly (`physical_gradient` calls `inverse_jacobian`, which for a
non-affine map is evaluated per quadrature point) but not independently re-verified here — that
verification is `IsoparametricMapping`'s own job, done in v0.5.0. Higher-order or curved
(sub-/super-parametric) quadrilaterals, and 3-D isoparametric elements (Hex), remain future work.
