# NanoFEM Euler-Bernoulli Beam Element (v0.10.0)

**Status:** implemented and tested. Companion to `docs/design/WALKING_SKELETON.md` (the same
ADR-002/SDS clause E-5 pattern `Bar` established), `docs/design/ELEMENT_INTEGRATION.md`
(`ContinuumElement`'s composed-path machinery, reused here for the verification oracle rather
than for the element itself), and SDS clauses E-5/E-10.

**Scope discipline.** One new element (`EulerBernoulliBeam`), one new theory/constitutive pair
(`EulerBernoulliBendingTheory`/`EulerBernoulliBendingConstitutive`), 1-D-in-1-D embedding only
(pure bending, no axial-flexural coupling). `Truss2D`/`Frame2D` (structural-suite completion,
already named as later work) own the real direction-cosine transformation and axial+bending
coupling once an element needs to live in a mesh with `d > 1`.

---

## 1. Sign convention (SDS clause E-10)

No normative Euler-Bernoulli sign convention exists anywhere else in the design docs — E-10 only
requires that one be *documented*, not what it must say. This element's convention, stated once
here and in `physics/elasticity/euler_bernoulli.py`'s module docstring:

- Local axis `x` runs node 1 (`x=0`) to node 2 (`x=L`).
- Transverse deflection `w` positive in local `+y`.
- Rotation `θ = dw/dx` positive by the right-hand rule about local `+z` — matching the `r.z` DOF
  component name below.
- Curvature `κ = d²w/dx²`; bending moment `M = EIκ`, sagging positive.

This reproduces the universal textbook stiffness matrix in DOF order `(w1, θ1, w2, θ2)`:

```
K = (EI/L^3) * [[ 12,   6L,  -12,   6L],
                [ 6L,  4L^2, -6L,  2L^2],
                [-12,  -6L,   12,  -6L],
                [ 6L,  2L^2, -6L,  4L^2]]
```

Moment/shear *recovery* from a solved DOF vector (post-processing) is out of scope this phase —
nothing here needs to referee that convention's sign at the recovery level.

---

## 2. DOF/field naming: not invented, taken from the SDS's own example

`Bar` uses field `u` with a single component `x` (`"u.x"`). This element's DOF signature,
`ElementDofSignature((("u.y","r.z"), ("u.y","r.z")))`, is not a fresh naming choice — it is
`NanoFEM_SDS.md`'s **own literal worked example** for C-2 DOF ordering: *"(node₁: u_x, u_y, r_z;
node₂: u_x, …)"*. `EulerBernoulliBendingTheory.fields()` declares `(("u", 1), ("r", 1))` — two
separate fields, not one field with two components — relying on SDS C-2's node-major-then-
field-declaration-order-then-component numbering to naturally produce Hermite's expected
`(w1, θ1, w2, θ2)` order. Multi-field `Theory`s are legal in general (the "single-field only"
restriction in `ContinuumElement` is local to that class, not a `Theory`-level rule).

A useful consequence: a future `Frame2D` combining `Bar`'s `u.x` with this beam's `u.y`/`r.z`
gets the SDS's exact illustrative per-node signature (`u.x, u.y, r.z`) for free, with zero
renaming.

---

## 3. Why the bending constitutive law is a new class, not a reuse

`EulerBernoulliBendingConstitutive`'s `M_per_I = E·κ` has the same functional form as
`IsotropicElasticConstitutive`'s `σ = E·ε` — both are "response = E × generalized-strain,
tangent = [[E]]," with the geometric multiplier (`area` for `Bar`, `I` for this beam) applied by
the *element*, not the constitutive law, in both cases. Reusing `IsotropicElasticConstitutive`
would satisfy "avoid duplicated code" literally, but that class's docstring, error messages, and
test suite all speak specifically about axial strain/stress; repurposing it for curvature/moment
would either mean rewriting frozen, already-verified v0.8.0 code for no functional gain, or
silently overloading its documented meaning. The ~10 duplicated lines are a deliberate, cheap
trade for keeping each physical law's name honest about what it actually models.

---

## 4. The Hermite reference-vs-physical-derivative subtlety

This is the one genuinely non-obvious finding this phase produced, confirmed by direct
computation (not just reasoned about) while building the composed-path verification test.

`HermiteInterpolation`'s DOF functionals include `derivative=(1,)` entries — but that derivative
is with respect to the **reference** coordinate `ξ`, i.e. `dw/dξ`, not the **physical** rotation
`θ = dw/dx` this element's global DOF actually represents. Composing the raw
`HermiteShapeFunctions.hessian()` → `AffineMapping.physical_hessian()` →
`second_gradient_tensor()` pipeline directly against physically-parametrized rotation DOFs
reproduces a stiffness matrix **off by exactly the Jacobian `J = L/2` on every rotation-
associated row/column, and `J²` on the rotation-rotation term** — confirmed numerically before
any test was written, by comparing the naive composed stiffness against the closed form and
observing the discrepancy was *exactly* `J`/`J²`, not noise.

The fix, `physics/elasticity/euler_bernoulli._reference_derivative_scale`: scale each shape
function's curvature contribution by `J ** derivative_order` (using each DOF's own
`.derivative` order, `0` for value-type, `1` for derivative-type) before assembling `K`. This
converts the curvature row from "per unit reference-derivative DOF" to "per unit physical-
rotation DOF." After this correction, the composed path matches the closed form to `rtol=1e-9`
for every tested case.

This is general to any Hermite/C1 family used with physically-parametrized derivative DOFs, not
specific to bending — but it has exactly one consumer today, so it lives in
`physics/elasticity/euler_bernoulli.py` rather than `numerics/interpolation` (no second consumer
yet to justify moving it there, matching this project's "don't build ahead of a real consumer"
discipline).

---

## 5. Verification

- `test_beam_eb_analytical.py`: closed-form stiffness against an independently hand-derived
  matrix; DOF signature; transformation matrix; contribution emission; input validation;
  symmetry and rigid-body null-space checks (`K @ [1,0,1,0] = 0` for rigid translation,
  `K @ [0,1,L,1] = 0` for rigid rotation about node 1 — **not** `[0,1,0,1]`, since a rigid
  rotation carries node 2 through `w2 = L·θ`, a detail this session's own scratch verification
  caught before it became a wrong test).
- `test_elasticity_euler_bernoulli_analytical.py`: theory/constitutive declarations, the
  moment-curvature law, tangent-consistency (SDS Section 5 condition 2), null response.
- `test_beam_eb_verification.py`: the ADR-002/E-5 composed-path equivalence proof described in
  Section 4 above, parametrized over two independent `(L, E, I)` cases.
- `test_beam_eb_cantilever_benchmark.py`: a single-element tip-loaded cantilever, hand-
  partitioned (no assembler needed), matching `w = PL³/(3EI)` and `θ = PL²/(2EI)` to `rtol=1e-9`
  — exact for a single cubic-Hermite element under a tip point load, and a free preview of the
  later verification-suite's "cantilever beam" benchmark item.
