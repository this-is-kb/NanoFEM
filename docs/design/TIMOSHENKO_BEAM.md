# NanoFEM Timoshenko Beam Element (v0.11.0)

**Status:** implemented and tested. Companion to `docs/design/BEAM_ELEMENT.md` (the
Euler-Bernoulli beam this element is contrasted against throughout) and SDS clauses
E-2/E-3/E-5/E-10.

**Scope discipline.** One new element (`TimoshenkoBeam`), one new theory/constitutive pair
(`TimoshenkoBeamTheory`/`TimoshenkoBeamConstitutive`), 1-D-in-1-D embedding only (pure
bending+shear, no axial coupling — matching `Bar`/`EulerBernoulliBeam`'s own scoping).

---

## 1. Why Timoshenko needs only `Continuity.C0`, not `EulerBernoulliBeam`'s `C1`

Euler-Bernoulli theory assumes plane sections stay perpendicular to the deformed beam axis,
so rotation is not independent: `theta = dw/dx`, and curvature `kappa = d(theta)/dx =
d^2w/dx^2` is a **second** derivative of `w` — which is why that element needs Hermite/C1
interpolation (`docs/design/BEAM_ELEMENT.md`).

Timoshenko theory drops that assumption: `w(x)` and `theta(x)` are **independent** fields,
and a shear strain `gamma = dw/dx - theta` appears. Curvature is now `kappa = d(theta)/dx` —
a **first** derivative of the independent rotation field, not a second derivative of `w`.
This is the precise kinematic reason `TimoshenkoBeamTheory.operators_used()` is
`("gradient",)` (order 1, forcing only `Continuity.C0`) rather than `("second_gradient",)`
— no Hermite family is needed; plain `LagrangeInterpolation(LINE, order=1)` suffices for
both `w` and `theta`.

---

## 2. A wrong assumption, caught before any code was written

The natural first instinct when planning this element was to mirror `Bar`/
`EulerBernoulliBeam` exactly: use the widely-known "exact" Timoshenko stiffness matrix,
parametrized by `Phi = 12EI/(GA_sL^2)`, found in most textbooks and commercial FE codes.
That formula is derived by solving the governing ODEs exactly, which requires a
cubic-`w`/quadratic-`theta` shape-function pair distinct from plain linear Lagrange — a
shape-function family this codebase's `Interpolation` framework does not have, and building
one would be exactly the kind of new abstraction the "classical FEM pipeline" development
phase is meant to avoid.

Before writing any production code, this assumption was tested: a from-scratch composed-path
calculation using plain `LagrangeInterpolation(LINE, order=1)` with selective reduced
integration (SDS clause E-3's actual, named remedy) was compared against the "exact
`Phi`-parametrized" formula. **They did not match** — a clean, exact 0.75 ratio discrepancy
at the shear-rigid limit, not numerical noise. Investigation traced this to a real fact: the
"exact" formula and the "SRI-linear" formula are two genuinely different, independently valid
element formulations; SDS E-3 names SRI specifically, not the ODE-derived alternative, so the
composed-path oracle should target — and does — the SRI formulation, and the closed form
built into `TimoshenkoBeam.local_stiffness()` is the SRI result, not the ODE-exact one.

This is recorded here (and in `docs/dev/notes.md` N-56) because it is exactly the kind of
mistake ADR-002/SDS E-5's "declare equivalence to the composed path... verification tests
enforce it" requirement exists to catch — caught here before the wrong formula ever reached
production code, not after.

---

## 3. The selective-reduced-integration (SRI) derivation

Per SDS clause E-3: *"Timoshenko declares selective-reduced integration of the shear term"* —
full (2-point Gauss) integration of the bending term, reduced (1-point Gauss) integration of
the shear term.

For a 2-node linear element with node 1 at `x=0`, node 2 at `x=L`:
- `theta(x)` is linear, so `kappa = d(theta)/dx` is **constant**. Full 2-point Gauss
  integration of the bending energy `EI*kappa^2` over `[0,L]` is exact (and a 1-point rule
  would give the same result, since the integrand is already constant).
- `w(x)` is linear, so `dw/dx` is constant; `theta(x)` varies linearly. The 1-point reduced
  rule evaluates `gamma = dw/dx - theta` at the element midpoint, where
  `theta(mid) = (theta1 + theta2)/2`.

Assembling both terms (`I`, `A_s` applied afterward as element-layer multipliers — exactly as
`Bar` applies `area`, except here there are **two** independent multipliers, one per
generalized-strain component, since bending and shear are physically distinct quantities)
gives the closed form implemented in `TimoshenkoBeam.local_stiffness()`:

```
K = [[ G*As/L,         G*As/2,          -G*As/L,        G*As/2         ],
     [ G*As/2,         E*I/L+G*As*L/4,  -G*As/2,        -E*I/L+G*As*L/4],
     [-G*As/L,        -G*As/2,           G*As/L,        -G*As/2        ],
     [ G*As/2,        -E*I/L+G*As*L/4,  -G*As/2,         E*I/L+G*As*L/4]]
```
DOF order `(w1, theta1, w2, theta2)`, same convention as `EulerBernoulliBeam`: `w` positive
local `+y`, `theta` positive right-hand-rule about local `+z` (matching the `r.z` DOF
component name), `M = EI*kappa`, `V = GA_s*gamma`.

---

## 4. Verification: composed-path equivalence, and mesh convergence instead of single-element exactness

`test_timoshenko_beam_verification.py::test_composed_path_matches_closed_form` builds the SRI
pipeline from scratch (`LagrangeInterpolation` + `gradient_matrix` + two separate quadrature
rules, `order=2` for bending and `order=0` for shear) and matches the closed form to
`rtol=1e-9` — the ADR-002/E-5 equivalence proof, confirmed for two independent `(L,E,G,I,A_s)`
cases.

**A single `TimoshenkoBeam` element is not exact for a cantilever** — unlike `Bar`/
`EulerBernoulliBeam`, both exact for one element under a tip load. This was verified directly:
chaining N elements into a cantilever and refining `N = 1, 2, 4, 8, 16, 32` converges
monotonically toward the classical closed-form tip deflection
`delta = PL^3/(3EI) + PL/(GA_s)`:

| N elements | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| ratio to exact | 0.751 | 0.938 | 0.984 | 0.996 | 0.999 | 0.9998 |

This is the correct, textbook-expected, non-locking convergence behavior of an SRI Timoshenko
element — `test_timoshenko_beam_verification.py::test_mesh_convergence_toward_exact_
cantilever_solution` checks exactly this (monotonic convergence, single-element ratio below
0.8, 32-element ratio within 0.1%) as a permanent regression test, and
`test_timoshenko_beam_cantilever_benchmark.py` provides an 8-element benchmark at a
correspondingly loose (`rtol=5e-3`) tolerance rather than claiming single-element exactness.

---

## 5. Verification summary

`test_timoshenko_beam_analytical.py`: closed-form stiffness against an independent hand
formula; DOF signature (`u.y`/`r.z`, same convention as `EulerBernoulliBeam`); identity/
orthonormal transformation matrix; STIFFNESS-only contribution emission; input validation;
symmetry and rigid-body null-space checks (`K@[1,0,1,0]=0` translation, `K@[0,1,L,1]=0`
rotation about node 1 — the same non-obvious null vector as `EulerBernoulliBeam`'s, since node
2 sweeps through `w2 = L*theta`).

`test_elasticity_timoshenko_analytical.py`: theory declarations including `Continuity.C0`
(Section 1 above); constitutive declarations (`response_components() == 2`); the diagonal
`[E*kappa, G*gamma]` law with exactly-zero off-diagonal tangent terms; tangent-vs-finite-
difference per component (SDS Section 5 condition 2); null response.
