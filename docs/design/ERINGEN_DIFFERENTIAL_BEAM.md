# NanoFEM Eringen Differential Nonlocal Elasticity: the Nonlocal Euler-Bernoulli Beam (v0.23.0)

**Status:** implemented and tested. Stage 4's third increment - the "nanobeam" target Step 9 of
the governing directive names explicitly, and the richest area of the published nonlocal
elasticity literature (Peddieson, Reddy, Wang and many others have closed-form nonlocal
Euler-Bernoulli beam results). Mirrors v0.20.0's nonlocal bar increment one derivative order up.

**Scope discipline.** No change to `EulerBernoulliBeam`/`EulerBernoulliBendingTheory` - the
nonlocal effect is entirely a load correction, exactly as it was for the bar. The only new code
is `NonlocalTransverseLoad`/`NonlocalTransverseLoadProvider`.

---

## 1. The derivation

Setup: a beam of length `L`, flexural rigidity `EI`, transverse deflection `w(x)`, distributed
transverse load `q(x)` (force per unit length, same `+y` direction as positive `w`). This
codebase's own sign convention (`physics/elasticity/euler_bernoulli.py`): curvature
`kappa = w''` (no minus sign), `M = EI*kappa`.

**Equilibrium's sign, derived from the existing weak form, not assumed.** Rather than trust an
independently-recalled shear/moment convention (multiple, mutually inconsistent conventions
exist across textbooks), the correct sign was derived from the classical weak form this
codebase's own `K = EI/L^3[...]` matrix already implements:

```
integral( M * delta_w'' ) dx = integral( q * delta_w ) dx        (virtual work, as-built)
```

Integrating the left side by parts twice (dropping boundary terms) gives the strong form
`M''(x) = q(x)` - this is the sign consistent with `kappa = w''`, `M = EI*kappa`, and this
codebase's own choice of `q`'s direction; a **first attempt using an independently-assumed
`M'' = -q` gave a wrong-signed intermediate result**, corrected before writing this derivation
down, by re-deriving equilibrium's sign from the existing, already-verified weak form itself
rather than from memory.

**Eringen's nonlocal relation:** `M*(x) - mu*M*''(x) = EI*w''(x)`, `mu = (e0*a)^2`. Eliminating
`M*` (using `M*'' = q(x)`, which holds regardless of nonlocality since it is a pure statics
statement):

```
EI*w''''(x) = q(x) - mu*q''(x)
```

**Verified two independent symbolic ways** (mirroring the bar's own two-route check): solving
this eliminated 4th-order ODE directly with a sinusoidal-load ansatz, and separately solving the
original coupled `(M*, w)` system by direct integration from equilibrium alone. Both give,
for a simply-supported beam (`w(0)=w(L)=0`, moment-free ends) under `q(x) = q0*sin(pi*x/L)`:

```
w(x) = (q0*L^4 / (pi^4*EI)) * (1 + mu*(pi/L)^2) * sin(pi*x/L)
```

All four integration constants come out exactly zero in both routes - no leftover "nonlocal
boundary condition" ambiguity for this problem (both ends fixed in deflection and moment-free),
the same clean result the fixed-fixed bar had.

---

## 2. The weak form, and a genuinely new subtlety it exposed

Integrating `EI*w'''' = q - mu*q''` against a test function `delta_w` and integrating by parts
(moving one derivative off `q''` onto `delta_w'`, the boundary term vanishing for a Dirichlet
`delta_w=0` at both simply-supported ends):

```
integral( EI*w''*delta_w'' ) dx = integral( q*delta_w ) dx + mu*integral( q'*delta_w' ) dx
```

The left side is exactly the classical `EulerBernoulliBeam` bilinear form - unmodified. The
right side is the classical consistent load plus `mu*integral(dN_a/dx * q'(x)) dx`, needing
shape function *first* derivatives (`delta_w'`) - not previously needed by this element (only
second derivatives, for curvature).

**A genuinely new bug, caught by numerical verification against the real Hermite stack.** The
first numerical check (mirroring the bar's `mu=0` sanity check) gave a residual of `~6.7e-5`
instead of the expected near-machine-precision agreement - unlike the bar, where linear-element
nodal superconvergence gave exact agreement immediately. Cubic Hermite beam elements have an
*analogous* classical superconvergence property (nodally exact `w` and `theta` for any smooth
load, under the consistent Galerkin load), so a non-trivial residual at `mu=0` was a real red
flag, not an expected approximation. The cause: `_reference_derivative_scale` (N-53) had only
ever been applied to the *curvature* B-matrix (a derivative quantity); the shape function
*values* themselves also need it. A Hermite basis's "derivative" DOFs are natively `dw/dxi`
(reference), so interpolation is `w(x) = sum_a N_a(x) d_a_ref`, and since
`d_a_ref = scale_a * d_a_physical`, the shape function effectively multiplying a *physical* DOF
is `N_a(x)*scale_a` everywhere a shape function is used - including a plain load-vector
integral, not only the bilinear stiffness form N-53 originally fixed. Applying the same scale to
`values` (not only to derivatives) took the idealized (`q` sampled exactly at quadrature points)
check from a non-converging residual to `2.2e-14` at 16 elements, and the production-realistic
(nodally-interpolated `q`) check to clean `O(h^2)` convergence.

---

## 3. `NonlocalTransverseLoad`/`NonlocalTransverseLoadProvider`

`constraints/loads.py` gains `NonlocalTransverseLoad(region, field, nodal_intensity,
nonlocal_parameter)`, mirroring `NonlocalAxialLoad`'s exact shape (`nonlocal_parameter = mu`
directly on the load object, matching the bar's existing convention for structural-member
load corrections - a different, independently-precedented choice from the 2-D continuum work's
material-property `e0a`, and that is fine: they are different load-vs-material design points,
not required to match).

`NonlocalTransverseLoadProvider` (`constraints/nonlocal_load.py`) does **not** take a
`field_components` argument, unlike every other provider in this module: a beam's consistent
load vector is conjugate to *both* `u.y` and `r.z` at each node (the classical Hermite
consistent-load pattern), never a single field's own component list - DOF resolution hardcodes
`("u","y")`/`("r","z")` exactly as `elements/factory.py::_bending_global_dofs` already does.

---

## 4. Verification

`test_nonlocal_transverse_load_provider.py` (6 tests): matches an independently-written
quadrature computation (not calling the provider's own code) for both `mu=0` and `mu>0`; the
uniform-load null effect; input validation (nodal-intensity length mismatch, wrong field,
negative `mu`).

`test_nonlocal_beam_benchmark.py` (7 tests): the full `Mesh -> Model -> LinearStaticAnalysis`
pipeline solving the simply-supported sinusoidally-loaded beam, checked against the closed form
at 8/16/32 elements; monotonic, roughly-`O(h^2)` mesh convergence; the `mu=0` reduction through
the full pipeline; confirmation the nonlocal correction is resolved well above discretization
error; the uniform-load Peddieson-paradox case (local and nonlocal solutions identical to
`rtol=1e-9`).

Full gate: black/isort/ruff/mypy strict/import-linter (4 kept, 0 broken)/pytest, all green.
