# NanoFEM Eringen Differential Nonlocal Elasticity: the Fixed-Fixed Nonlocal Bar (v0.20.0)

**Status:** implemented and tested. Stage 4's first target, confirmed with the project owner
(nonlocal axial bar, 1-D - see the "Stage 4 start" decision). Companion to
`docs/design/WALKING_SKELETON.md` (`Bar`, unmodified by this phase) and
`docs/dev/notes.md` Phase 20 notes.

**Scope discipline.** No new element, no new `Theory`, no new `ConstitutiveModel`. `Bar` and
`IsotropicElasticity(dim=1)` are exactly right and untouched. The only new code is a load
contribution: `NonlocalAxialLoad`/`NonlocalAxialLoadProvider`.

---

## 1. Why this needed a careful derivation before any code

Eringen's differential nonlocal elasticity is a genuinely different kind of law than anything
built so far in this codebase: `sigma(x) - mu*sigma''(x) = E*epsilon(x)`, `mu = (e0*a)^2` - an
*implicit*, PDE-defined relation between local strain and nonlocal stress, not a pointwise
algebraic map like every `ConstitutiveModel` built through Stage 3. The literature around this
model is also unusually treacherous: it contains a well-documented, easy-to-miss property
(sometimes called the "Peddieson paradox," after Peddieson, Buchanan & McNitt, *Application of
nonlocal continuum models to nanotechnology*, 2003) where the differential model's displacement
field is **identical** to the classical local solution for a bar/beam under a concentrated end
load or a spatially uniform distributed load - the nonlocal parameter simply drops out. A
benchmark built the "obvious" way (a cantilever under a tip point load, mirroring every prior
Stage-3 beam benchmark) would silently prove nothing about nonlocal correctness at all.

Given that trap, every step below was verified independently - symbolically two different ways,
then numerically against the real NanoFEM stack - **before** any production code was written,
continuing the discipline N-53/N-56 established (a memorized or freshly-derived formula gets
checked before it ships, not after).

---

## 2. The strong-form derivation

Setup: bar of length `L`, cross-section area `A`, Young's modulus `E`, axial displacement `u(x)`,
distributed axial load `q(x)` (force per unit length). Kinematics stay local: `epsilon = u'(x)`.
Eringen's nonlocal constitutive law and axial equilibrium:

```
sigma(x) - mu*sigma''(x) = E*u'(x)      (constitutive, mu = (e0*a)^2 >= 0)
A*sigma'(x) = -q(x)                     (equilibrium)
```

Eliminating `sigma` (differentiate the constitutive relation, substitute `sigma''` and `sigma'`
from equilibrium's derivatives) gives a single 2nd-order ODE in `u` alone:

```
A*E*u''(x) = mu*q''(x) - q(x)
```

This is the source of the Peddieson paradox in closed form: **only `q''`** carries the nonlocal
parameter into the equation for `u`. A uniform load (`q'' = 0`) or a point load (`q = 0` in the
interior) leaves `u` exactly classical.

**Verified two independent ways** (`eringen_bar_derivation.py`, run interactively, not checked
in - a derivation script, not a test):
1. Solving the eliminated 2nd-order ODE directly with `sympy.dsolve`.
2. Solving the *original coupled system* (`sigma` from equilibrium's own first-order ODE,
   substituted into the constitutive relation, then integrated once more for `u`) independently,
   carrying its own free integration constant through to the boundary conditions.

Both routes agree exactly once the two boundary conditions (`u(0)=u(L)=0`) are applied - the
extra integration constant in route 2 (from `sigma`'s own arbitrary additive constant) is forced
to zero by the two displacement boundary conditions, so this particular problem (both ends fixed
in displacement) has no leftover "nonlocal boundary condition" ambiguity - a real and separately
documented complication for cantilever/free-end nonlocal problems, sidestepped here by choosing
a fixed-fixed bar for the first benchmark.

### The chosen benchmark: `q(x) = q0*sin(pi*x/L)`, both ends fixed

Chosen because `sin` is an eigenfunction of `d^2/dx^2`, giving a clean closed form and a genuine,
sizable nonlocal effect:

```
u(x) = (q0 / (pi^2 * A * E)) * (L^2 + pi^2*mu) * sin(pi*x/L)
     = u_classical(x) * [1 + pi^2*mu/L^2]
```

The nonlocal parameter enters as a clean multiplicative amplification on the classical solution
- larger `mu` gives a softer (larger-displacement) response, the expected direction for this
model.

---

## 3. The weak form: no stiffness change, a load correction only

Integrating the strong-form ODE against a test function `delta_u` (vanishing at both Dirichlet
boundaries) and integrating by parts once - moving one derivative off `q''` and onto
`delta_u'`, so only `q'` is ever needed, not `q''` pointwise - gives:

```
integral( A*E*u'*delta_u' ) dx  =  integral( q*delta_u ) dx  +  mu * integral( q'*delta_u' ) dx
```

The left-hand side is **exactly** the classical bar's bilinear form - `Bar`'s own closed-form
stiffness is already correct, unmodified. The right-hand side is the classical consistent load
plus one new term: `mu * integral(dN_a/dx * q'(x)) dx`, a "gradient body force" needing shape
function *derivatives* (not just values) contracted against the load's own derivative. This is
genuinely novel to the codebase - every prior load (`NodalLoad`, `TractionLoad`, `BodyForce`)
integrates only shape function *values*.

**Verified numerically** against the real `LagrangeInterpolation`/`AffineMapping`/quadrature
stack (`eringen_bar_weak_form_check.py`), two ways:
- `q(x)` sampled exactly at quadrature points (an idealized check of the weak form alone):
  matches the closed form to floating-point noise even at 2 elements - the classical
  superconvergence property of 1-D linear-element Galerkin FEM for a self-adjoint problem.
- `q(x)` sampled at nodes and isoparametrically interpolated (the production-realistic case,
  since a frozen dataclass cannot hold a callable): clean **O(h^2)** convergence, `mu=0`
  reduces exactly to the classical solution, and the nonlocal correction is resolved roughly
  30x above discretization error already at 16 elements.

---

## 4. `NonlocalAxialLoad`/`NonlocalAxialLoadProvider`: what actually got built

`constraints/loads.py` gains `NonlocalAxialLoad(region, field, nodal_intensity, nonlocal_parameter)`
- `nodal_intensity` is `q` sampled at every mesh node (dense, one entry per global node id, not a
callable - matching every other load's frozen-dataclass contract), `nonlocal_parameter` is `mu`.

`constraints/nonlocal_load.py` (new) hosts `NonlocalAxialLoadProvider`, a CELL FORCE
`ContributionProvider` mirroring `ContinuumBodyForceProvider` (cell integral) and
`TractionLoadProvider` (its deferred-import discipline, dev note N-66: `numerics.interpolation`/
`mapping`/`quadrature` are imported inside the functions that need them, not at module scope,
since this module is reached from `nanofem/__init__.py`'s eager top-level re-exports on every
`import nanofem.anything`). Per `line2` cell: `q_h(x)` and `q_h'(x)` are read from the same
linear shape functions/physical gradients already used for the classical term, so both integrals
share one quadrature loop.

`analysis/static.py`'s load-case dispatch gained one more `isinstance` branch, alongside
`NodalLoad`/`TractionLoad` - no other change.

---

## 5. Why this doesn't need `helmholtz_matrix` or a new `Theory`/`ConstitutiveModel` (yet)

The project's Eringen-readiness directive frames the constitutive swap as
`LinearElasticMaterial -> ConstitutiveModel -> EringenDifferentialMaterial`, and the original
plan for this increment assumed exactly that shape, built around the existing `helmholtz_matrix`
operator (`numerics/operators/helmholtz.py`, shipped v0.7.0, unused since). The derivation above
shows that assumption was only half right: for a **statically determinate 1-D bar problem**,
`sigma` can be eliminated in closed form, collapsing the nonlocal effect entirely into the load
vector - no independent stress field, no Helmholtz PDE to solve, no constitutive-law swap needed
at all for *this* benchmark. `Bar`/`IsotropicElasticity(dim=1)`/`IsotropicElasticConstitutive`
are exactly correct, untouched.

This elimination is specific to problems where `sigma` can be solved for directly from
equilibrium's own first-order ODE (true for a 1-D bar with known boundary tractions/forces, not
true for a general 2-D/3-D continuum, where equilibrium is a *divergence* condition on a tensor
field with no closed-form inversion in general). A general Eringen-differential continuum element
(2-D/3-D, `ContinuumElement`-based) genuinely will need the mixed/mixed-adjacent treatment the
original plan anticipated - an independent nonlocal-stress field solved via `helmholtz_matrix`'s
weak form, coupled to the displacement field. That is real, separate future work; this increment
deliberately stayed at the smallest problem that proves the overall physics and pipeline
end-to-end first, matching the "one thing at a time, fully verified" sequencing every prior
Stage-3 element followed.

---

## 6. Verification

`test_nonlocal_axial_load_provider.py` (6 tests): the provider's two integrals against a
hand-derived closed form for a single element with linearly-varying nodal intensity; the
`mu=0` reduction; the uniform-load (`q'=0`) zero-nonlocal-correction case; input validation
(nodal-intensity length mismatch, multi-component field rejection, negative `mu` rejection).

`test_nonlocal_bar_benchmark.py` (8 tests, parametrized): the full `Mesh -> Model ->
LinearStaticAnalysis` pipeline solving the fixed-fixed sinusoidally-loaded bar, checked against
the closed form at 4/8/16/32 elements (within 5% by 4 elements); a monotonic, roughly-`O(h^2)`
mesh-convergence sweep from 2 to 32 elements; the `mu=0` reduction through the full pipeline;
confirmation the nonlocal correction is resolved well above discretization error at a moderate
mesh; and the Peddieson-paradox uniform-load case (local and nonlocal solutions identical to
`rtol=1e-10`).
