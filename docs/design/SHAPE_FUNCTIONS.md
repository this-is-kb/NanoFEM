# NanoFEM Shape Function Library

**Status:** implemented and tested (phase 4). Companion to ARCHITECTURE_v2.md, the SDS,
OBJECT_MODEL.md, REFERENCE_ELEMENTS.md, and INTERPOLATION.md.

**Scope discipline.** Shape functions and their derivatives **in reference coordinates**. There
is no Jacobian, no mapping, no physical derivative, no B-matrix, no quadrature, no integration,
and no assembly. Every point passed to this layer is a point in the reference domain, supplied
by the caller.

---

## 1. Derivation from (K, P, Σ)

Phase 3 established the Ciarlet triple and stopped one step short of the basis. This layer takes
that step.

**The setup.** `K` is the reference domain, `P` a finite-dimensional space spanned by monomials
`{m₁ … mₙ}`, and `Σ = {ℓ₁ … ℓₙ}` a set of linear functionals. A shape function is by definition
an element of `P`, so it is a combination of the spanning monomials:

> `Nᵢ(x) = Σⱼ C[i,j] mⱼ(x)`

**The defining property.** The basis is *dual* to the functionals:

> `ℓₖ(Nᵢ) = δᵢₖ`

**Solving for C.** Expand the duality using linearity of `ℓₖ`:

> `ℓₖ(Nᵢ) = ℓₖ(Σⱼ C[i,j] mⱼ) = Σⱼ C[i,j] ℓₖ(mⱼ) = Σⱼ C[i,j] M[k,j] = (C Mᵀ)[i,k]`

Requiring that to be `δᵢₖ` gives `C Mᵀ = I`, hence:

> **C = M⁻ᵀ**

where `M[k,j] = ℓₖ(mⱼ)` is the generalized Vandermonde phase 3 built and proved invertible. The
entire library is that one line.

**Why the inverse exists.** It is not assumed here — it was *proved* in phase 3 by
`verify_linear_independence()`, which is unisolvence: `M` square and full rank. This is the
payoff of the phase split. The failure mode ("the shape functions don't exist") was eliminated
before the code that could hit it was written. `coefficients` still guards the solve, because a
third-party family could reach this layer without validating; when it does, it gets a named
`UnisolvenceError`, not a `LinAlgError` from inside numpy.

---

## 2. The generalized Vandermonde

`M` is *generalized* because its rows are arbitrary functionals, not just point evaluations:

| Family | Row `k` of `M` | |
|---|---|---|
| Lagrange | `mⱼ(xₖ)` | the classical Vandermonde |
| Hermite | `D^α mⱼ(xₖ)` | rows for derivative functionals |

For the cubic Hermite line, `M` pairs `(u(-1), u′(-1), u(1), u′(1))` with `(1, ξ, ξ², ξ³)`:

```
        1    ξ    ξ²   ξ³
u(-1)   1   -1    1   -1
u'(-1)  0    1   -2    3
u(1)    1    1    1    1
u'(1)   0    1    2    3
```

Inverting the transpose gives exactly the classical basis, which the tests assert to 1.1e-16:

```
H₁ = (2 - 3ξ + ξ³)/4      H₂ = (1 - ξ - ξ² + ξ³)/4
H₃ = (2 + 3ξ - ξ³)/4      H₄ = (-1 - ξ + ξ² + ξ³)/4
```

Filling `M` needs the derivative of a *monomial*, which the phase-3 `DofFunctional` already
computes one entry at a time. This layer adds a vectorized tabulator for whole batches; a test
asserts the two agree to 1e-14 across every element, pinning the arithmetic across the phase
boundary.

---

## 3. Basis construction and reference derivatives

Because `D^α` is linear and `C` is a matrix of constants, **every** derivative is the same code
path:

> `D^α Nᵢ(x) = Σⱼ C[i,j] · D^α mⱼ(x)`

So one primitive, `derivative(points, multi_index)`, tabulates `D^α m` and multiplies by `Cᵀ`.
Everything else composes:

| Method | Multi-indices used | Shape |
|---|---|---|
| `evaluate(points)` | `(0,…,0)` | `(n_pts, n_fun)` |
| `gradient(points)` / `derivatives(points)` | `eₐ` for each axis | `(n_pts, n_fun, dim)` |
| `hessian(points)` | `eₐ + e_b` for each pair | `(n_pts, n_fun, dim, dim)` |
| `tabulate(points, max_derivative=…)` | all of the above, batched | `Tabulation` |
| `interpolate(dof_values, points)` | `(0,…,0)` | the field itself |

The monomial rule is `D^α x^β = ∏_d [β_d!/(β_d−α_d)!] · x_d^(β_d−α_d)`, vanishing when any
`β_d < α_d`. The implementation uses `math.perm`, which returns 0 in exactly that case, and
clamps the residual power so `0 ** -1` never arises.

**Hessians are always available.** Polynomials are smooth. For an order-1 family the Hessian is
identically zero — which is correct, not "inapplicable" — except on the quadrilateral, where
`Q₁` retains the `ξη` term so the cross derivative survives while the diagonal vanishes. Tests
pin both.

**Caching.** Results are memoized per `(points, multi-index)` on the family instance. A repeated
point set — which is precisely what a quadrature rule will later supply — is tabulated once, and
`evaluate(pts) is evaluate(pts)` holds. Cached arrays are read-only, so one consumer cannot
corrupt a batch shared with others; this is the sharing pattern SDS C-8 anticipates.

---

## 4. Verification: which check catches which bug

The suite was designed by asking, for each identity, *what could be wrong that this would
catch* — and the answer forced a real change.

**The trap.** The natural check is "reproduce every monomial": `Σᵢ M[i,j] Nᵢ(x) = mⱼ(x)`. But the
left side expands to `table · Cᵀ · M`, and `Cᵀ M = M⁻¹ M = I`, so it collapses to `table`
*identically*. Comparing it against `table` again passes for **any** tabulation, correct or not.
The check would silently be measuring only the accuracy of the solve.

Two fixes, both applied:

1. `verify_polynomial_reproduction` evaluates its right-hand side with
   `_naive_monomial_values`, a deliberately independent power-product loop with none of the
   factorial or clamping logic. A verification that routes both sides through the same machinery
   proves nothing about that machinery.
2. `verify_derivative_consistency` uses **central finite differences**, not the analytic
   identity. The analytic version is vacuous for derivatives by the same collapse; only an
   independently computed derivative catches an error in the tabulated one.

The resulting coverage:

| Check | Catches | Independent standard |
|---|---|---|
| `verify_kronecker_delta` | wrong `C`; wrong derivative tabulation *at DOF points* | expectation is `I` |
| `verify_partition_of_unity` | wrong `C`; wrong values or gradients | expectation is `1` and `0` |
| `verify_polynomial_reproduction` | wrong `C`; wrong value tabulation | naive power product |
| `verify_derivative_consistency` | wrong derivative tabulation | central finite differences |
| `verify_symmetry` | an asymmetric Hessian | structural (see below) |
| `verify_boundary_restriction` | a basis that breaks conformity | expectation is `0` on the facet |
| `verify_interpolation_exactness` | a bad solve; a broken operator API | `M^T C = I`, `C M^T = I` |

A test makes the FD point concrete: a family whose derivatives are scaled by 1.5 — plausible and
systematic — **passes** Kronecker and reproduction, and is caught only by the finite-difference
check.

**Honest note on `verify_symmetry`.** Mixed partials commute, and with the multi-index
construction both `H[…,a,b]` and `H[…,b,a]` come from the same tabulation, so this is a
structural guard rather than a discovery. It earns its place by pinning the invariant for any
future implementation that composes directional derivatives. The independent confirmation is the
finite-difference Hessian, whose two routes to a mixed partial are computed separately.

**Boundary restriction** is the one that matters physically: shape functions dual to DOFs at
nodes *off* a facet must vanish *on* it, so the trace is determined entirely by the DOFs two
neighbouring cells share, and they agree. Without it, C0 conformity is a hope. A test goes
further and shows the tri6 edge trace *is* the line3 basis in the edge parameter.

---

## 5. Examples

```python
from nanofem.numerics.interpolation import LagrangeInterpolation, shape_functions
from nanofem.numerics.reference.enums import CellType

basis = shape_functions(LagrangeInterpolation(CellType.TRIANGLE, 2))

basis.evaluate([[1/3, 1/3]])      # (1, 6)     values at the centroid
basis.gradient([[1/3, 1/3]])      # (1, 6, 2)  reference gradients
basis.hessian([[1/3, 1/3]])       # (1, 6, 2, 2)
basis.verify()                    # every identity above
```

Verified closed forms (each compared against an independently written reference):

| Element | Basis |
|---|---|
| Line `P₁` | `(1∓ξ)/2` |
| Line `P₂` | `ξ(ξ−1)/2`, `ξ(ξ+1)/2`, `1−ξ²` |
| Line `P₁₋₃` | the Lagrange product formula `∏_{j≠i} (x−xⱼ)/(xᵢ−xⱼ)` |
| Triangle `P₁` | `1−ξ−η`, `ξ`, `η` — the barycentric coordinates |
| Triangle `P₂` | `λ(2λ−1)` at vertices; `4λₐλ_b` on edges |
| Quad `Q₁` | `(1±ξ)(1±η)/4` |
| Quad `Q₁₋₃` | the tensor product `Lₐ(ξ)·L_b(η)` |
| Hermite line | the classical cubics, to 1.1e-16 |
| BFS quad | the tensor product `Hₐ(ξ)·H_b(η)`, cross terms included |

The tensor-product tests deserve a note: the library builds the quad basis by inverting a 2-D
Vandermonde and has **no notion that `Q_k` separates**. That the result factors exactly is a real
check on both sides.

The geometric symmetry tests include a subtle one. Under `ξ → −ξ` the Hermite *value* functions
swap, while the *slope* functions swap **and negate** — the derivative functional picks up the
map's Jacobian sign. That is the chain rule showing up in the basis, and it is the first hint of
what the mapping layer will have to handle for Hermite DOFs on a real mesh.

---

## 6. Design notes

**The construction is family-agnostic.** Nothing in `C = M⁻ᵀ` mentions Lagrange or Hermite; all
family information lives in `Σ`, which phase 3 captured. So `LagrangeShapeFunctions` and
`HermiteShapeFunctions` are thin, and that is the *result*, not an omission: **a new family needs
zero shape-function code.** Serendipity and spectral will get their bases free once their
`Interpolation` exists and is unisolvent. What the concrete classes add is each family's
idiomatic verification — the classical `Nᵢ(xⱼ) = δᵢⱼ` for Lagrange, the value/slope pattern for
Hermite — each a restatement of duality in the form a reader of that literature recognizes.

**The phase-0 seam was filled, not bypassed.** `ShapeFunctionFamily` implements the SDS 2.4
`ShapeFunctions` contract (`cell()`, `continuity()`, `completeness_degree()`, `evaluate()`,
`derivatives()`) rather than declaring a parallel one. The phase-0 stubs `LagrangeLine2` and
`HermiteBeamLine2` are superseded and removed — `LagrangeShapeFunctions` covers them generically,
which is what those placeholders were placeholders for.

One wrinkle worth recording: `ShapeFunctions.cell()` returns a `ReferenceCell`, a mesh-*name*
record. That signature was drafted in phase 0, before phase 2 established `ReferenceElement` as
the actual geometric domain. It is honored for contract compatibility, and
`basis.reference_element` is the semantically correct accessor.

**Dev note N-21 is closed.** Phase 3 recorded that `quad9`, `line4`, `tri10`, and `quad16` were
absent from `REFERENCE_CELLS`, with the rule that entries land with the family that consumes
them. `cell()` is that consumer, so they were registered here. `quad8` remains registered and
unused — it is the serendipity cell, still waiting for its family.

---

## 7. Future extension strategy

**The next layer is mapping.** Everything here is a reference derivative `∂N/∂ξ`. Physical
derivatives need `∂N/∂x = J⁻ᵀ ∂N/∂ξ`, and the Jacobian is the mapping layer's job. The seam is
already the right shape: `gradient()` returns `(n_pts, n_fun, dim)`, exactly what a Jacobian
contracts against, and nothing here needs to change to support it.

**Then quadrature.** Quadrature supplies *points*, and this layer already takes points and caches
per point set — so a rule's tabulation is computed once and shared across every element of a
block, which is the SDS C-8 pattern. Note the ordering constraint this closes from phase 3: the
spectral family needs Gauss–Lobatto–Legendre nodes, which are quadrature points, so
`SpectralInterpolation` still waits on that layer. When it arrives, it needs no code here.

**Higher derivatives.** The multi-index primitive already accepts any order; `hessian()` is a
convenience wrapper over `(1,1)`-type indices. Strain-gradient elasticity (roadmap phase 5) needs
third derivatives, which is `derivative(points, alpha)` with `sum(alpha) == 3` — available today,
with no new machinery.

**3-D.** The construction is dimension-agnostic: `monomial_table` loops over axes, and
`gradient`/`hessian` loop over `support_dimension`. What 3-D needs is the phase-2 volume elements
and the phase-3 node placement, not new code here.

**Conditioning.** `cond(M)` stays under ~320 for every implemented element, and the coefficients
are obtained by `solve(Mᵀ, I)` rather than an explicit inverse. At high order on equispaced nodes
the conditioning degrades (phase 3 measures it), which is the argument for spectral nodes rather
than for a different construction.
