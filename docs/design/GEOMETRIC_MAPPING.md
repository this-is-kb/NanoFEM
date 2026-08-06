# NanoFEM Geometric Mapping

**Status:** implemented and tested (phase 5). Companion to ARCHITECTURE_v2.md, the SDS,
OBJECT_MODEL.md, REFERENCE_ELEMENTS.md, INTERPOLATION.md, and SHAPE_FUNCTIONS.md.

**Scope discipline.** Geometry only. No quadrature, no weak forms, no element matrices, no
assembly, no linear solvers, no constitutive models. The layer answers where a reference point
goes, how the map stretches there, and how a derivative transforms — nothing else.

---

## 1. Reference-to-physical mapping theory

A finite element computation is written on a fixed reference domain `K̂` and pushed onto each
physical element by a map:

> `x : K̂ → K`, `ξ ↦ x(ξ)`

Everything the element needs follows from differentiating that one function. This layer computes
the derivatives and their algebraic consequences; it never integrates them.

**The Jacobian** is the derivative of the map:

> `J[i,d] = ∂xᵢ/∂ξ_d`, shape `(embedding_dim, topological_dim)`

**The chain rule** relates the two gradients of any field `N`:

> `∂N/∂ξ_d = Σᵢ (∂N/∂xᵢ)(∂xᵢ/∂ξ_d)`  ⟹  `∇_ξ N = Jᵀ ∇_x N`

which is the identity everything else in this document is a rearrangement of.

---

## 2. Affine mappings

`x = A ξ + b`. The Jacobian is `A` everywhere, the mapping Hessian vanishes, and the inverse is
closed-form: `ξ = A⁺(x − b)`.

**Deriving A and b, and why the fit is checked.** Requiring `x(vᵢ) = Xᵢ` at every reference
vertex is a linear system in `(A, b)`. For a `d`-simplex the `d+1` vertices make it exactly
determined. For a quadrilateral, four corners impose **eight** conditions on **six** unknowns —
overdetermined, solvable only if the corners are consistent.

So `AffineMapping` solves in the least-squares sense and then **verifies the fit is exact**. A
non-parallelogram quadrilateral leaves a residual and raises `NonAffineError`, naming the
isoparametric map as the answer. The parallelogram rule is nowhere in the code; it *falls out*.

`IdentityMapping` is a genuine subclass (`A = I`, `b = 0`), not a parallel implementation, so it
inherits every identity and every check. It is the control every other map is read against:
physical gradients equal reference gradients, the metric is `I`, and the measure scaling is 1.

---

## 3. Isoparametric mappings

> `x(ξ) = Σₐ Nₐ(ξ) xₐ`

with `Nₐ` a geometry basis and `xₐ` its nodes' physical coordinates. Differentiating:

> `J[i,d] = Σₐ xₐ[i] ∂Nₐ/∂ξ_d`  and  `K[i,a,b] = Σₐ xₐ[i] ∂²Nₐ/∂ξ_a∂ξ_b`

so the mapping is a **contraction of phase-4 shape function tables against node coordinates**.
This module writes no basis mathematics of its own; `IsoparametricMapping` holds a
`ShapeFunctionFamily` and consumes it.

**The name is looser than the class.** "Isoparametric" means the geometry basis equals the field
basis. Nothing here requires that — the class takes whatever basis it is given, so a
sub-parametric element (linear geometry, quadratic field: the usual choice for straight-sided
high-order elements) is the same class with a different argument. The field basis is not this
layer's business.

**The geometry basis must be nodal.** An isoparametric map interpolates node *coordinates*; a
Hermite basis would demand nodal *derivatives of the geometry*, which are not node data. The
constructor rejects it.

**Affineness is a property of the geometry, not the element type.** A `Q1` quadrilateral is
bilinear in general but affine when it is a parallelogram; a `P2` triangle with mid-side nodes at
the midpoints is affine despite its quadratic basis. Both are common. So `is_affine` is answered
by inspecting the mapping Hessian, and both cases are tested.

**Inversion.** A general isoparametric map is a polynomial system with no closed-form inverse, so
`inverse_map` runs Newton from the reference centroid using the Jacobian the map already
provides. For affine geometry the residual is linear and it lands in one step.

Newton may return coordinates **outside** the reference cell — correctly. The map is a
polynomial, defined beyond `K̂`, and the inverse answers "which reference coordinates map here".
Whether the answer lies in the cell is the caller's question, asked with
`ReferenceElement.contains`. This is the same choice deal.II's `transform_real_to_unit_cell`
makes.

---

## 4. Jacobian derivation, and the non-square case

The Jacobian is `(embedding_dim, topological_dim)`. When those are equal it is square. When the
element is **embedded** — a bar in a plane, a shell in space, both of which the roadmap needs for
`Truss2D` and `Frame2D` — it is tall, and three things change:

| | square | embedded |
|---|---|---|
| determinant | `det J`, signed | **does not exist** |
| measure scaling | `\|det J\|` | `√det(JᵀJ)`, the Gram determinant |
| inverse | `J⁻¹` | `J⁺ = (JᵀJ)⁻¹Jᵀ`, the pseudo-inverse |
| `J J⁻¹` | `I` | the **tangent projector**, symmetric and idempotent but not `I` |

Writing every formula through the pseudo-inverse gives **one code path** for both cases, since
`J⁺ = J⁻¹` exactly when square. That is why embedded elements need no architectural change later.
`jacobian_determinant` raises `EmbeddedMappingError` for a tall Jacobian rather than inventing a
sign — a bar in a plane has no handedness — and points at `volume_scale`, which is defined in
both cases.

---

## 5. Metric tensor, covariant and contravariant bases

**Covariant basis:** `g_a = ∂x/∂ξ_a` — the tangent vectors the reference axes map onto, i.e. the
columns of `J`. Generally neither unit nor orthogonal, which is exactly why a metric is needed.

**Metric tensor (first fundamental form):** `G = JᵀJ`, i.e. `G[a,b] = g_a · g_b`. Symmetric,
positive definite for a non-degenerate map, and **square even when `J` is not** — which is what
makes it the right object for embedded elements.

**Contravariant basis:** `g^a` with `g^a · g_b = δᵃ_b`. Working it out: `g^a = G^{ab} g_b`, so the
matrix is `G⁻¹Jᵀ` — **which is exactly `J⁺`**. `contravariant_basis` and `inverse_jacobian`
therefore return the same array. Both names are kept because the two ideas are worth
distinguishing and a reader arriving from differential geometry should find the one they expect.

---

## 6. Gradient transformation

From `∇_ξ N = Jᵀ ∇_x N`:

> **push-forward:** `∇_x N = J⁺ᵀ ∇_ξ N`   **pull-back:** `∇_ξ N = Jᵀ ∇_x N`

For a square `J` the push-forward is the familiar `J⁻ᵀ ∇_ξ N`. For an embedded element it yields
the **tangential** gradient — the only part that exists. Equivalently
`∇_x N = Σ_a (∂N/∂ξ_a) g^a`: the reference derivatives are the gradient's components in the
contravariant basis.

Note the asymmetry: the **pull-back needs no inverse** and is always well posed; the push-forward
is where all the delicacy lives.

### The physical Hessian, and the term everyone drops

Differentiating the chain rule again:

> `∂²N/∂ξ_a∂ξ_b = Σᵢⱼ (∂²N/∂xᵢ∂xⱼ) J[i,a] J[j,b] + Σᵢ (∂N/∂xᵢ) K[i,a,b]`

Solving:

> **`H_x = J⁺ᵀ (H_ξ − ∇_x N · K) J⁺`**

The subtracted term is what the phase-0 SDS 2.6 note calls *mandatory unless the map declares
affineness*. For an affine map `K = 0` and the formula collapses to `J⁻ᵀ H_ξ J⁻¹`. On a bilinear
or curved element, dropping it is **silently wrong** — and the error is invisible in the
gradient, surfacing only when a C1 theory needs second derivatives. A test shows the naive
formula disagreeing on a trapezoid, and the corrected one agreeing with a finite difference taken
in *physical* space, which shares no code with it.

`physical_hessian` raises for embedded elements. This is not laziness: the second derivative of a
field defined only on a manifold is not determined by the field alone — the second fundamental
form enters. Returning a plausible wrong answer would be worse than refusing.

---

## 7. Orientation

For a square Jacobian, `det J` carries the orientation. `det J < 0` means the element is
inverted, which is what bad node ordering looks like from here — so the error says exactly that
rather than reporting a number.

Orientation is **invisible to the measure**: a reversed triangle has the same `volume_scale` and
the same `|det J|`. Only the sign distinguishes them, which is why `volume_scale` cannot be the
validity check and `jacobian_determinant` is not merely `abs`-of-something. A test pins this.

For embedded elements orientation is undefined and `verify_orientation` correctly does nothing.

---

## 8. Common implementation pitfalls

This layer is where the classic errors live. Each is guarded, and each guard has a test that
fires it.

**1. Absolute tolerances on `det J`.** The obvious degeneracy check — "is the measure scaling
near zero" — is *wrong*, and for this package dangerously so. The measure scaling has units of
lengthᵈ, so any absolute threshold encodes an assumed element size. **A 1 nm triangle in SI units
has an area scaling near 1e-18** and would be rejected as degenerate by a library built for
nanoscale mechanics. The criterion here is `σ_min/σ_max` of `J`, which is dimensionless and
therefore invariant under the uniform scaling that separates those cases. A test sweeps five
decades of element size and asserts identical quality and validity.

**2. `det(JᵀJ)` squares the condition number.** A rank-deficient map leaves a Gram determinant
near machine epsilon and a measure scaling near its *square root* — about **1e-8**, far above any
threshold a reader would think to write. The collinear triangle in the tests demonstrates exactly
this. Singular values lose nothing, so they are what the check uses.

**3. Building `J⁺` as `(JᵀJ)⁻¹Jᵀ`.** Same root cause: forming `JᵀJ` squares the conditioning, so
a merely awkward Jacobian yields a numerically singular metric and numpy raises `LinAlgError`
from inside what the caller thinks is a geometry query. `inverse_jacobian` uses the SVD
pseudo-inverse — the same object without the squaring.

**4. Dropping the mapping Hessian.** See §6. Invisible until a C1 theory needs it.

**5. Transposing the push-forward.** `J⁻¹` where `J⁻ᵀ` belongs is dimensionally fine and silently
wrong. It is caught by requiring `Σₐ xₐ ⊗ ∇_x Nₐ = J J⁻¹ = I`. Note the test for this needs a
**non-symmetric** Jacobian: a symmetric one makes `Jᵀ J⁻¹` collapse to the identity and the
misindexed contraction is accidentally correct. The first draft of that test used a symmetric
triangle and passed against broken code.

**6. Blaming the element for the solver's excursion.** A diverging Newton iterate can wander
somewhere the map genuinely *is* degenerate. Propagating that degeneracy reports an element
defect and sends the user to inspect a mesh that is perfectly sound. `inverse_map` translates it:
the failure is an inverse that could not be continued, and the message says the point most likely
has no preimage.

**7. Confusing the centroid with the centroid.** `centroid` here is the **image of the reference
centroid**, not the area centroid `∫x dA / ∫dA` — the latter needs integration this layer must
not perform. For an affine map they coincide; for a curved one they do not. Documented on the
property rather than left to be discovered.

---

## 9. Validation and verification

Two separate concerns, deliberately.

**`validate()` — is this physical element usable?** Coincident nodes; a rank-deficient Jacobian
(degenerate); a negative determinant (inverted, i.e. bad node ordering); a near-singular map (a
sliver). Raises `DegenerateCellError`, which the exception tree already declares as "non-positive
Jacobian or zero-measure cell (SDS 2.6)" — reused rather than redefined, and it sits under
`MeshError` because a degenerate map is almost always a bad mesh, which is where the user should
look. This layer raises with what it knows (the reference point, the singular values); the
element layer above adds element ids when it re-raises.

**`verify()` — is the mathematics right?** Jacobian against central finite differences (the
independent standard, sharing no code with the analytic route); metric is `JᵀJ`, symmetric,
positive definite, and dual to the contravariant basis; the reference↔physical round trip;
gradient push-forward/pull-back mutually inverse plus the `J J⁻¹ = I` identity; orientation
preserved; and measure scaling against a closed-form chord length or shoelace area.

`verify_measure_scaling` runs only for affine maps — a non-affine element's measure genuinely
*is* an integral, so `physical_measure` raises and points at the quadrature layer rather than
approximating.

---

## 10. Future extension strategy

Each placeholder records a `BLOCKED_BY` string. What is worth noticing is how *little* each
needs: because the base class is written through the pseudo-inverse and the mapping Hessian
rather than around them, every one of these is a new `map` / `jacobian` / `mapping_hessian`
triple and inherits the metric, the bases, the derivative transformations, the validation, and
the verification suite unchanged.

**CurvilinearMapping** — geometry following an exact surface rather than a polynomial interpolant.
Blocked on having an analytic surface description in the library at all; a geometry/CAD concern,
not a mapping one.

**NURBSMapping** — `x(ξ) = Σₐ Rₐ(ξ) Pₐ` over rational B-splines. Structurally an isoparametric map
with a different basis, which is exactly why it is blocked **in the interpolation layer**: a NURBS
basis is rational (not spanned by a monomial space), patch-supported (not cell-supported), and
non-interpolatory (control points are not nodes). It is not a `ShapeFunctionFamily`, and that
must be solved one layer down first.

**HighOrderMapping** — honest about being *mathematically redundant today*: `IsoparametricMapping`
is order-agnostic and a cubic geometry already maps and verifies (a test proves it). What is
missing is what makes high order *trustworthy*: Gauss–Lobatto node sets (which is the spectral
family, still blocked on quadrature), interior validity checking — a high-order map can invert in
a cell's interior while every vertex looks fine — and transfinite blending for partially curved
elements.

**What the next phases need from here.** Quadrature supplies *points*, and this layer already
takes points and caches every derived batch per point set — so a rule's Jacobians are computed
once and shared across a block (SDS C-8). Element matrices will need `∇_x N` and `volume_scale` at
those points; both are here. Facet mappings and outward physical normals are the natural next
addition and slot in beside the cell map.

**The roadmap's harder physics is already provisioned.** Eringen nonlocal integral formulations
need physical distances between points in *different* elements — which is `map` on two elements,
with no new mapping concept. Strain-gradient and couple-stress theories need the physical Hessian
*with* its correction term, which is implemented and tested rather than deferred. Auxetic RVE
work needs affine periodicity, which is `AffineMapping`'s `linear`/`translation` pair.
