# NanoFEM Quadrature

**Status:** implemented and tested (phase 6). Companion to ARCHITECTURE_v2.md, the SDS,
REFERENCE_ELEMENTS.md, INTERPOLATION.md, SHAPE_FUNCTIONS.md, and GEOMETRIC_MAPPING.md.

**Scope discipline.** Integration rules on reference domains. A rule is points and weights such
that `sum_q w_q f(x_q)` approximates `integral f`, exactly for polynomials up to a declared
degree. There is no Jacobian-weighting, no integration over physical elements, no element
matrix, no assembly, and no constitutive model. A rule integrates scalar functions on a
reference domain and nothing more.

---

## 1. Quadrature theory

Numerical integration replaces an integral with a weighted sum at chosen points:

> `integral_K f(x) dx ≈ sum_q w_q f(x_q)`

The design freedom is the `n` points and `n` weights — `2n` numbers. A rule is **exact to
degree p** when the approximation is an equality for every polynomial of total degree `≤ p`.
Exactness is a statement about *polynomials only*; on anything else the sum is an approximation
that improves with order, which is why `integral exp` over `[-1,1]` is never exact but reaches
machine precision by eight points.

Everything in this package is integration over a **reference** domain. Integrating over a
physical element means composing a rule with a mapping's measure scaling `√det(JᵀJ)` — and that
composition is the element layer's job. A rule that multiplied its own weights by a Jacobian
would have to know what a Jacobian is, welding two independent ideas together; keeping them
apart is what lets one rule serve every element of a block.

---

## 2. Why Gaussian quadrature is optimal

An `n`-point rule has `2n` free parameters, so it can satisfy at most `2n` moment conditions and
integrate polynomials up to degree `2n − 1`. **Gauss–Legendre attains exactly that ceiling**,
and the ceiling is real:

> Given any `n`-point rule with points `x_q`, the polynomial `∏_q (x − x_q)²` has degree `2n`,
> is strictly positive except at the points, and integrates to **zero** under the rule while its
> true integral is **positive**. So no `n`-point rule is exact at degree `2n`.

The points that reach `2n − 1` are the **roots of the Legendre polynomial `P_n`**. That is the
"Legendre" in the name: the Legendre polynomials are orthogonal on `[-1,1]` under unit weight,
and placing points at the roots of the degree-`n` orthogonal polynomial integrates the extra `n`
degrees for free — because any degree `2n−1` polynomial is `q·P_n + r` with `deg q, r < n`, the
`q·P_n` part integrates to zero by orthogonality, and the rule integrates the remainder `r`
exactly since it has only `n − 1` degree.

The tests verify both halves independently: the points are checked to be Legendre roots against
a fresh `legval`, and the exactness is checked against the hand-derived closed forms in
`moments`.

---

## 3. Legendre polynomials and Gauss–Lobatto

**Gauss–Lobatto** fixes two points at the endpoints `−1` and `+1`. That spends two of the `2n`
degrees of freedom, so an `n`-point Lobatto rule reaches degree `2n − 3` — two behind Gauss.
Its interior points are the roots of `P'_{n−1}`, and its weights are
`w_i = 2 / (n(n−1) P_{n−1}(x_i)²)`.

What the endpoints buy is **collocation with nodal points**. When the quadrature points coincide
with a nodal basis's interpolation nodes, the mass matrix `∫ N_i N_j` comes out **diagonal**,
because `N_i(x_q) = δ_iq` makes every off-diagonal integrand vanish at every quadrature point.
That diagonal mass matrix is the whole engine of the spectral element method, and it is why
Lobatto exists as a family despite being two degrees weaker.

The tests pin the low-order cases against their classical names: two-point Lobatto is the
trapezoid rule, three-point is Simpson's rule (weights `1/3, 4/3, 1/3`), and the family is shown
costing exactly two degrees against Gauss at equal point count.

This is the ordering constraint from phase 3 made concrete. `SpectralInterpolation` was blocked
because its nodes *are* Gauss–Lobatto–Legendre points — which live in this package. That is why
`numerics.interpolation` will one day import `numerics.quadrature`, and therefore why quadrature
must import nothing from interpolation (§7).

---

## 4. Dunavant rules on the triangle

A tensor product does not fit a simplex: the inner integration limit depends on the outer
variable, so the integral does not separate. The triangle needs rules built for it directly, and
**Dunavant's** (1985) are the standard — fully symmetric under the triangle's six vertex
permutations, with the fewest points known per degree.

They are published in **barycentric coordinates**, organized into symmetric **orbits**:

| orbit | points | generator |
|---|---|---|
| centroid | 1, at `(1/3, 1/3, 1/3)` | — |
| three-fold | 3, permutations of `(a, b, b)` | one parameter `a`, with `b = (1−a)/2` |
| six-fold | 6, from `(a, b, c)` | needed from degree 6 up |

| degree | points | orbits | positive? |
|---|---|---|---|
| 1 | 1 | centroid | yes |
| 2 | 3 | one three-fold | yes |
| 3 | **4** | centroid + three-fold | **no** |
| 4 | 6 | two three-fold | yes |
| 5 | 7 | centroid + two three-fold | yes |

**The degree-3 rule has a negative centroid weight of `−27/48`.** This is not a transcription
error — it is the known price of reaching degree 3 with only four points. The rule reports it
through `has_positive_weights = False` rather than hiding it, because negative weights cost
accuracy through cancellation and can make an integrated mass matrix indefinite; a caller who
needs positivity asks for degree 4 (six points) instead. `verify_weight_positivity` checks the
*claim*, never imposes positivity, so this legitimate rule is not failed.

**Storing the generator, not the pair.** Deriving `b = (1−a)/2` from the single generator `a`
removes a transcription error: a table listing `a` and `b` independently to fifteen digits
leaves `a + 2b` off by around 1e-15, which drags points microscopically off the triangle.
Deriving it leaves at most one rounding of the division — so the barycentric sum is correct to
within a single ulp of the cartesian round-trip, which the tests assert.

Why the triangle earns its own family, stated as a regression test: degree-5 Dunavant needs
**seven** points where a degree-5 tensor product needs **nine** — and the product does not even
fit a triangle.

---

## 5. Tensor products, and the two notions of exactness

On a product domain the integral separates, so a rule follows from one 1-D rule per axis: points
are the cartesian product, weights are the products. If each factor is exact to degree `p`, the
product is exact for every monomial with **per-variable** degree `≤ p` — the whole space `Q_p`,
which *contains* `P_p` and reaches total degree `d·p`.

But the product's **total-degree** exactness is only `p`, because `ξ^(p+1)` — total degree `p+1`
— is a monomial the first factor gets wrong. So a tensor-product rule is genuinely stronger than
its reported degree, and the library records this honestly:

- `exactness_degree` returns `p`, the conservative total-degree number the rest of the library
  means by "exactness";
- `per_variable_exactness` returns `(p, p, …)`, the real per-axis strength.

Smuggling the extra strength into the total-degree number would break the meaning every other
layer relies on. A test pins both: `Q_3` integrates `ξ³η³` (total degree 6) exactly while `ξ⁴`
(total degree 4) fails.

**Anisotropic composition.** `from_rules` composes different rules per axis — cubic in one
direction, linear in the other — which is exactly what a field with different directional
degrees needs, without paying for cross terms. This surfaced the symmetry design (§6).

---

## 6. Symmetry: reported, not imposed

A symmetric rule integrates a polynomial the same way regardless of how a mesh generator numbered
the element's vertices; a non-symmetric one does not. So symmetry matters — but it is a
**property a rule may or may not have**, like positive weights, not a law to enforce.

The symmetry group itself is **derived, not hard-coded per cell**: try every vertex permutation,
fit an affine map, keep the fits that are exact and measure-preserving. Six fall out for the
triangle, eight for the square (the dihedral group — the other sixteen permutations admit no
affine map), two for the line. The tetrahedron's and hexahedron's groups will fall out of the
same loop.

The design that took a correction during implementation: an **anisotropic** tensor product is
legitimately *not* symmetric — swapping the square's axes swaps two unequal factors — so imposing
the full dihedral symmetry would *reject a valid rule*. The resolution:

- `declares_symmetry` — does the rule *claim* the domain's full symmetry (default yes; an
  anisotropic product says no);
- `invariant_symmetries` — the subgroup the rule *actually* respects, computed from the points;
- `is_symmetric` — whether those two coincide;
- `verify_symmetry` — fails only a rule that **claims** symmetry it lacks.

The orbits are then built from the invariant subgroup, so an anisotropic rule's orbits are the
true ones. The isotropic `3×3` Gauss grid recovers orbits `[4, 4, 1]` — four corners, four edge
midpoints, one centre — and Dunavant degree 5 recovers `[1, 3, 3]`, exactly its published
structure.

---

## 7. A leaf of the numerics layer

Quadrature imports the reference domains and **nothing else** from the library. This is
structural, not stylistic. Phase 3 recorded that the spectral interpolation family cannot precede
quadrature, because its nodes are GLL points; when it lands, `numerics.interpolation` imports
`numerics.quadrature`. Were the dependency to already run the other way, the two packages could
not both be imported.

So monomial evaluation (`monomial_values`) and the exact integrals (`exact_monomial_integral`)
are written *here*, not borrowed from the interpolation layer's tabulator — even though the
arithmetic overlaps. A subprocess test proves no interpolation or mapping module is imported, and
a source scan proves no `shape_function`, `stiffness`, `assembly`, or `constitutive` term appears
in executable code. The symmetry module likewise re-derives its affine fit rather than importing
the mapping layer's, for the same six-lines-versus-a-dependency reason.

---

## 8. Moment conditions and verification

A rule is verified against **analytic closed forms**, never against itself or another rule — a
rule checked against a rule proves only that they agree. Two closed forms cover every domain:

- **Products of intervals** (`[-1,1]^d`): `∫ ξ^a` is `2/(a+1)` for even `a`, zero for odd, and
  the domain integral is the product — so one odd exponent annihilates it.
- **Unit simplices**: `∫ ∏ x_i^{a_i} = (∏ a_i!)/((∑ a_i + d)!)`. For the triangle,
  `p! q!/(p+q+2)!`, which is `1/2` (the area) at `p=q=0`.

The `verify()` suite:

| check | catches |
|---|---|
| `verify_weight_normalization` | a mis-scaled table (weights don't sum to the measure) |
| `verify_weight_positivity` | a rule *claiming* positivity while carrying a negative weight |
| `verify_points_in_domain` | a transcription error putting a point outside the cell |
| `verify_exactness` | any monomial within the degree integrated wrongly |
| `verify_exactness_is_maximal` | a rule *understating* itself, so callers don't overpay |
| `verify_moment_identities` | measure and centroid, cross-checking phase 2 |
| `verify_symmetry` | a rule claiming symmetry it lacks |

**Maximality deserves note.** Without it, a rule could report a degree lower than it delivers and
every exactness test would still pass — and a caller would buy points it did not need. The check
requires *some* monomial one degree higher to fail; it tries every monomial of that degree,
because an odd one may vanish by symmetry and prove nothing. Each check has a trip test that
breaks the rule on purpose and confirms the check fires.

---

## 9. Examples

```python
from nanofem.numerics.quadrature import quadrature, DunavantQuadrature

rule = quadrature("triangle", 5)          # Dunavant, 7 points, exact to degree 5
rule.integrate(lambda p: 1 - p[:,0] - p[:,1])   # 1/6, by hand
rule.measure()                            # 1/2, the triangle's area
rule.centroid()                           # (1/3, 1/3)
rule.verify()                             # every identity in §8
```

The success-criterion chain, executable in `examples/ex06_quadrature.py`:
`ReferenceTriangle → DunavantQuadrature(order=5) → points → weights → verify exactness →
integrate scalar functions`, with no finite element constructed.

---

## 10. Rule selection and the SDS policy

`quadrature(cell, order, family=None)` returns a rule of exactness **at least** `order` — never
less. This is the SDS 2.5 policy the phase-0 seam declared: **full integration is the default and
implicit reduction is prohibited**. Reduced integration (deliberately under-integrating to soften
locking) is legitimate and documented, but it is a *choice* a caller makes by asking for a lower
degree, never something delivered silently.

Choosing the degree an element needs — the `m ≥ 2p` rule of thumb and its mapping-adjusted
refinements — is the *element* layer's business, since that layer knows the polynomial order of
what it integrates. This factory only guarantees that what was asked for is what is delivered.
Defaults are the fewest-point family per domain: Gauss on the line, its tensor product on the
square, Dunavant on the triangle. Rules are immutable value objects with read-only arrays, so the
factory memoizes them — identical requests return the identical instance.

The phase-0 `QuadratureFactory` seam (which takes a `ReferenceCell`, the mesh-name record the
mesh layer holds) is filled rather than bypassed, exactly as the `ShapeFunctions` seam was in
phase 4.

---

## 11. Future extension strategy

Each placeholder records a `BLOCKED_BY` string; none is blocked by scheduling.

**Gauss–Jacobi** — points exact against a weight `w(x) = (1−x)^α (1+x)^β` absorbed into the rule.
Two uses on this roadmap. The Duffy map collapsing a square onto a triangle carries a Jacobi
weight `(1−η)/2`, so Gauss–Jacobi is the standard route to **high-order simplex rules** past
Dunavant's tables. More importantly, **Eringen's nonlocal integral formulation** weights strain
by an attenuation kernel `α(|x − x'|)` that is *not polynomial* — precisely the situation a
weight-absorbing family is designed for. Needs Jacobi roots (numpy has none, so the recurrence
would be written here) and a place for `α, β` on the interface.

**Adaptive** — subdivide until an error estimate falls below tolerance. This one *does not fit
the interface*, which is the interesting part: every rule here has a fixed point set decided
before it sees an integrand, which is what lets a tabulation be computed once and shared across a
block (SDS C-8). An adaptive rule's points depend on the *function*, so it is an integration
**algorithm that calls rules**, not a rule — it belongs beside this package, not inside it.

**Sparse grid** — Smolyak combination for high dimensions, where the point count grows
polynomially in `d` rather than as `n^d`. Useless for element integrals (`d ≤ 3`, where a full
product is already cheap); it earns its place in **parametric and stochastic** analysis, where
the integration dimension is the number of uncertain material inputs. Its weights are signed by
construction — which is the original reason `has_positive_weights` had to be data rather than an
assumption.

**What the next phases take from here.** Element stiffness and mass integration compose a rule
with a mapping: evaluate shape functions at `rule.points` (cached once, SDS C-8), weight by
`rule.weights × mapping.volume_scale(rule.points)`, and sum. Every piece — the points, the
weights, the measure scaling from phase 5, the tabulation from phase 4 — already exists; the
element layer assembles them. **Facet rules** for boundary integrals and traction loads are the
natural next addition, and arrive once facet mappings do.
