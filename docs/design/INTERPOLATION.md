# NanoFEM Interpolation Framework

**Status:** implemented and tested (phase 3). Companion to ARCHITECTURE_v2.md, the SDS,
OBJECT_MODEL.md, and REFERENCE_ELEMENTS.md.

**Scope discipline.** This layer describes finite elements; it does not evaluate them. There
are no shape function equations, no gradients, no B-matrices, no quadrature, no Jacobians, no
mapping. What it *does* contain is the mathematical framework that makes those things
well-posed — and the verification suite that proves they will be.

---

## 1. Interpolation theory: what a finite element is

Following Ciarlet, a finite element is a triple:

> **(K, P, Σ)** — a domain `K`, a finite-dimensional space of functions `P` on it, and a set
> of linear functionals `Σ = {ℓ₁ … ℓₙ}` on `P` called the degrees of freedom.

The triple is **unisolvent** when the DOFs uniquely determine a member of the space: for any
values `(c₁ … cₙ)` there is exactly one `u ∈ P` with `ℓₖ(u) = cₖ`. Unisolvence is what
licenses the *nodal basis* `{N₁ … Nₙ} ⊂ P`, defined by duality:

> **ℓᵢ(Nⱼ) = δᵢⱼ**

Those `Nⱼ` are the shape functions. Note the direction of the logic: the shape functions are
not primary. They are *derived* from the triple, and they exist at all only because the triple
is unisolvent.

This layer supplies `K` (the phase-2 reference element), `P` (§2), and `Σ` (§3), and proves
unisolvence. It stops one step short of the dual basis.

### 1.1 The phase boundary, stated algebraically

Write the space in a *prime basis* of monomials `{m₁ … mₙ}` and pair every functional with
every monomial. The result is the **generalized Vandermonde**:

> **M[k, j] = ℓₖ(mⱼ)**

Every shape function is a combination of monomials, `Nᵢ = Σⱼ Cᵢⱼ mⱼ`. Imposing duality gives
`ℓₖ(Nᵢ) = Σⱼ Cᵢⱼ M[k,j] = δᵢₖ`, that is:

> **C = M⁻ᵀ**

So the whole of shape function construction is *one matrix inverse*, and this phase's boundary
is exactly that inverse:

| | this phase | the next |
|---|---|---|
| build `M` | ✅ `unisolvence_matrix()` | |
| prove `M` invertible | ✅ `verify_linear_independence()` | |
| invert `M` to get `C` | ❌ | ✅ |
| evaluate `N`, `∇N` | ❌ | ✅ `ShapeFunctions.evaluate` / `derivatives` |

Both contracts live in `interpolation/base.py` so the boundary is visible in one file.

### 1.2 Why the verifications need no shape functions

The requested checks all reduce to properties of `P` and `Σ` alone:

| Verification | Reduces to | Why |
|---|---|---|
| **Linear independence** | `M` square and full rank | ⟺ unisolvence, by definition |
| **Kronecker delta** | functionals distinct **and** `M` invertible | `ℓᵢ(Nⱼ) = δᵢⱼ` is how `N` is *defined*; what can fail is existence and uniqueness |
| **Partition of unity** | the constant is in `P` (given unisolvence) | `u ≡ 1` has DOF vector `ℓₖ(1)`, so `1 = Σₖ ℓₖ(1)·Nₖ` |
| **Polynomial completeness** | structural check on exponent tuples | the space *is* its monomial set |
| **Polynomial degree** | structural check on exponent tuples | as above |
| **Node ordering** | geometry against the reference element | uses phase-2 containment and incidence |

Only one of these touches numbers at all — the rank of `M` — and its entries are functionals
applied to *monomials*, not to shape functions. Differentiating a monomial (`D^α x^β`) is
elementary calculus on an exponent tuple; there is no nodal basis at this layer to
differentiate.

The partition-of-unity result deserves its statement in full, because it is sharper than the
usual folklore. Since a derivative functional annihilates a constant, `ℓₖ(1)` is **1 on value
DOFs and 0 on derivative DOFs**. Hence:

- for a **nodal** family (all DOFs are point values), `Σᵢ Nᵢ = 1` — the classical partition of
  unity;
- for **Hermite**, the *value*-dual functions sum to 1 while the *derivative*-dual functions
  contribute nothing.

`verify_partition_of_unity()` checks exactly this pattern, and the test asserts
`[ℓₖ(1)] == [1, 0, 1, 0]` for the cubic Hermite line.

---

## 2. Polynomial spaces

A space is described structurally, by the multi-indices of its spanning monomials
(`PolynomialSpace`). Two construction rules are implemented:

| Space | Rule | Dimension | Natural on |
|---|---|---|---|
| `P_k` | total degree `Σα ≤ k` | `C(k+d, d)` | simplices |
| `Q_k` | per-variable degree `max α ≤ k` | `(k+1)^d` | tensor-product cells |
| `S_k` | `P_k` plus selected higher monomials | hand-specified | *(serendipity, future)* |

Monomials are ordered **graded lexicographically** (by total degree, then lexicographically) —
deterministic and stable, so payloads and cache keys do not drift.

### 2.1 Three degrees, routinely conflated

The framework keeps them distinct because they genuinely differ:

- **order** `k` — the family's nominal order;
- **completeness degree** — the largest `p` with `P_p ⊆ P`. *This is what governs the
  approximation rate*;
- **maximum total degree** — the highest total degree present.

For `P_k` all three coincide. For `Q_k` in 2-D the completeness degree is `k` but the maximum
total degree is `2k`: `Q₂` contains `ξ²η²` (degree 4) yet is complete only to degree 2, because
`ξ³` is absent. A bilinear quad is not "second order" merely because it holds `ξη`. The
`verify_polynomial_degree()` check enforces `order == completeness_degree`, and
`verify_polynomial_completeness()` additionally enforces *maximality* — the next degree must be
incomplete, or the reported rate understates the space.

### 2.2 P ⊆ Q, and the one-variable coincidence

`P_k ⊆ Q_k` always (a total degree `≤ k` implies each variable's degree is `≤ k`). In **one**
variable they are the same set, which is why `ReferenceLine` may be treated as either a simplex
or a tensor-product cell without ambiguity (§5).

---

## 3. Lagrange vs Hermite

The families differ in `Σ`, not in `K` or (necessarily) `P`.

| | Lagrange | Hermite |
|---|---|---|
| DOFs | point values only | point values **and derivatives** |
| `is_nodal` | `True` | `False` |
| DOFs per node | 1 | 2 (line), 4 (quad) |
| Kronecker | classical `Nᵢ(xⱼ) = δᵢⱼ` | generalized `ℓᵢ(Nⱼ) = δᵢⱼ` only |
| Continuity | C0 | C1 (by design) |
| Extra order costs | more **nodes** | more **DOFs at the same nodes** |

That last row is the structural point, and it shows up in the metadata: the cubic Hermite line
has `num_nodes == 2` and `num_dofs == 4`, and its `mesh_cell_name` is `line2`, not `line4`. The
mesh does not grow; the DOF handler does — which is exactly the generalized `(field, component)`
DOF machinery architecture decision **D3** provisioned for in phase 1.

### 3.1 Implemented elements

| Family | Cell | Order | Space | Nodes | DOFs | Continuity |
|---|---|---|---|---|---|---|
| Lagrange | line | 1, 2, 3 | `P_k` | 2, 3, 4 | same | C0 |
| Lagrange | triangle | 1, 2, 3 | `P_k` | 3, 6, 10 | same | C0 |
| Lagrange | quadrilateral | 1, 2, 3 | `Q_k` | 4, 9, 16 | same | C0 |
| Hermite | line | 3 | `P₃` | 2 | 4 | C1 |
| Hermite | quadrilateral | 3 | `Q₃` | 4 | 16 | C1 (rectangles) |

The cubic Hermite line carries `(u, du/dξ)` at each vertex — DOF order `(w₁, θ₁, w₂, θ₂)`, the
classical Euler–Bernoulli beam ordering. The bicubic quadrilateral carries
`(u, ∂u/∂ξ, ∂u/∂η, ∂²u/∂ξ∂η)` at each vertex: the **Bogner–Fox–Schmit** element, which is the
tensor product of two 1-D cubic Hermites, cross-derivative and all.

### 3.2 Why there is no Hermite triangle

`HermiteInterpolation(CellType.TRIANGLE, 3)` raises, with the reason. This is deliberate.

The natural candidate — the 10-DOF cubic Hermite triangle (3 vertex values, 6 vertex
derivatives, 1 centroid value on `P₃`) — *is* unisolvent, and the framework would happily build
it. But it is only **C0**. On an edge, the trace is a cubic determined by the two endpoint
values and tangential derivatives, which neighbours share, so the trace matches. The **normal**
derivative along that edge is a *quadratic* — three conditions — while only the two endpoint
normal derivatives are shared. Two conditions cannot pin three coefficients, so gradients do
not match and C1 fails.

Shipping it labelled C1 would be wrong; shipping it labelled C0 would offer a Hermite element
that delivers nothing Hermite is chosen for, in a package whose roadmap needs C1 for
strain-gradient elasticity (phase 5). The honest answer is the real one: **C1 on simplices
requires the Argyris element** (quintic `P₅`, 21 DOFs, with second derivatives at vertices and
normal derivatives at edge midpoints) **or a macroelement such as Hsieh–Clough–Tocher**. Both
are genuine future work; neither is a cubic Hermite triangle.

---

## 4. Continuity classes

`Continuity` (C0, C1) is reused from `numerics/operators` — the same enum a theory's operators
derive their requirement from (dev note N-5), so an element's *provision* and a theory's
*demand* speak one vocabulary.

**Continuity is a property of the element and the mesh, not of the element alone.** The
`continuity` attribute is the element's *design* class, realized when its mesh conditions hold:

- Lagrange C0: unconditional (shared facet nodes).
- Hermite line C1: unconditional.
- Hermite quadrilateral C1: **on rectangular (axis-aligned, affinely mapped) meshes**. On a
  generally distorted quad the cross-derivative DOFs no longer align across a facet and only C0
  survives.

This caveat is documented on the class rather than buried, because a user reaching for BFS to
get C1 on a distorted mesh would otherwise get a silent downgrade.

---

## 5. Tensor-product vs simplex

The flags follow the cell shape:

| Cell | `is_simplex` | `is_tensor_product` | Space |
|---|---|---|---|
| line | ✅ | ✅ | `P_k` ≡ `Q_k` |
| triangle | ✅ | ❌ | `P_k` |
| quadrilateral | ❌ | ✅ | `Q_k` |

**The line is both**, and that is not a bug. A line is a 1-simplex *and* a 1-cube. Because
`P_k ≡ Q_k` in one variable (§2.2), the ambiguity has no consequence — the space is the same
set either way. A test asserts this explicitly so it reads as intent rather than oversight.

The practical difference the flags carry: simplex spaces are complete (`P_k`, cheapest for a
given rate), tensor-product spaces are separable (`Q_k`, which is what will later let
quadrature and tabulation factor across dimensions).

---

## 6. Node numbering conventions

**The reference element is the single source of topological truth** (SDS C-3). Node placement is
*derived* from its vertex and edge numbering, never restated:

1. **Vertices first**, in reference vertex order. Node `i` is reference vertex `i`.
2. **Then edge interiors**, edge by edge in reference edge order; within an edge, ordered from
   its first vertex toward its second.
3. **Then the cell interior.**

Each node records its owning entity (`EntityType` + index + local index), and
`verify_node_ordering()` checks every claim *geometrically* against the reference element — a
vertex node must sit exactly on that vertex, an edge node must lie strictly between that edge's
endpoints, an interior node must be strictly inside (using phase-2 `contains()` and
`signed_distance_to_boundary()`).

### 6.1 This differs from gmsh and VTK — deliberately

The reference triangle's edges are `((1,2), (2,0), (0,1))` — facet `i` opposite vertex `i`. So
`tri6` node 3 is the midpoint of `(V1, V2)` at `(0.5, 0.5)`. gmsh instead numbers node 3 as the
midpoint of `(V0, V1)`. The two orderings are a cyclic permutation of each other.

Adopting gmsh's ordering here would embed a *second* topological convention inside the geometry
core, contradicting the rule that the reference element owns those conventions. Translating
between internal and external orderings is the **mesh I/O adapter's** job — a thin edge at the
package boundary (architecture P5). A regression test pins our ordering and states the
divergence in its docstring, so the adapter's author meets the fact rather than discovering it.

### 6.2 The line's interior nodes belong to the cell

A line's only edge *is* the cell, so `line3`'s midpoint is associated with `EntityType.CELL`,
not an edge. This matters for sharing: a cell-interior node is never shared with a neighbour.

### 6.3 Mesh cell names, and a gap this surfaced

`mesh_cell_name` is derived from the geometric node count (`tri6`, `quad9`, `line2` for Hermite).
Cross-checking against the phase-0 `REFERENCE_CELLS` registry surfaced a real gap, now asserted
by a test so it stays visible:

- registered and used: `line2`, `line3`, `tri3`, `tri6`, `quad4`
- **`quad8` is registered but unused** — it is the *serendipity* cell, awaiting its family
- **`quad9` is not registered** — yet it is what Lagrange order 2 on a quad needs
- `line4`, `tri10`, `quad16` are likewise unregistered

The registry was populated with the classical *serendipity* quad and the Lagrange triangle. The
entries land with the families that consume them; nothing is broken today because nothing meshes
those cells yet.

---

## 7. Future extension strategy

Each declared placeholder is blocked by something specific, recorded in its `BLOCKED_BY` string
and its `PROVISIONAL_METADATA`. None is blocked merely by scheduling.

**Serendipity** (`S_k`; `quad8`, `quad12`). Boundary-only nodes, no interior nodes — same rate
as `Q_k` at lower cost, losing separability. Blocked because the `S_k` monomial set is
hand-specified rather than generated by a rule, so `PolynomialSpaceType.SERENDIPITY` has no
dimension formula. Everything else it needs — node placement on entities, unisolvence via the
same Vandermonde oracle — already exists. This family also closes the `quad8` gap in §6.3.

**Hierarchical** (modal, for p-refinement). Order-`k` functions are a *superset* of order-`(k-1)`
functions, so raising the order adds rather than replaces. Blocked because its DOFs are
**moments** — integrals against entity modes — not point functionals, so
`DofFunctional.apply_to_monomial` would need an integration rule, which is quadrature machinery
this phase excludes. It is also the family that makes the `evaluation_points()` /
`node_locations` split in the base class earn its keep: hierarchical elements have **no
interpolation nodes at all**, and `is_nodal` will be `False` with an empty `evaluation_points()`.

**Spectral** (Gauss–Lobatto–Legendre nodes). Same spaces and same point-value DOFs as Lagrange,
but with nodes at the GLL points — clustered near the boundary, keeping the Vandermonde well
conditioned at high order, and making the mass matrix diagonal when quadrature collocates with
the nodes. Blocked *structurally*: GLL points are roots of a derivative of a Legendre
polynomial, which is quadrature machinery. **This family cannot precede the quadrature layer.**

Its motivation is already measurable here.
`Interpolation.unisolvence_condition_number()` reports `cond(M)`, and a test asserts it grows
monotonically with order on the equispaced Lagrange nodes — the Runge phenomenon in another
guise. That diagnostic is why spectral families exist, and it is available now rather than being
asserted later.

**3-D cells.** `CellType` already carries the tetrahedron, hexahedron, prism, and pyramid, and
the node-placement helpers are written against `ReferenceElement` topology rather than hard-coded
shapes — `_edge_placements` works for any cell whose edges the reference element reports. What
3-D needs is the phase-2 volume elements (see REFERENCE_ELEMENTS.md §5), plus face-interior node
placement, which is a new placement helper rather than a change to the framework.

**The next phase.** Invert `M`, obtain `C = M⁻ᵀ`, and implement `ShapeFunctions.evaluate` and
`derivatives` by tabulating `C` against monomials at requested points. Because unisolvence is
already proven here, that inverse is guaranteed to exist — the failure mode has been eliminated
before the code that could hit it is written.
