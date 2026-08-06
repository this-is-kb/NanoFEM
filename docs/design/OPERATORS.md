# NanoFEM Operators

**Status:** implemented and tested (v0.7.0). Companion to ARCHITECTURE_v2.md, the SDS
(Section 8), and TENSORS.md, which this layer consumes for Voigt/Mandel conventions and
rotations.

**Scope discipline.** Stateless recipes: tabulated data already evaluated at quadrature points
(shape values, physical gradients, physical Hessians, facet normals - all plain `NDArray`
batches) in, a matrix mapping nodal DOFs to a pointwise operator value out. No mesh, no
materials, no constitutive model, no assembly, and no `ShapeFunctionFamily`/`GeometricMapping`/
`QuadratureRule` instance ever crosses this layer's boundary - only their already-tabulated
output does.

---

## 1. The operator vocabulary, and the seam it fills

`operators/base.py` (v0.1.0) already declared the vocabulary: `OPERATOR_CATALOG`, the
`OPERATOR_DERIVATIVE_ORDER` table, `derived_continuity`, and the `DiscreteOperator` ABC
(`name()`, `required_derivative_order()`). This phase fills the seam it left open: a compute
method for each of the eleven catalog entries. Ten get a real recipe; `nonlocal_integral` is a
declared, blocked placeholder (§9).

Every recipe module pairs a **plain function** (the thing actually called in a hot loop) with a
thin `DiscreteOperator` subclass whose own `verify()`/`is_valid()` self-checks it against
synthetic data it builds internally - matching the class-based, subclass-trippable pattern
every prior numerics phase used, since these genuinely are classes (unlike `tensors`, which is
pure functions with no natural class to hang verification off).

**Per-element, not per-mesh.** `GeometricMapping`/`ShapeFunctionFamily` operate on one element
at a time, shape `(n_points, n_functions, dim[, dim])` - there is no multi-element batching
axis anywhere in the codebase yet, even though SDS Section 8's notation is written
prospectively for one (`(n_e, n_qp, rows, n_dof)`). This layer follows the convention actually
in force: every recipe here operates **per element**, shape `(n_qp, rows, cols)`.

**DOF axis left unflattened.** `symmetric_gradient`/`divergence`/`curl`/`surface_gradient`
return `(n_qp, rows, n_fun, dim)` rather than a flattened `(rows, n_dof)` B-matrix, deferring
the node-major-vs-other DOF ordering choice to the not-yet-built `elements/` layer - a
premature commitment here would be exactly the kind of ahead-of-a-consumer decision this
project's phase discipline avoids.

---

## 2. Gradient, divergence, curl

`gradient_matrix` is the transpose of `physical_gradients`' last two axes: `G[d, a] =
dN_a/dx_d`. `divergence_matrix` is `physical_gradients` itself with an inserted row axis - a
divergence contribution is a single scalar row per quadrature point. `curl_matrix` covers both
the 2-D scalar rotation (`theta = du_y/dx - du_x/dy`, **un-halved** - couple-stress theory's
`theta = 1/2 curl(u)` scaling is theory-level, applied downstream) and the 3-D vector curl
(built from the Levi-Civita symbol: `row[k,a,j] = sum_i eps_kij dN_a/dx_i`); there is no 1-D
curl, and asking for one raises `UnsupportedDimensionError` rather than returning a silent
zero.

---

## 3. Symmetric gradient (the B operator)

`symmetric_gradient_matrix` is the strain-displacement operator every elasticity-family theory
composes, Voigt-ordered via the exact `VOIGT_ORDER` table `tensors/conventions.py` uses, so the
two never drift apart. For a diagonal Voigt pair `(i,i)`, only DOF component `i` contributes
(`dN_a/dx_i`); for an off-diagonal pair `(i,j)`, both components `i` and `j` contribute
(`dN_a/dx_j` and `dN_a/dx_i` respectively), producing the engineering-shear sum `gamma_ij =
du_i/dx_j + du_j/dx_i` on contraction with nodal DOFs.

---

## 4. Laplacian and Helmholtz

`laplacian_matrix` is the one recipe in this package that is already a full element matrix
rather than a per-point row - a weak-form pairing has nothing further to compose with, which
is exactly the operator/integrand blur SDS Section 8 notes and keeps here anyway, since it
remains a physics-free recipe (no material, no assembly). `helmholtz_matrix` is
`mass_term + l^2 * laplacian_matrix` - the building block of Eringen's differential nonlocal
elasticity (SDS Section 4.2). The length scale `l` is a **per-call float argument**, never
constructor state: a `DiscreteOperator` recipe is contractually stateless and
materials-blind, and a length scale is a material/nonlocal parameter - exactly why
`kernels.Kernel` objects, which *do* carry state, live in a separate, higher layer.

---

## 5. Surface gradient

`surface_projector(normals) = I - n (x) n` and `surface_gradient_matrix = P grad(u)`, projecting
only the *direction* of differentiation onto the tangent plane - it does not mix vector
components, so the same projected-direction row is used for every DOF component (broadcast,
not contracted, over the component axis). This is the building block Gurtin-Murdoch surface
elasticity (SDS Section 4.6) composes into a symmetrized surface strain; that symmetrization is
theory-specific and lands with `physics/surface`, which does not exist yet.

---

## 6. Second gradient (Hessian)

`second_gradient_tensor` is a deliberately thin, validated pass-through of
`GeometricMapping.physical_hessian`'s existing output - the object SDS calls `B_H`. Its only
job here is the shape/rank contract and the continuity consequence (composing this operator
forces `Continuity.C1` through `derived_continuity`, unconditionally). The specific contraction
a strain-gradient or couple-stress theory needs from the raw Hessian is theory-specific and
belongs with `physics/strain_gradient`/`physics/couple_stress`, neither of which exists yet -
inventing that contraction now would be exactly the kind of ahead-of-a-consumer work this
project's phase discipline forbids.

---

## 7. Voigt operators and transformation

`identity_vector`, `deviatoric_projector_voigt`, and `trace_row_voigt` are constant matrices
per dimension built from `tensors.conventions.VOIGT_ORDER` - the same source of truth the
tensor layer itself uses, so `m . strain_voigt(eps) = tr(eps)` and
`deviatoric_projector_voigt(d) @ strain_voigt(eps) = strain_voigt(dev(eps))` hold by
construction rather than by two independently-hand-derived formulas. `transformation` is a thin
dispatcher over `tensors.rotations`' Bond transformation pair: `vector_transformation_matrix(Q)
= Q` itself, and `tensor_transformation_matrix_voigt(Q, kind=...)` returns `M_sigma` or
`M_epsilon` for rotating a Voigt-form stress or strain.

---

## 8. A leaf of the numerics layer

`operators` imports `tensors` (one direction only - Voigt conventions, projectors, and
rotations) and nothing else beyond `numpy` and `nanofem.utils.exceptions`: no mesh, no
materials, no physics, no elements, no interpolation, no mapping, no quadrature, no kernels, no
search. Recipe functions take plain, already-tabulated `NDArray` batches - never a
`ShapeFunctionFamily`/`GeometricMapping`/`QuadratureRule` instance - which is what keeps this
package a true leaf rather than merely an unimported one. Proven the same way every prior
numerics phase proved it: a subprocess test imports the package fresh, exercises every real
operator, and asserts none of the forbidden package names appear in `sys.modules`; a source
scan strips comments and docstrings and asserts no `DofHandler`, `assembl`, `SparsityPattern`,
`Material`, or `ShapeFunctionFamily` term appears in executable code.

`operators` depends on `tensors` in exactly one direction (four of ten real recipes -
`symmetric_gradient`, `surface_gradient`, `voigt_map`, `transformation` - consume it); `tensors`
depends on nothing new. Both are siblings under `numerics`, so rule R1 (`numerics` imports
nothing from mechanics) is untouched, and `physics/base.py` already imports
`numerics.operators.base` directly - confirming `operators` is meant to be physics-visible
while `interpolation`/`mapping`/`quadrature` are not (rule R2).

---

## 9. Why `nonlocal_integral` is a declared placeholder

Eringen's integral (two-phase) formulation needs a pair-block recipe over `(e, e')` weighted by
an attenuation kernel `alpha(|x-x'|)` (SDS Section 4.3). That kernel is `kernels.Kernel`
territory - still a phase-0 skeleton with no concrete kernel - and the horizon pairs it needs
come from `numerics.search.NeighborSearch`, currently a one-method stub. Both are **structural**
prerequisites this operator's own contract names as inputs, not merely items later on a
schedule; `kernels/` additionally sits *above* `numerics` in the import-linter layer contract,
so `operators` could not import it even if a concrete `Kernel` existed today. `future.py`
declares `NonlocalIntegralOperator` exactly as `GaussJacobiQuadrature`/`CurvilinearMapping` were
declared: `PROVISIONAL_METADATA` describing the intended shape, and a `BLOCKED_BY` string naming
the real blocker. It is excluded from `OPERATOR_REGISTRY`, mirroring how
`AVAILABLE_QUADRATURES` excludes its own placeholders.

---

## 10. Verification

| check | catches |
|---|---|
| each operator's own `verify()` | a broken recipe, checked against a hand-derived closed form specific to that operator (§2-§7) |
| `verify_registry_self_consistency` | a registered class whose `name()`/`required_derivative_order()` drifts from the `OPERATOR_CATALOG` table |
| `verify_continuity_derivation` | `second_gradient` no longer forcing `Continuity.C1` through the shared derivation rule |
| `verify_cross_operator_consistency` | `divergence_matrix` and `trace(symmetric_gradient_matrix(...))` disagreeing on the same field - two independently-written recipes checked against each other |

`verify_operator_library()` runs all of the above in sequence and raises on the first failure;
`is_operator_library_valid()` wraps it as a boolean. `NonlocalIntegralOperator`'s own test is
that it refuses construction and names its blocker, mirroring the placeholder tests in every
prior phase.

---

## 11. Examples

```python
from nanofem.numerics.operators import (
    symmetric_gradient_matrix, laplacian_matrix, verify_operator_library,
)
from nanofem.numerics.tensors import strain_to_voigt

matrix = symmetric_gradient_matrix(physical_gradients)   # (n_qp, n_voigt, n_fun, dim)
strain_voigt = matrix.reshape(...) @ nodal_dofs           # element-layer's flattening choice
verify_operator_library()                                # every identity in the table above
```

The success-criterion chain, executable in `examples/ex07_operators_and_tensors.py`: a
triangle's `ShapeFunctionFamily` gradients and `IsoparametricMapping` push-forward feed
`gradient_matrix`/`symmetric_gradient_matrix`, whose output is converted to Voigt and Mandel
form and checked for work conjugacy, with both this package's and `tensors`' verification
suites run at the end - no element, no assembly, no constitutive model.

---

## 12. Future extension strategy

**`nonlocal_integral`** becomes a real recipe once `kernels/` has a concrete `Kernel`
(evaluation, support radius, normalization policy) and `numerics/search/` has a working
`NeighborSearch` - both named, both currently stubs. The pair-block recipe itself
(`K_NL^(e,e') = integral_e integral_e' B^T(x) alpha(|x-x'|) C B(x') dOmega' dOmega`) is already
fully specified in SDS Section 4.3; nothing about its *mathematics* is blocked, only its
*inputs*.

**What the next phases take from here.** The not-yet-built `elements/` layer is this package's
first real consumer: it will flatten each operator's `(n_qp, rows, n_fun, dim)` output into the
`(rows, n_dof)` B-matrix shape SDS Section 8's notation anticipates, choosing the node-major DOF
ordering SDS C-2 specifies. `physics/elasticity`'s first `Theory` will compose
`symmetric_gradient_matrix` with a constitutive tangent from `numerics.tensors.fourth_order` to
build the first real stiffness integrand - the walking-skeleton milestone this phase is one step
closer to.
