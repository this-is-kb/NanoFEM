# NanoFEM Walking Skeleton (v0.8.0)

**Status:** implemented and tested. Companion to `ARCHITECTURE_v2.md` (which names this
milestone explicitly: "the phase-0 walking skeleton: a bar element end-to-end through every
layer... the first failing verification test (a bar under end load) that exercises every seam
in this document"), the SDS (Sections 2.2, 2.7-2.8, 2.11-2.15, 2.18), and ADR-002/ADR-003.

**Scope discipline.** One theory (`IsotropicElasticity`, dim=1 axial only), one element
(`Bar`, embedded in 1-D space only), one cross-section computed for real (`CircularSection`),
one load type (`NodalLoad`), one boundary condition kind (`DirichletBC`), one analysis
(`LinearStaticAnalysis`). Every other theory/element/section/load/analysis in the object model
stays exactly as declarative-only as it was in v0.1.0-v0.7.0. This phase's job is not breadth;
it is proving the seams fit, once, completely, with a real number coming out the other end.

---

## 1. Why a bar, and why 1-D-in-1-D

A two-node axial bar is the smallest element for which every layer in the pipeline is
non-trivial: it has a real cross-section, a real material law, a real stiffness, a real
assembly, a real constraint, and a real solve - and its closed form (`K = (EA/L)[[1,-1],[-1,1]]`)
is simple enough to be an independent oracle written by hand, not derived from nanofem code.

Restricting the embedding to 1-D-in-1-D (node coordinates are `(2, 1)`, not `(2, d)` for
`d > 1`) means the element's own local axis *is* the global axis: no direction-cosine
transformation exists to get wrong. `Bar.transformation_matrix()` returns `np.eye(2)` -
asserted orthonormal (E-10) like any other transformation, not a stand-in for missing rotation
logic. `Truss2D`/`Frame2D` (future `elements/structural/{truss,frame}.py`) own the genuine
direction-cosine case once an element needs to live in a mesh with `d > 1`.

---

## 2. The pipeline, end to end

```
Mesh (line2 cells, node regions)
  -> Material (E, nu, rho)          -> CrossSection (CircularSection: A, I, J, ...)
  -> IsotropicElasticity (Theory)   -> IsotropicElasticConstitutive (sigma = E eps, D = [E])
  -> Model (domains, materials, sections, theories, dirichlet_bcs, load_cases)
  -> Model.build_dof_handler()      -> DofHandler (node-major numbering, SDS C-2)
  -> build_elements(model, dh)      -> tuple[Bar, ...]                         (elements/factory.py)
  -> SparsityPattern.from_providers(elements, STIFFNESS, dh.num_dofs)          (numerics/assembly)
  -> Assembler(pattern).assemble(elements, STIFFNESS)  -> K (csr_matrix)
  -> ConstraintHandler(mesh, dh, model.dirichlet_bcs).partition() -> DofPartition
  -> per load case:
       NodalLoadProvider(load, mesh, dh, field_components, factor)
       Assembler(pattern).assemble(providers, FORCE)   -> f (dense NDArray)
       GlobalSystem(dh, {STIFFNESS: K}, {FORCE: f})
       ReducedSystem.from_global(system, free, constrained, prescribed)
       SparseDirectSolver().solve(k_ff, f_f)           -> u_f
       reduced.recover(u_f), reduced.reactions(u_f)    -> StaticResult
  -> LinearStaticAnalysis.run() -> dict[str, StaticResult]
```

Every arrow above is a real function call as of this phase; none was a stub before it and none
became one after.

---

## 3. `Bar`: the closed-form exception, and its equivalence proof

ADR-002 designates closed-form structural elements as the one place a matrix is *not* built by
composing `numerics.operators` + a `ConstitutiveModel` + quadrature: `K = (EA/L)[[1,-1],[-1,1]]`
is written directly in `Bar.local_stiffness()`, because it is a textbook-exact discrete weak
form with no discretization error to approximate away. SDS clause E-5 requires exactly one
thing in exchange: the element "declares equivalence to the composed path in its theory-manual
chapter - verification tests enforce it." This phase discharges that requirement in
`test_bar_verification.py::test_composed_path_matches_closed_form`, which builds the general
pipeline from scratch -

```
AffineMapping(LINE, nodes) -> LagrangeInterpolation(LINE, 1) shape functions
  -> quadrature(LINE, order=1)   (exact: the strain is constant for a linear bar)
  -> symmetric_gradient_matrix(physical_gradients)             (the B operator)
  -> IsotropicElasticConstitutive().respond_batch(...)         (D = [E])
  -> K_continuum = sum_q w_q |J_q| B_q^T D_q B_q
  -> K = area * K_continuum
```

- and asserts it equals `Bar.local_stiffness()` to `1e-9` relative tolerance. The area
multiplication happens by hand, in the test, after the length integral: a continuum `Theory`
integrates its weak form over length only (SDS Section 1's pipeline carries no
cross-sectional dimension), so cross-sectional area is an *element-layer* multiplier, not
something hidden inside the composed path.

A second, independent cross-check in the same file
(`test_constitutive_tangent_matches_isotropic_oracle_at_dim_one`) compares
`IsotropicElasticConstitutive`'s tangent against `numerics.tensors.fourth_order.isotropic_stiffness`
- a verification-oracle-only 3-D-generalized tensor, not a materials-layer law - which happens
to collapse to exactly `E` at `dim=1` because its deviatoric term vanishes identically there.
That is a genuine coincidence of the general formula, not a hidden dependency: the real
constitutive law (`physics/elasticity/isotropic.py`) implements `sigma = E eps` directly and
never calls the tensor-layer oracle.

---

## 4. `Model` accessors (the one "existing interface touched" item)

`core/model.py` gained five read-only accessors, each a one-line mirror of the existing
`theories` property, so `elements/factory.py` and `analysis/static.py` could read a fully
built `Model` without new coupling: `domains`, `materials`, `sections` (each
`-> dict[str, T]`), `dirichlet_bcs -> tuple[ConstraintLike, ...]`, and
`load_case(name) -> LoadCaseLike` (dict lookup, `ModelError` naming registered cases on a
miss - the same idiom `theories`/`load_case_names` already used). No existing method's
signature or behavior changed; two tests were adjusted to stop asserting the now-real cases
still raise (`test_materials_geometry.py` for `CircularSection`,
`test_model_and_analysis.py` for `LinearStaticAnalysis`).

---

## 5. SDS module -> concrete class map

| SDS section | Contract | Concrete class this phase |
|---|---|---|
| 2.2 Geometry | Section properties | `CircularSection` (`geometry/standard.py`) - all 9 methods, exact closed forms |
| 2.7 Theory | Discrete statement of governing equations | `IsotropicElasticity` (`physics/elasticity/isotropic.py`), dim=1 |
| 2.8 Constitutive model | Strain/state -> stress/tangent | `IsotropicElasticConstitutive` (same file) |
| 2.11 Element | Composition rule (E-1..E-13) | `Bar` (`elements/structural/bar.py`) |
| 2.12 Contribution provider | Assembly currency | `Bar.contributions`, `NodalLoadProvider` (`constraints/loads.py`) |
| 2.13 Assembler | Scatter into global operators | `SparsityPattern`, `Assembler` (`numerics/assembly/{sparsity,assembler}.py`) |
| 2.14 Boundary conditions | DOF partition, elimination | `ConstraintHandler`, `DofPartition` (`constraints/handler.py`), `ReducedSystem` (`numerics/assembly/system.py`) |
| 2.15 Linear solver | Solve `Ax = b` | `SparseDirectSolver` (`numerics/linalg/linear.py`) |
| 2.18 Analysis | Orchestrate: validate -> number -> assemble -> constrain -> solve -> package | `LinearStaticAnalysis.run()`, `StaticResult` (`analysis/static.py`) |

Everything else in the object model (Eigen solver, Time integrator, other theories, other
elements, MPC transformation, other load/BC kinds) is untouched declarative metadata, exactly
as before this phase.

---

## 6. What this phase deliberately does not cover

- **No MPCs.** `ReducedSystem.from_global` implements only Dirichlet elimination
  (`K_ff u_f = f_f - K_fc u_c`); the MPC transformation `T` (`u = T u_tilde + g0`) is real SDS
  2.14 scope but has no consumer yet.
- **No dynamics/buckling roles exercised.** `IsotropicElasticity` still declares
  `MASS`/`GEOMETRIC_STIFFNESS` per SDS 4.1 (so `ModalAnalysis`/`LinearBucklingAnalysis`'s
  existing `validate()` cross-check keeps passing), but nothing in this phase assembles them.
- **No 2-D/3-D elasticity.** `IsotropicElasticity(dim=2)` raises `PhysicsError` naming the gap
  (`physics/elasticity/plane.py`); `Truss2D`/`Frame2D` similarly wait on the direction-cosine
  transformation.
- **No richer singular-solve diagnostic.** `SparseDirectSolver` catches a non-finite solution
  and raises `SingularMatrixError`, but the zero-pivot -> (node, field, component) back-map SDS
  2.15 describes needs more machinery than one `spsolve` call exposes; deferred, not dropped.
