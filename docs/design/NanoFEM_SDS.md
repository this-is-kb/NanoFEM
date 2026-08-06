# NanoFEM — Software Design Specification (SDS)

**Document status:** Pre-implementation design specification. Architecture per ARCHITECTURE_v2 is frozen; this document defines the mathematical and software contracts *between* the frozen packages.
**Audience:** developers implementing modules independently. Two developers who each read only this document and implement different modules MUST produce compatible code.
**Normative language:** MUST / MUST NOT / SHOULD / MAY are used in the RFC-2119 sense. Everything marked *Design decision* or *Limitation* is rationale, not contract.

---

## 0. Preliminaries

### 0.1 Folder amendments (the only three, recorded as ADRs)

The frozen architecture is amended in exactly three places, each forced by a section of this SDS and each additive or consolidating:

| Amendment | What | Why | ADR |
|---|---|---|---|
| A1 | `numerics/operators/` added | §8 mandates an operator library; it is pure approximation machinery, so it belongs under the mechanics-free `numerics/` umbrella (rule R1 unchanged) | ADR-015 |
| A2 | `numerics/tensors/` added | §9 mandates a tensor algebra module; same reasoning as A1 | ADR-016 |
| A3 | `research/verification/` renamed and restructured to `research/validation/` with the §14 taxonomy | One home for evidence, not two; `tests/verification/` (fast, CI) is unchanged | ADR-017 |

No other folder changes. Everything else in this SDS binds behavior *within* the existing structure.

### 0.2 Notation

| Symbol | Meaning |
|---|---|
| d | spatial dimension (1 or 2 now; 3 designed for) |
| Ω, Γ | domain, boundary; Ω_e a cell, Γ_f a facet |
| K̂, ξ | reference cell and reference coordinates; x = F(ξ) the mapped point |
| n_en, n_qp | nodes per element, quadrature points per rule |
| n_dof_e, N | element DOF count, global DOF count |
| n_σ | Voigt length of symmetric tensors: 1 (d=1), 3 (d=2), 6 (d=3) |
| N_a, ∇N_a | shape function of node a and its physical gradient |
| B, D | generalized strain–DOF operator; constitutive tangent |
| u, ε, σ | displacement, small strain, Cauchy stress |
| (p, q, …) | array shape specification — shapes are contract, not implementation |

Array-shape statements such as "(n_qp, n_en, d)" are normative interface facts; they are not code.

### 0.3 Global conventions (the interoperability core)

Independent implementations diverge at conventions, not at equations. The following are therefore binding on every module.

**C-1 Voigt ordering and shear convention.** 2-D: (xx, yy, xy); 3-D: (xx, yy, zz, yz, xz, xy). Strain vectors are *kinematic* Voigt (engineering shear, γ_xy = 2ε_xy); stress vectors are *kinetic* Voigt (plain components). Work conjugacy σᵀε_voigt = σ:ε then holds without extra factors. Conversions to/from Mandel and full tensor form are provided by `numerics/tensors` (§9); the Voigt forms above are the wire format of all B/D/σ/ε exchanges.

**C-2 Element DOF ordering.** Node-major, then field declaration order, then component order: (node₁: u_x, u_y, r_z; node₂: u_x, …). A DOF signature (§3) makes this explicit; every element and the DofHandler MUST agree with this rule.

**C-3 Reference cells and orientation.** Line: ξ ∈ [−1, 1]. Triangle: vertices (0,0), (1,0), (0,1), counterclockwise; facet i is opposite vertex i. Quad: [−1,1]², vertices counterclockwise from (−1,−1); facets enumerated bottom, right, top, left. Facet normals point outward from the cell. Cited from `numerics/reference`; no module may embed its own copies.

**C-4 Numeric types.** float64 for all field/operator data; int64 for DOF indices; C-ordering.

**C-5 Determinism.** DOF numbering, quadrature ordering, and assembly iteration order MUST be pure deterministic functions of the model definition. This is what makes restart (§7), regression tests, and reproducibility manifests (§14) possible.

**C-6 Units.** The library is unit-agnostic; one consistent unit system per model is the user's obligation. Validation checks bounds, never units. The Nondimensionalizer addresses scale spread (nm geometry × GPa moduli); its use is advised by diagnostics (§13), never applied silently.

**C-7 Symmetry tolerance.** Any operator declared symmetric MUST satisfy ‖A − Aᵀ‖_∞ ≤ 10⁻¹² ‖A‖_∞ at emission. Debug builds assert; release builds trust.

**C-8 Batching.** Every hot-path contract (constitutive response, operator evaluation, contribution emission) is specified over *element blocks* — leading array axes (n_e, n_qp, …). A single element is a block of one. This is the GPU/vectorization enabler referenced throughout.

---

## SECTION 1 — Overall mathematical pipeline

The pipeline below is the *evaluation chain*. Statically, dependencies point down the chain (per the frozen dependency graph); at runtime, an Analysis orchestrates top-down and data flows back up. Two senses of "geometry" are distinguished once and for all: **domain geometry** (the meshed body, entering at stage 1) and **section geometry** (cross-section property models, entering at stage 7 as element data).

| # | Stage | Consumes | Produces (exact data crossing the boundary) |
|---|---|---|---|
| 1 | Geometry (domain) | parametric solids/curves, physical group names, mesh size fields | a meshing request: geometry kernel entities tagged with region names |
| 2 | Mesh | stage-1 request (via gmsh) or a file (via meshio) | node coordinates X (n_nodes, d); cell blocks (cell type, connectivity (n_cells, n_en)); region tags on cells, facets, edges, vertices |
| 3 | Reference element | cell type keys from the mesh | K̂ topology: vertex/facet/edge enumerations, orientation rules (C-3), cell dimension and measure |
| 4 | Interpolation | K̂ + requested family, degree, continuity | shape values N̂(ξ) and reference derivatives ∇̂N̂(ξ), ∇̂∇̂N̂(ξ) where the family supports it; DOF functionals (values; derivatives for Hermite); completeness degree p; continuity class |
| 5 | Quadrature | K̂ + required exactness order m | points and weights {ξ_q, w_q}, q = 1…n_qp, with Σ w_q = |K̂| |
| 6 | Mapping | X_e (element node coords), N̂, ∇̂N̂ at ξ_q | x(ξ_q); Jacobians J_q = ∂x/∂ξ (n_qp, d, d); det J_q; physical gradients ∇N = J⁻ᵀ∇̂N̂ (n_qp, n_en, d); on facets: outward normal n_q, surface Jacobian |
| 7 | Element | mapping batch + DOF signature + section geometry + injected Theory/Material/State | the *kinematic batch*: {N, ∇N, (∇∇N), x_q, w_q·detJ_q, n_q} plus element DOF index arrays — handed to the theory; receives back integrand blocks and packages them as contributions |
| 8 | Theory | kinematic batch + constitutive + material + state | generalized strain operators B_q built by its kinematic operator; integrand blocks per (kind, role), e.g. Σ_q Bᵀ_q D_q B_q w_q detJ_q |
| 9 | Constitutive model | generalized strain batch E_q, material properties at x_q, state views | tangent batch D_q (n_qp, n_ε, n_ε); stress batch Σ_q (n_qp, n_ε) when a solution/state exists; trial-state writes; optional energy density ψ_q |
| 10 | Material | property key, evaluation points x_q, optional temperature T_q | property value arrays (n_qp,) or constants; failure if an undeclared key is requested |
| 11 | State | layout declarations; read committed / write trial | zero-copy views (n_e, n_qp, n_comp) per declared variable |
| 12 | Contribution provider | element/theory output | a stream of tuples (kind, role, row_dofs, col_dofs, dense block) — the only currency assembly understands |
| 13 | Assembly | contribution stream + sparsity pattern | global sparse operators per role: A(role) ∈ R^{N×N} (CSR); global vectors f ∈ R^N |
| 14 | Global operators | assembled A(role), f | operator handles carrying declared symmetry/definiteness metadata (§11) |
| 15 | Boundary conditions | Dirichlet data g, MPC relations, DOF partition | reduced systems: K_ff u_f = f_f − K_fc g; or transformed operators TᵀAT for MPCs; recovery map back to full vectors |
| 16 | Linear algebra | reduced operators and right-hand sides | u_f; eigenpairs (λ_i, φ_i); time-advanced states (u, v, a)_{n+1} |
| 17 | Analysis | everything above, via the template method | immutable results: field vectors keyed by (field, component), spectra, load factors, time histories, plus the run report (§13) |
| 18 | Post-processing | results + mesh + state views | recovered nodal fields, sampled curves, member diagrams, exported files, figures |

Three pipeline-level facts are contractual. First, stages 1–3 are purely declarative — no matrix exists before stage 13, so a model can be validated, serialized, and diffed cheaply. Second, stage 12 is the *only* interface assembly sees; local physics (CELL), boundary physics (FACET), nonlocal physics (PAIR), and point terms (VERTEX) are indistinguishable above it. Third, stage 9's batched signature is the single performance covenant of the framework (C-8): nothing in stages 7–12 may call stage 9 point-at-a-time.

*Design decision.* The theory (stage 8) receives *evaluated* kinematic data and never imports interpolation or mapping. This is dependency rule R2 restated as data flow: theories are testable against hand-written arrays, with no mesh in sight.

*Limitation.* The pull-based chain assumes operators can be formed from quadrature-local data plus, for PAIR, pairwise quadrature data. Genuinely global constitutive couplings that are not expressible as kernels over point pairs (e.g., some history-dependent homogenization schemes) would need a new contribution kind — the vocabulary is extensible (§10), but that extension is an ADR, not a default.

---

## SECTION 2 — Mathematical contracts, module by module

Format per module: **Purpose · Inputs · Outputs · Dependencies · Equations · Returned objects · Failure conditions.** Where a later section deepens a contract (§3–§7, §10), this section states the pipeline-level view and defers details.

### 2.1 Mesh

**Purpose.** Hold and query the discretized domain: nodes, homogeneous cell blocks, and named regions on entities of every codimension.
**Inputs.** Node array X (n_nodes, d); per block: cell type key, connectivity (n_cells, n_en); tag maps region-name → entity index sets (cells, facets, edges, vertices).
**Outputs.** Coordinate lookups; connectivity slices per block; region queries ("cells in 'lattice'", "facets on 'left_edge'" with (cell, local-facet) pairs); global facet/edge enumeration consistent with C-3.
**Dependencies.** `io/` for import; `utils/`. Nothing physics-side.
**Equations.** None — data integrity only: every cell references existing nodes; no duplicate nodes within tolerance; facet orientation derivable from cell orientation.
**Returned objects.** Immutable views; the mesh is frozen after model construction (C-5).
**Failure conditions.** MeshImportError; orphan/duplicate nodes → MeshError with node ids; a load or BC targeting an unknown region raises at model validation, never silently drops (v1 §9 policy).

### 2.2 Geometry (section geometry)

**Purpose.** Compute cross-section and thickness properties consumed by structural and plane elements.
**Inputs.** Shape parameters (b, h; r; r_o, r_i; flange/web dims); or, for a custom section, a user-supplied validated property table.
**Outputs.** A = ∫_A dA; first moments/centroid; I_y = ∫ z² dA, I_z = ∫ y² dA (about centroidal axes); polar moment I_p = I_y + I_z; torsion constant J_t (≠ I_p except circular — the API names them separately by design); shear correction κ(ν) and shear area A_s = κA; warping constant C_w; shear center. Plane: thickness t.
**Dependencies.** `utils/` only; ν enters κ(ν) as a plain number (no `materials/` import).
**Equations.** Closed forms per shape; e.g. rectangle: A = bh, I_z = bh³/12, κ = 10(1+ν)/(12+11ν) (Cowper); circle: I = πr⁴/4, J_t = I_p = πr⁴/2.
**Returned objects.** Frozen value objects; all quantities in centroidal principal axes with the axis convention documented per shape.
**Failure conditions.** Non-positive dimensions; inner radius ≥ outer; custom table missing a property an element later requests → MissingSectionError naming the property and element type.

### 2.3 Reference element

**Purpose.** Single source of truth for cell topology and orientation (C-3).
**Inputs.** Cell type key (line2/3, tri3/6, quad4/8; tet/hex names reserved).
**Outputs.** Reference vertex coordinates; dimension; measure |K̂|; facet/edge/vertex enumerations with orientation and outward-normal conventions; permutation rules for shared entities between neighboring cells.
**Dependencies.** None internal beyond `utils/`.
**Equations.** Topology tables; |K̂| values (2 for the line, 1/2 for the unit triangle, 4 for the biunit quad).
**Returned objects.** Immutable value objects; hashable (used as cache keys for tabulated interpolation/quadrature data).
**Failure conditions.** Unknown cell type key → InputValidationError listing registered types.

### 2.4 Interpolation

**Purpose.** Evaluate shape functions and their reference derivatives on K̂ for a given family, degree, and continuity class.
**Inputs.** Reference cell; family (Lagrange, Hermite); degree; evaluation points ξ (n_pts, dim).
**Outputs.** N̂ (n_pts, n_en·n_dpn) where n_dpn = DOF-functionals per node; ∇̂N̂ (n_pts, ·, dim); ∇̂∇̂N̂ where supported (Hermite line; families that cannot supply Hessians declare so); completeness degree p; continuity class ∈ {C⁰, C¹}; DOF functional list (value; value+derivative for Hermite).
**Dependencies.** `numerics/reference`.
**Equations.** Partition of unity Σ_a N̂_a = 1 and Σ_a ∇̂N̂_a = 0 (for value-functionals); Kronecker property N̂_a(ξ_b) = δ_ab (Lagrange); Hermite line: value and slope reproduced exactly at nodes — H₁(−1)=1, H₁′(−1)=0, H₂′(−1)=1, etc., cubic completeness p = 3.
**Returned objects.** Tabulated batches cached per (cell, family, degree, quadrature rule) — the tables are shared across all elements of a block (C-8).
**Failure conditions.** Unsupported (family, cell, degree) combination; Hessian requested from a family that does not provide it → the *continuity mismatch* error path of §4.

### 2.5 Quadrature

**Purpose.** Numerical integration rules on reference cells with declared exactness.
**Inputs.** Reference cell; required exactness order m (or an explicit named rule for reduced/selective integration).
**Outputs.** {ξ_q, w_q}, n_qp points; declared exactness m.
**Dependencies.** `numerics/reference`.
**Equations.** ∫_K̂ f dξ ≈ Σ_q w_q f(ξ_q), exact for all polynomials of order ≤ m; Σ_q w_q = |K̂|. Rule selection default for stiffness-type integrands: m ≥ 2p for affine maps, adjusted upward for non-affine mappings; reduced rules are chosen only by explicit element policy (Timoshenko shear), never inferred.
**Returned objects.** Frozen rule value objects, hashable, cached.
**Failure conditions.** No rule of requested order for the cell → InputValidationError; a *reduced* rule requested implicitly is a contract violation (silence policy).

### 2.6 Mapping

**Purpose.** Carry reference-cell quantities to physical space.
**Inputs.** Element node coordinates X_e (n_en, d); tabulated N̂, ∇̂N̂ at ξ_q.
**Outputs.** x_q = Σ_a N̂_a(ξ_q) X_a; J_q = Σ_a X_a ⊗ ∇̂N̂_a; det J_q; J_q⁻¹; ∇N = J⁻ᵀ ∇̂N̂ (n_qp, n_en, d); integration weights w_q·det J_q; facet variants: tangent(s), outward unit normal n_q, surface Jacobian (‖J·t̂‖ on 1-D facets; ‖t₁×t₂‖ in 3-D).
**Dependencies.** `numerics/interpolation`, `numerics/reference`.
**Equations.** dΩ = det J dξ; ∇_x N = J⁻ᵀ ∇_ξ N̂; for physical Hessians (C¹ theories) the full second-derivative transformation including ∂J/∂ξ terms MUST be implemented — affine shortcuts are permitted only when the map is affine, and the mapping object declares affineness.
**Returned objects.** Batched mapping data per element block.
**Failure conditions.** det J_q ≤ 0 → DegenerateCellError carrying element id and the offending ξ_q.

### 2.7 Theory

**Purpose (pipeline view).** Convert kinematic batches + constitution into integrand blocks per (kind, role); declare fields, continuity, properties, state, locality. Full contract: §4.
**Inputs.** Kinematic batch (§1 stage 7); a ConstitutiveModel; Material; State views; for PAIR: two kinematic batches plus kernel evaluations.
**Outputs.** Dense integrand blocks per (kind, role); declarations (fields, continuity, roles, kinds, locality).
**Dependencies.** `materials/`, `kernels/`, `state/`, `numerics/operators`, `numerics/tensors`, `utils/`. Never interpolation, mapping, elements (rule R2).
**Equations.** Exemplar (local elasticity, CELL, STIFFNESS): K_e = Σ_q Bᵀ_q D_q B_q w_q det J_q. All others in §4.
**Returned objects.** Blocks in the element's DOF ordering (C-2); symmetric where the role demands (C-7).
**Failure conditions.** Missing material property (names property + material); continuity requirement exceeds interpolation capability (raised at model build, not at assembly); state layout not allocated.

### 2.8 Constitutive model

**Purpose (pipeline view).** Batched generalized stress and tangent from generalized strain, properties, and state. Full contract: §5.
**Inputs.** E_q (n_e, n_qp, n_ε); property arrays; state views; optionally T_q, time increment.
**Outputs.** Σ_q, D_q = ∂Σ/∂E, trial-state writes, optional ψ_q.
**Dependencies.** `materials/`, `state/`, `numerics/tensors`, `kernels/` (for kernel-bearing decorators).
**Equations.** §5.
**Failure conditions.** Required property undefined; state layout mismatch; non-finite outputs → PhysicsError with element block id.

### 2.9 Material

**Purpose (pipeline view).** Validated property store; spatial and temperature variation. Full contract: §6.
**Inputs/Outputs.** key × (x_q, T_q) → value array (n_qp,) or scalar.
**Failure conditions.** Unknown key (lists defined keys); bound violations at construction.

### 2.10 State

**Purpose (pipeline view).** Trial/committed banks of quadrature-point variables. Full contract: §7.
**Inputs/Outputs.** Layout declarations in; zero-copy views out; commit/revert transitions.
**Failure conditions.** Reading trial before any write; committing with non-finite entries (debug); layout/schema mismatch on restart.

### 2.11 Element

**Purpose (pipeline view).** Compose §§2.3–2.10 into a contribution provider. Full contract: §3.
**Inputs.** Mesh slice (its nodes/cells), DOF map from the DofHandler, section geometry, injected Theory/Material, state views, load descriptions to convert.
**Outputs.** Contributions per (kind, role); recovery data (§2.19); DOF signature.
**Failure conditions.** §3.

### 2.12 Contribution provider

**Purpose.** The assembly currency. Full contract: §10.
**Inputs.** A role request.
**Outputs.** Stream of (kind, role, row_dofs int64, col_dofs int64, block float64 (len(rows), len(cols))). Vector contributions use a single dof array and a 1-D block.
**Failure conditions.** DOF index outside [0, N) → DofMappingError naming the provider.

### 2.13 Assembler

**Purpose.** Scatter contribution streams into global sparse operators using a precomputed pattern.
**Inputs.** Providers; the sparsity pattern; role to assemble.
**Outputs.** A(role) in CSR; vectors dense (N,).
**Dependencies.** `numerics/assembly` internals only (rule R1).
**Equations.** A = Σ_c P_r(c)ᵀ B_c P_c(c) — scatter-add of dense blocks through row/col index maps; duplicate (i, j) entries sum (COO semantics).
**Returned objects.** Operators tagged with role metadata (§11); the pattern is reused across repeated assembly (dynamics, Newton) — pattern identity is a contract, enabling factorization reuse.
**Failure conditions.** Block shape ≠ (len(rows), len(cols)); pattern violation (an (i,j) outside the precomputed pattern) → AssemblyError naming the provider; symmetry check per C-7 in debug.

### 2.14 Boundary conditions

**Purpose.** Describe constraints; produce the DOF partition and reduction maps. Application is elimination-based (ADR-003) to preserve symmetric definiteness.
**Inputs.** Dirichlet data (region, field, components, value or g(x)); MPC relations u_s = Σ c_i u_{m,i} + g; load cases with scale factors and time functions.
**Outputs.** Partition {F, C}; reduced system data; MPC transformation T (N × Ñ) with u = Tũ + g₀.
**Equations.** K_ff u_f = f_f − K_fc u_c with u_c = g; MPC-reduced operators Ã = TᵀAT, f̃ = Tᵀ(f − A g₀). Reactions: R = K_cf u_f + K_cc u_c − f_c.
**Returned objects.** ReducedSystem handles with the recovery map u = recover(u_f).
**Failure conditions.** Conflicting Dirichlet values on one DOF (both named); MPC cycles or a slave that is also Dirichlet-constrained → ConstraintConflictError; empty target region.

### 2.15 Linear solver

**Purpose.** Solve A x = b for sparse A with declared structure.
**Inputs.** Operator (with symmetry/definiteness metadata), right-hand side(s); solver-specific options (tolerances, preconditioner hooks, factorization caching).
**Outputs.** x; solve metadata (iterations, residual history) for the run report.
**Equations.** Direct: PA = LU, cached for repeated b (transient, buckling chain). Iterative CG: applicable iff metadata says SPD; convergence ‖b − Ax_k‖ ≤ tol_rel‖b‖ + tol_abs.
**Failure conditions.** Singular factorization → SingularMatrixError carrying suspected under-constrained (node, field, component) triples via the zero-pivot → DofHandler back-map; iterative non-convergence → ConvergenceError with the residual history attached; SPD solver invoked on non-SPD metadata → InputValidationError (the metadata is trusted and enforced, both ways).

### 2.16 Eigen solver

**Purpose.** Generalized symmetric eigenproblems for modal and buckling analysis.
**Inputs.** Operator pair; count k; shift σ; which-end selector.
**Outputs.** (λ_i, φ_i), i = 1…k, with declared normalization.
**Equations.** Modal: K φ = ω² M φ on the reduced space; returned φ MUST be mass-normalized, φᵢᵀ M φⱼ = δ_ij; f_i = √λ_i / 2π. Buckling: (K + λ K_g(σ₀)) φ = 0, solved as the generalized pair (K, −K_g); report smallest positive λ_cr and flag negative ones (load-reversal buckling) rather than discarding them.
**Failure conditions.** Semi-definite M (lumped rotational DOFs) MUST be handled by shift-invert or declared reduction — a raw failure of the underlying routine is re-raised as ConvergenceError with guidance; missing pre-stress state for buckling → ModelError.

### 2.17 Time integrator

**Purpose.** Advance (u, v, a) one step for M a + C v + K u = f(t).
**Inputs.** Operators M, C, K (pattern-identical, C-5); state (u, v, a)_n; f_{n+1}; Δt; parameters (β, γ).
**Outputs.** (u, v, a)_{n+1}; step diagnostics.
**Equations.** Newmark: u_{n+1} = u_n + Δt v_n + Δt²[(½ − β)a_n + β a_{n+1}]; v_{n+1} = v_n + Δt[(1 − γ)a_n + γ a_{n+1}]; effective system (M + γΔt C + βΔt² K) a_{n+1} = f_{n+1} − C v* − K u* with the starred predictors above. Unconditional stability iff 2β ≥ γ ≥ ½; default average acceleration (β = ¼, γ = ½), which conserves energy for undamped linear systems — a property the validation suite exploits.
**Failure conditions.** Δt ≤ 0; conditionally stable parameter choices proceed with a stability-estimate warning (ω_max Δt bound), never silently.

### 2.18 Analysis

**Purpose.** Orchestrate one run type via the template method: validate → number DOFs → assemble (roles per its recipe) → constrain → solve → package.
**Inputs.** A complete Model; analysis options (solver choices, k modes, time grid, load-case selection).
**Outputs.** Immutable result objects + the run report (§13).
**Equations.** None of its own — the defining property (physics/analysis separation, ADR-008). Buckling is the composition rule: static solve for σ₀, then GEOMETRIC_STIFFNESS assembly with that state, then the eigenproblem of §2.16.
**Failure conditions.** Incomplete model (missing material/section/BC target) → ModelError before any assembly; downstream failures propagate with stage context added.

### 2.19 Post-processing

**Purpose.** Derived quantities and export from immutable results.
**Inputs.** Results; mesh; state views; sampling geometry.
**Outputs.** Nodal fields from Gauss data; sampled curves; member diagrams (N, V, M); VTU/XDMF files; figures.
**Equations.** Recovery default: per-element least-squares extrapolation of Gauss values to nodes, then volume-weighted nodal averaging *within* regions (never across material interfaces — averaging across a bimaterial boundary is a physics error, so region awareness is contract, not option). Error-indicator hook: η_e² = ∫_{Ω_e} (σ* − σ_h)ᵀ D⁻¹ (σ* − σ_h) dΩ (Zienkiewicz–Zhu form) — defined here, consumed by future adaptivity (§15).
**Failure conditions.** Requesting a field absent from a result (lists available); sampling outside the domain → PostProcessError with the offending points.

---

## SECTION 3 — Element contract

An element (family) is a *composition rule*, not a physics owner. Everything below is normative for any element, built-in or plugin.

**E-1 DOF signature.** Declares, per node, the (field, component) unknowns it attaches, in C-2 order, including derivative-DOF functionals (Hermite: w and θ = w′ at each node). The DofHandler consumes signatures; per-node DOF counts may vary across the mesh (mixed frame–continuum models are legal).

**E-2 Reference cell and interpolation, per field.** The element names its reference cell and an interpolation family *per field* — mixed interpolation is first-class (a Timoshenko variant MAY interpolate w and θ with different families; field-consistent locking remedies live here). The element's provided continuity per field is the interpolation's continuity class.

**E-3 Quadrature policy, per term.** Default: full integration (m ≥ 2p, adjusted for non-affine maps). Deviations are explicit named policies bound to roles/terms: Timoshenko declares selective-reduced integration of the shear term. Implicit reduction is prohibited (§2.5).

**E-4 Mapping.** Obtained from `numerics/mapping` (§2.6); elements MUST NOT reimplement Jacobian algebra. Structural (closed-form) elements use length/orientation from node coordinates and document their local frame instead.

**E-5 Kinematic operators.** The element does not define B; it requests its theory's kinematic operator(s) (§4) and supplies the kinematic batch. Closed-form structural elements are the documented exception (ADR-002): their matrices *are* the discrete weak form, and they declare equivalence to the composed path in their theory-manual chapter — verification tests enforce it.

**E-6 Contributions.** For each supported (kind, role) the element emits blocks per §10. The canonical CELL set:
stiffness K_e = ∫ Bᵀ D B dΩ; mass M_e = ∫ ρ Nᵀ N dΩ (consistent; HRZ lumping as an explicit option, never default); geometric stiffness K_g,e = ∫ Gᵀ Ŝ(σ₀) G dΩ with G the displacement-gradient operator and Ŝ the stress matrix of the pre-state; body/consistent loads f_e = ∫ Nᵀ b dΩ (+ FACET tractions ∫ Nᵀ t̄ dΓ); internal force f_int,e = ∫ Bᵀ σ dΩ — present in the contract from phase 0 so nonlinearity changes internals, not signatures.

**E-7 Recovery.** Strain/stress evaluation at natural points from a given element DOF vector, returning quadrature-point batches in C-1 Voigt order plus the natural coordinates used — post-processing owns extrapolation, the element owns evaluation.

**E-8 State participation.** The element allocates nothing; it forwards state views (§7) between the global state and its theory, and reports its (n_qp per cell) so layouts can be sized.

**E-9 Continuity compatibility.** At model build: provided continuity (E-2) ≥ required continuity (theory, §4) per field, else IncompatibleContinuityError naming element type, field, theory. This check is central — it is how "strain gradient needs C¹" becomes impossible to violate silently.

**E-10 Local frame and transformation.** Structural elements define an orthonormal local frame from geometry (member axis; documented sign conventions for V and M diagrams) and a transformation T with K_global = Tᵀ K_local T; T orthonormality is asserted in debug. Continuum elements work in global axes; orthotropic *material* axes are the material frame's business (§9), not the element's.

**E-11 Error-estimation hook.** Elements MUST expose strain-energy density at quadrature points and support the recovered-vs-raw stress comparison of §2.19 — this is the entire per-element surface adaptivity needs.

**E-12 Refinement hook.** Elements declare whether they support h-refinement data transfer: a parent→children map for DOF values and a quadrature-state transfer rule (nearest-point now; L²-projection later). Declaring "unsupported" is legal; lying is not.

**E-13 Batch evaluability (GPU compatibility).** All E-5/E-6/E-7 operations MUST be expressible over element blocks with tabulated shape data shared per (cell, interpolation, quadrature) triple, no per-element Python state in the hot path, and pure-function semantics. This does not make the code GPU code; it makes the future array-backend swap (§15) a backend change instead of a rewrite.

*Design decision.* E-5 removes the historical FEM sin of baking one strain measure into every element. *Limitation.* E-13 constrains element authors (no stateful cleverness); the conformance kit (§12) tests it.

---

## SECTION 4 — Theory contract

A theory is the discrete statement of a set of governing equations, independent of discretization (rule R2). Its declaration block is:

| Declaration | Meaning |
|---|---|
| Fields | named unknowns and component counts, e.g. u (d), φ (1), T (1) |
| Continuity | per field, C⁰ or C¹, *derived* as the max requirement of the operators it composes (§8) |
| Required properties | material keys it will query (validated at model build) |
| Required state | its constitutive models' layouts (union) |
| Operators used | entries from §8 it composes |
| Produced (kind, role) set | which contribution kinds feed which global operators |
| Locality | LOCAL or PAIRWISE (PAIRWISE implies a kernel + horizon) |
| Supported analyses | which of static/modal/buckling/transient are meaningful |
| Dimensions / families | d values and element families (structural, continuum) it serves |

### 4.1 Local elasticity
Weak form: ∫_Ω ε(v) : C : ε(u) dΩ = ∫_Ω v·b dΩ + ∫_Γ v·t̄ dΓ. Fields u(d); C⁰; properties E, ν (or C), ρ; operators: symmetric gradient, Voigt; kinds/roles: CELL→{K, M, K_g, f}; all analyses; d = 1, 2 (3 by extension); both families.

### 4.2 Eringen differential (Helmholtz form)
Constitutive statement (1 − (e₀a)²∇²)σ = C : ε. For beams (Reddy 2007), equilibrium in terms of local quantities yields, e.g., EB dynamics EI w′′′′ + ρA(ẅ − (e₀a)² ẅ′′) = q − (e₀a)² q′′. Discrete consequence: **K stays local**; MASS gains a gradient term M = ∫ ρA (NᵀN + (e₀a)² N′ᵀN′) dx; GEOMETRIC and FORCE gain analogous (e₀a)²-scaled terms. Fields u; continuity C¹ for the EB variant (Hermite exists); properties + e₀a; kinds CELL; locality LOCAL; analyses all. *Documented limitation:* the cantilever/point-load paradoxes of this form are recorded in the theory manual and motivate 4.3 — never papered over.

### 4.3 Eringen integral (two-phase local/nonlocal)
σ(x) = ξ₁ C:ε(x) + ξ₂ ∫_Ω α(‖x − x′‖; e₀a) C:ε(x′) dΩ′, ξ₁ + ξ₂ = 1, ξ₁ > 0 for well-posedness (Polizzotto; Romano–Barretta). Discrete: K = ξ₁K_L + ξ₂K_NL with pair blocks K_NL^{(e,e′)} = ∫_{Ω_e}∫_{Ω_{e′}} Bᵀ(x) α(‖x−x′‖) C B(x′) dΩ′ dΩ, including self-pairs (e, e). Kinds: CELL (local phase, mass, loads) + PAIR (nonlocal phase); locality PAIRWISE with horizon = kernel support radius; properties + e₀a, ξ₂; requires a kernel (§8) and neighbor pairs; boundary normalization policy declared (§6 of ARCHITECTURE_v2's D-6). Verification anchor: ξ₂ → 0 or e₀a → 0 recovers the local solution to discretization accuracy.

### 4.4 Strain gradient
One-parameter form: ψ = ½ ε:C:ε + ½ ℓ² ∇ε ∴ C ∴ ∇ε → K = ∫ (Bᵀ D B + B_Hᵀ D_H B_H) dΩ, with B_H built from interpolation Hessians. Multi-parameter (Lam et al. l₀, l₁, l₂) is a property-set change, not a structural one. Continuity C¹ (Hermite beams now; C¹ 2-D continuum is hard — Argyris-type or mixed C⁰ reformulation is an explicit future ADR, and the theory declares which families it currently serves). Kinds CELL; roles K, M, K_g, f; verification: local limit as ℓ → 0; size-effect trends vs. published microbeam data.

### 4.5 Couple stress (modified, single length scale)
θ = ½ ∇×u; χ = ½(∇θ + ∇θᵀ); m = 2μℓ²χ; U = ½∫(σ:ε + m:χ) dΩ. Operators: curl ∘ gradient composition (Hessian-bearing → C¹ for displacement-based variants; mixed C⁰ variants future). Properties + ℓ; kinds CELL; the standard microbeam comparison set (modified-couple-stress EB/Timoshenko beams) is its validation anchor.

### 4.6 Surface elasticity (Gurtin–Murdoch, linearized)
Surface stress τ = τ₀ I_s + (λ_s + τ₀)(tr ε_s) I_s + 2(μ_s − τ₀) ε_s on Γ; adds ∫_Γ ε_s(v) : C_s : ε_s(u) dΓ to stiffness and a residual-stress load from τ₀. Operators: surface gradient ∇_s = P∇, P = I − n⊗n. Fields u; C⁰; properties μ_s, λ_s, τ₀ (ρ_s optional for surface inertia); kinds **FACET** (+ **EDGE** line tension where a surface patch has a boundary — the general theory's edge term, and the reason EDGE exists in §10); roles K, M, f. Physical payoff: size-dependent nanowire/nanobeam stiffness via surface-to-volume ratio.

### 4.7 Piezoelectricity
Fields u(d), φ(1); E = −∇φ; stress-charge form σ = C^E ε − eᵀE, D_el = e ε + κ^S E. Blocks: K_uu = ∫ B_uᵀ C^E B_u; K_uφ = ∫ B_uᵀ eᵀ B_φ; K_φφ = ∫ B_φᵀ κ^S B_φ; assembled convention (normative, sign errors here are the classic interop bug): [[K_uu, K_uφ],[K_uφᵀ, −K_φφ]]{u, φ} = {f, q}, symmetric indefinite. Kinds CELL; roles STIFFNESS(u), COUPLING(u, φ), STIFFNESS(φ), FORCE incl. charge; C⁰ both fields; analyses static/modal/transient; solver note: indefiniteness routes to direct or block solvers (§11), declared as the phase gate.

### 4.8 Thermoelasticity (sequential first)
Given T(x) (state or prior scalar solve): ε_th = α ΔT m (m the Voigt identity of normal components, reduction-aware for plane strain); σ = C(ε − ε_th) → FORCE contribution f_th = ∫ Bᵀ C ε_th dΩ. Fields u (+T when fully coupled later, adding conduction ∫ ∇v·k∇T and coupling roles); properties α (+k, c_p later); kinds CELL; temperature enters through state (§7), which is why state exists before nonlinearity does.

*Design decision.* Continuity is derived from operators, not asserted: a theory composing the Hessian operator is C¹ whether its author remembered or not. *Limitation.* The table cannot prevent a physically meaningless composition (e.g., surface elasticity on a 1-D bar has no facets of interest); theories therefore also declare dimension/family applicability, checked at model build.

---

## SECTION 5 — Constitutive model contract

The constitutive model is a *pointwise-in-space, batched-in-evaluation* map from generalized strain and state to generalized stress and tangent. It never sees shape functions, cells, or DOFs.

**Inputs.**
Generalized strain batch E_q (n_e, n_qp, n_ε) — the theory defines the composition of E (plain Voigt strain for elasticity; strain ⊕ strain gradient for §4.4; strain ⊕ curvature for §4.5; strain ⊕ electric field for §4.7). Where a theory needs raw gradients beyond the generalized strain (rare), it passes them as an explicitly named extra batch — nothing implicit.
Material property arrays evaluated at x_q (and T_q where declared).
State views per its declared layout (§7): read committed, write trial.
Optional scalars: time increment Δt, load-step pseudo-time.

**Outputs.**
Generalized stress batch Σ_q (n_e, n_qp, n_ε), work-conjugate to E (C-1 conventions make ΣᵀδE the internal-work density with no hidden factors).
Tangent batch D_q (n_e, n_qp, n_ε, n_ε) with D = ∂Σ/∂E — for history-dependent models this MUST be the *algorithmic (consistent)* tangent of the returned stress update, not the continuum tangent; quadratic Newton convergence is the acceptance test.
Updated trial state (in place, via the views).
Optional energy density ψ_q — consumed by error indicators (E-11), energy-balance diagnostics (§13), and verification.

**Consistency conditions (contractual, tested by the conformance kit §12).**
1. Work conjugacy: for hyperelastic models Σ = ∂ψ/∂E and D = ∂²ψ/∂E², hence D = Dᵀ (C-7).
2. Tangent consistency: D matches a finite-difference derivative of the stress map to O(h) verification tolerance at random states.
3. Null response: E = 0 with virgin state yields Σ = 0 — unless the model *declares* residual stress (surface τ₀ does; the declaration routes it to FORCE, § 4.6).
4. Objectivity (small-strain form): rotating inputs by Q ∈ SO(d) and material frame accordingly rotates outputs consistently (§9 provides the transforms; isotropy tests use random Q).
5. Thermodynamic admissibility: D positive definite for stable parameter ranges — auxetic ν < 0 remains PD within (−1, ½); models MAY have legitimately indefinite branches (post-peak damage, future) and then MUST declare it so solvers are chosen accordingly.
6. Locality of writes: only variables in the declared layout are touched.

**Internal variables and evolution.** Declared via the layout; evolution is whatever the update algorithm does between read-committed and write-trial — invisible to everything upstream. The two-phase nonlocal decorator is the exemplar of composition: it wraps any local law, scales its response by ξ₁ for CELL use, and exposes the ξ₂-weighted law for PAIR integrands, adding no state of its own.

*Design decision.* Defining E as theory-composed keeps one constitutive interface across all eight theories — piezo does not need a second contract, only a longer E. *Limitation.* Truly nonlocal-in-state models (state at x depending on state at x′) do not fit the pointwise map; they enter via PAIR integrands and state post-averaging instead, an accepted asymmetry.

---

## SECTION 6 — Material contract

A material is a validated, immutable record of physical properties. It computes nothing constitutive.

**Canonical property keys** (registry-extensible; a closed list is a maintenance trap): E, nu, G (if given alongside E and ν, consistency G = E/2(1+ν) is checked to tolerance), rho, alpha_thermal, eta_damping (material loss factor — Rayleigh α, β are *analysis* settings, not material data); piezoelectric e or d matrices and permittivity kappa; length scales e0a, l_sg (or l0, l1, l2), l_cs; surface parameters mu_s, lambda_s, tau0, rho_s; user keys under a namespaced prefix.

**Evaluation contract.** value(key) for constants; value(key, x_q) when the property carries a spatial law; value(key, x_q, T_q) when temperature-dependent. Return shape follows the query: scalar or (n_qp,). Vector/tensor-valued properties (e, κ, orthotropic C) return their C-1-consistent arrays plus, where anisotropic, a material frame (§9) — rotation to global happens once per element block, in the theory, not per call.

**FGM grading.** Any scalar key MAY carry a grading law: power law P(z) = (P_t − P_b)(z/h + ½)ⁿ + P_b; exponential; arbitrary user function of x. Grading composes with everything downstream — including the nonlocal decorator — because it is invisible to constitutive code beyond the evaluated arrays.

**Temperature dependence.** Tabulated (T_i, P_i) with declared interpolation, or a function; T comes from state (§7), closing the sequential thermoelastic loop of §4.8.

**Units.** Per C-6: unit-agnostic, one consistent system per model, bounds-only validation. The documentation ships a recommended nanoscale set (length nm, force nN, hence stress nN/nm² = GPa exactly) and states its time-unit consequence for dynamics; the Nondimensionalizer remains the sanctioned answer to conditioning, advised by §13 diagnostics.

**Validation rules** (raised at construction, InputValidationError with the offending value): E > 0; −1 < ν < ½ strictly (auxetics welcome; the bounds are thermodynamic, not conventional); ρ ≥ 0 (0 legal: massless members); α unrestricted in sign; length scales ≥ 0, with 0 meaning the local limit — deliberately legal because verification depends on it; ξ₁ ∈ (0, 1]; κ (permittivity) symmetric positive definite; **surface moduli MAY be negative** (first-principles results report negative μ_s for some crystal faces — Miller–Shenoy) — validation therefore checks only total-energy stability where determinable and otherwise warns rather than refuses.

*Design decision.* The negative-surface-modulus clause is the archetype of this package's philosophy: validate physics, not habit. *Limitation.* Bounds-only validation cannot catch a user mixing MPa and GPa; C-6 says so out loud, and §13's stiffness-spread diagnostic is the safety net.

---

## SECTION 7 — State variable contract

**Banks and lifecycle.** Two banks per model: **committed** s_n (last accepted equilibrium/time state) and **trial** s (working). Constitutive code reads committed and writes trial — never the reverse. Transitions: *commit* (trial becomes committed — end of a converged Newton solve or accepted time step); *revert* (trial discarded — diverged iteration, rejected step). Linear analyses degenerate gracefully: assemble, solve, write trial once, commit once.

**Standard variable registry** (names are registry keys; theories/constitutive models declare which they use): stress, strain (recovered fields, when a model elects to persist them); eps_p (plastic strain) and hardening variables; damage d ∈ [0,1]; temperature T; nonlocal averaged quantities (e.g., a nonlocal equivalent strain ε̄ written by PAIR-side averaging — the landing slot for future nonlocal damage); user variables under namespaced keys.

**History and time.** Any layout variable persists across commits by definition; a bounded ring of past committed snapshots (depth declared by the integrator/analysis) serves multi-step schemes. Long histories belong in the streaming writer (§2.19), not in memory.

**Memory layout.** Per element block, structure-of-arrays: one contiguous float64 array of shape (n_cells_block, n_qp, n_comp) per variable. Views handed to constitutive code are zero-copy slices. Allocation is layout-driven: a model whose laws declare empty layouts allocates nothing — linear elasticity pays zero bytes, by contract.

**Checkpointing.** A checkpoint is: model fingerprint (mesh hash, field specs, element-set manifest, DofHandler version), committed bank, solution vectors, time/step counters, RNG seeds, schema version. Format: HDF5-backed (XDMF-compatible) via `io/`. **Restart contract:** rebuilding the model from the same definition MUST reproduce identical DOF numbering and state layout (C-5 is what makes this a theorem rather than a hope); fingerprint mismatch → a clear refusal naming what differs, never a silent renumbered restart.

*Design decision.* Trial/commit is specified *before* any nonlinear code exists because it shapes every constitutive signature; retrofitting it is the single most expensive refactor in FEM-code history, and this document exists to not repeat it. *Limitation.* Two banks double state memory for heavily stateful models; an in-place-with-undo-log optimization is permitted later behind the same lifecycle semantics.

---

## SECTION 8 — Operator library (`numerics/operators`, amendment A1)

A **discrete operator** is a stateless recipe: given tabulated interpolation data at quadrature points (N, ∇N, ∇∇N, facet normals), produce the matrix that maps an element DOF vector to the operator's pointwise value. Theories *compose* operators; they do not implement them. Each operator declares the highest interpolation derivative it consumes — and a theory's continuity requirement (§4) is **derived** as the maximum over its operators. That derivation is the design decision that makes continuity errors structurally impossible.

| Operator | Mathematical action | Discrete object at ξ_q | Highest derivative | Consumed by |
|---|---|---|---|---|
| Gradient | s ↦ ∇s (scalar→vector) | G_q (d, n_dof) rows ∇N_a | 1 | thermoelastic flux, piezo E = −∇φ |
| Symmetric gradient | u ↦ ε = ∇_s u | B_q (n_σ, n_dof), C-1 Voigt | 1 | all elasticity-family theories |
| Divergence | u ↦ ∇·u | tr∘gradient row (1, n_dof) | 1 | volumetric splits, future mixed forms |
| Curl (2-D scalar / 3-D vector) | u ↦ ∇×u | rotation rows from ∇N | 1 | couple stress θ = ½∇×u |
| Laplacian (weak) | pairing ∫∇N_a·∇N_b | element matrix recipe | 1 | Helmholtz building block, conduction |
| Helmholtz | (I − ℓ²∇²) weak: ∫(N_aN_b + ℓ²∇N_a·∇N_b) | element matrix recipe | 1 | Eringen differential (implicit-gradient realization), future implicit-gradient damage |
| Nonlocal integral | double pairing ∫∫ Bᵀ(x) α(‖x−x′‖) C B(x′) | pair-block recipe over (e, e′) | 1 (+ kernel) | Eringen integral; future peridynamics |
| Surface gradient | u ↦ ∇_s u = P∇u, P = I − n⊗n | B_s from ∇N and n_q | 1 (+ normal) | surface elasticity |
| Second gradient (Hessian) | u ↦ ∇∇u → η | B_H from ∇∇N | 2 → **C¹** | strain gradient; couple stress (displacement-based) |
| Voigt operators | tensor↔vector maps, identity vector m, deviatoric projector P_dev, trace | constant matrices per d | 0 | every theory; thermal strain ε_th = αΔT·m |
| Transformation | frame rotations of vectors/tensors/operators | T matrices (§9) | 0 | structural elements, orthotropic frames, principal transforms |

**Contract per operator.** Inputs: the tabulated batch it declares. Output: batched matrices (n_e, n_qp, rows, n_dof) or matrix recipes for the pairing forms. Purity: no state, no mesh, no materials. Failure: requesting a derivative the interpolation cannot supply → the §4/E-9 continuity error, raised at build.

*Design decision.* New theories usually mean new *compositions*, not new operators — micropolar (§15) composes gradient + a relative-rotation operator; the library grows sublinearly with the theory count. *Limitation.* Pairing-form operators (Laplacian, Helmholtz, nonlocal) blur the operator/integrand line; they are kept here because they are physics-free recipes, and the blur is documented rather than hidden.

---

## SECTION 9 — Tensor library (`numerics/tensors`, amendment A2)

Scope: small dense tensor algebra over batched arrays (leading axes free), mechanics-free.

**Second order.** Symmetric/skew parts, trace, determinant, inverse, deviator dev A = A − ⅓(tr A)I; norms.
**Fourth order.** Identity I, symmetrizer Ī; volumetric/deviatoric projectors J = ⅓ I⊗I, K = Ī − J with the canonical isotropic form C = 3κJ + 2μK (the cleanest possible statement of isotropic elasticity, used by verification as an oracle); minor/major symmetry classification and checks.
**Products and contractions.** ⊗, symmetric products, single and double contraction A:B, C:ε.
**Voigt and Mandel.** C-1 Voigt is the *wire format* (engineering shear, kinematic/kinetic split). Internally, spectral and norm computations are performed in Mandel form (orthonormal, √2 factors) or full tensor form, because there contractions and eigenproblems are plain matrix algebra with no factor bookkeeping. Lossless converters Voigt↔Mandel↔full are provided and are the *only* sanctioned way to change representation — hand-inserted factors of 2 are the classic cross-developer bug this section exists to kill.
**Rotation.** Q ∈ SO(d): vectors v′ = Qv; second order A′ = QAQᵀ; fourth order via the 6×6 (3×3 in 2-D) Bond transformation pair M_σ, M_ε with M_ε = M_σ⁻ᵀ, so D′_voigt = M_σ D M_σᵀ stays convention-consistent.
**Invariants and spectra.** I₁, I₂, I₃; J₂, J₃; von Mises √(3J₂); batched symmetric eigendecomposition for principal values/directions (post-processing, principal-stress plots, future failure criteria).
**Anisotropy (future-facing).** Storage of anisotropic stiffness in a declared material frame + one rotation to global per element block; symmetry-class utilities (isotropic → cubic → transversely isotropic → orthotropic → triclinic) for validation and for importing first-principles (DFT) elastic tensors — the quantum-informed on-ramp of §15.

*Design decision.* One tensor module, two representations, converters as the only bridge — representation bugs become impossible to write without visibly bypassing the API. *Limitation.* Batched einsum-style algebra in pure numpy is memory-bandwidth-bound; the module is the natural first beneficiary of a future array backend, and its purity (no state) makes that swap trivial.

---

## SECTION 10 — Contribution provider: the five kinds

A contribution is (kind, role, row_dofs, col_dofs, block). Kinds are defined by the **topological dimension of the integration domain** — a definition that is complete in any d:

| Kind | Domain (codim) | Measure | Typical block | Sparsity effect |
|---|---|---|---|---|
| CELL | Ω_e (0) | dΩ = detJ dξ | square n_dof_e², or rectangular for field-coupling | standard FEM pattern |
| FACET | Γ_f (1) | dΓ (surface Jacobian) | square on the adjacent cell's DOFs (or facet-restricted) | subset of CELL pattern |
| PAIR | Ω_e × Ω_e′ | dΩ dΩ′ | rectangular n_dof_e × n_dof_e′ | fills (e, e′) couplings within the horizon — bandwidth grows with e₀a |
| EDGE | 1-D entity (codim 2) | ds along the edge | square on adjacent-cell DOFs restricted to the edge | subset pattern |
| VERTEX | a point (codim d) | none (point evaluation) | scalar/small block on the node's DOFs | diagonal-dominant additions |

In d = 2, codim-2 coincides with points: EDGE is normatively mapped to VERTEX; EDGE becomes distinct at d = 3. This keeps the vocabulary dimension-complete without dimension-special cases.

**Per-kind contract.**

**CELL.** Inputs: one kinematic batch. Assembly rule: A[r, c] += block. Required data: mapping weights w_q detJ_q. Examples: every §4 theory's K, M, K_g; body-force f; internal force.

**FACET.** Inputs: facet-restricted kinematic batch + outward normal n_q + surface Jacobian. Rule: as CELL, on the adjacent cell's DOF set. Examples: traction loads ∫Nᵀt̄ dΓ; surface-elasticity stiffness ∫ B_sᵀ C_s B_s dΓ; elastic-foundation (Robin) terms; *interior* facets are reserved (future DG/interface physics) and declared unsupported until an ADR opens them.

**PAIR.** Inputs: kinematic batches of both cells; kernel values α(‖x_q − x_q′‖) on the quadrature product; the pair list from `numerics/search` (unordered pairs within the horizon, self-pairs included). Rule: emit K_ee′ at (rows_e, cols_e′); for symmetric kernels the provider emits the transpose partner K_e′e = K_ee′ᵀ from the same evaluation — one numerical integration per unordered pair, by contract, so symmetry is exact by construction rather than by luck. Examples: Eringen-integral nonlocal phase; future peridynamic bond stiffness; nonlocal averaging writes to state.

**EDGE.** Inputs: edge-restricted kinematics + edge tangent + arc-length Jacobian. Rule: as CELL on the edge's DOFs. Examples (d = 3, future-facing but specified now): line loads along 3-D edges; Gurtin–Murdoch **line tension** where a surface patch terminates (§4.6's general form); stiffener/wire reinforcements along edges.

**VERTEX.** Inputs: a node id and its DOF map; no integration. Rule: direct add. Examples: nodal loads (FORCE), point masses (MASS), grounded springs (STIFFNESS), point dampers (DAMPING). *Design decision with reach:* nodal loads and point attachments are hereby providers like everything else — the assembler becomes the **only writer** of global operators, which collapses the special-case load path that plagues most FEM codes and makes "add a point absorber" a one-provider exercise.

**Emission discipline (all kinds).** Global int64 DOF indices; float64 blocks shaped (len(rows), len(cols)); blocks in C-2 ordering; every (i, j) touched must lie inside the precomputed sparsity pattern (the pattern is built from the same providers' declared index maps — pattern and emission derive from one source, so violation indicates a provider bug, and the assembler says whose).

*Limitation.* PAIR density: at large horizons K_NL tends toward dense; the SDS accepts this as physics (§4.3) and routes the consequence to solver selection (§11), plus §13 reports fill-in so the user sees the cost.

---

## SECTION 11 — Global operator definitions

Every operator NanoFEM may assemble, with its defining sum, declared structure, producers, and consumers. Structure metadata (symmetry, definiteness, block layout by field) travels *with* the operator; solvers trust and enforce it (§2.15).

| Operator | Definition | Structure | Produced by (kind, role) | Consumed by |
|---|---|---|---|---|
| Stiffness K | Σ ∫ Bᵀ D B (+ FACET, PAIR, VERTEX terms) | symmetric; PD after constraints for stable theories (two-phase nonlocal: PD for ξ₁ > 0, symmetric kernel) | any kind → STIFFNESS | all analyses |
| Mass M | Σ ∫ ρ Nᵀ N (+ nonlocal-differential gradient term §4.2; + VERTEX point masses; + FACET ρ_s) | symmetric PD (consistent); PSD if lumped with rotational zeros — declared | CELL/FACET/VERTEX → MASS | modal, transient |
| Geometric stiffness K_g(σ₀) | Σ ∫ Gᵀ Ŝ(σ₀) G | symmetric indefinite | CELL → GEOMETRIC_STIFFNESS, needs a pre-stress state | buckling; future nonlinear tangents |
| Damping C | αM + βK (Rayleigh, analysis-level) + VERTEX dashpots; material loss factor routes to modal damping ratios | symmetric PSD | analysis recipe + VERTEX → DAMPING | transient |
| Coupling K_ab | ∫ B_aᵀ D_ab B_b for field pair (a, b) | rectangular block; global symmetry via the declared convention (§4.7) | CELL → COUPLING(a, b) | coupled static/modal/transient |
| Constraint operators | MPC transformation T (N × Ñ), u = Tũ + g₀; Lagrange blocks [[K, Gᵀ],[G, 0]] and penalty are named future policies (ADR) | T sparse, full column rank | ConstraintHandler | assembly reduction, solvers |
| Load vector f(t) | Σ_cases γ_i λ_i(t) f_i, f_i from CELL/FACET/EDGE/VERTEX FORCE providers | vector | any kind → FORCE | all analyses |
| Residual r(u) | f_ext − f_int(u); linear specialization r = f − Ku | vector | internal-force providers (E-6) | Newton (future), diagnostics now (equilibrium check §13) |
| Jacobian ∂r/∂u | −K_T (tangent); = −K in the linear regime | symmetric where the physics is | same providers, tangent path | Newton, arc-length (future) |
| Sensitivity ∂A/∂p, pseudo-loads | direct: K(∂u/∂p) = ∂f/∂p − (∂K/∂p)u; adjoint: Kᵀλ = ∂g/∂u, dg/dp = ∂g/∂p − λᵀ(∂K/∂p)u | per parameter p | providers implementing SENSITIVITY(p) — the same protocol, one more role | optimization, inverse identification |
| Optimization operators | objective/constraint gradients assembled from adjoint solves + SENSITIVITY contributions | — | analysis/optimization recipes | topology/inverse studies (§15) |

*Design decision.* Sensitivities are *just another role*: a theory that can differentiate its stiffness with respect to e₀a emits SENSITIVITY(e₀a) blocks through the unchanged provider protocol — which is precisely how inverse identification of nonlocal parameters (a headline research goal) arrives without touching the assembler. *Limitation.* Automatic differentiation is not assumed; providers supply analytic or finite-difference parameter derivatives, and the conformance kit checks them against finite differences.

---

## SECTION 12 — Plugin interface

**Discovery.** Python entry points in fixed groups: nanofem.elements, nanofem.theories, nanofem.constitutive, nanofem.kernels, nanofem.materials (property keys + grading laws), nanofem.solvers, nanofem.analyses. On import, NanoFEM's Registry scans groups and registers keys; a model definition then references plugins by string key exactly as it references built-ins. Nothing in NanoFEM's source changes — that is the acceptance test of this section.

**Namespacing and collision.** Plugin keys MUST be namespaced ("mygroup.fractional_kernel"); registering a duplicate key is an error, never a silent override; built-in keys are reserved.

**Per-kind contract binding.** A plugin is valid iff it satisfies the corresponding SDS section: element → §3 (all thirteen clauses, including batch evaluability E-13), theory → §4 declaration block, constitutive → §5 consistency conditions, kernel → §2/§8 (evaluation, support radius, normalization policy), material extension → §6 (key registration, bounds, evaluation signature), solver → §2.15/2.16/2.17 including structure-metadata honesty, analysis → §2.18 template conformance.

**Conformance kit — the load-bearing idea.** NanoFEM ships its contract tests as an importable, parameterized suite: a plugin repository points the kit at its registered key and inherits, automatically, the element invariants (symmetry, rigid-body modes, patch test where applicable), constitutive consistency checks (tangent-vs-finite-difference, objectivity under random rotations, null response), kernel checks (normalization on unbounded domains, support radius honesty), and solver checks. "Certified against nanofem-sds vX.Y" becomes a meaningful, machine-checkable claim — this is the MOOSE-style ecosystem move, adapted to Python packaging.

**Versioning.** This SDS is versioned with the package; contracts follow SemVer (a signature-breaking change to §3–§10 is a major version); plugins declare a compatible range; deprecations ship with one minor version of overlap and a migration note.

**Worked example (prose only).** A researcher with a fractional-order attenuation kernel: implements the kernel contract (evaluate on distance batches, declare support radius and normalization policy), registers "mygroup.fractional" via an entry point in her own package, runs the kernel conformance kit, and writes a model whose Eringen-integral theory names that key. NanoFEM's repository receives zero commits; her package cites its certification; her results land in her own research/ tree with the §14 scorecard format.

*Limitation.* Entry-point discovery imports plugin modules at scan time; a broken plugin can poison startup. Mitigation is contractual: scanning isolates failures per plugin, reports them as warnings with the offending distribution named, and continues.

---

## SECTION 13 — Logging and diagnostics

Every analysis produces a **run report**: one machine-readable document (JSON alongside results) plus mirrored human-readable log lines (namespaced loggers; INFO = pipeline milestones, DEBUG = per-element detail; no print, ever). The report is part of the result, not an option — reproducibility (§14) consumes it.

**Report contents (normative minimum).**
Mesh statistics: node/cell counts per block, region inventory, quality summary (min/median scaled Jacobian, worst-element ids).
DOF statistics: per field, total N, constrained count, MPC count, reduced size Ñ.
Conditioning: a cheap one-norm condition estimate κ̂₁ (Hager–Higham style, a few solves against the cached factorization) by default; exact condition number opt-in only. Plus the free first alarm: diagonal spread max|K_ii| / min|K_ii| — when it exceeds a threshold the report *names the Nondimensionalizer* as the remedy (C-6).
Timings: per pipeline stage (numbering, assembly per role, constraint reduction, factorization, solve, recovery), wall and CPU.
Memory: peak RSS; per-operator nnz and bytes; PAIR fill-in ratio when nonlocal physics is active (§10's cost made visible).
Solver telemetry: iteration counts, full residual history arrays, eigenpair residuals ‖Kφ − λMφ‖ / ‖Kφ‖, factorization reuse counts.
Convergence history: (future Newton) per-iteration residual norms; (transient) step acceptance log.
Warnings ledger: every warning with category, message, and the mechanics ids involved (element/node/region), machine-parseable.

**Physical consistency checks (the research-grade part; each opt-outable, none silent).**
Symmetry residual ‖A − Aᵀ‖_∞ / ‖A‖_∞ per symmetric operator (C-7 witnessed, not assumed).
Rigid-body test: pre-constraint K applied to translational/rotational rigid modes ≈ 0 (opt-in; O(d²) matvecs).
Total mass: 1ᵀM1-based mass vs. Σρ_eV_e to tolerance.
Equilibrium: after every static solve, ΣR + ΣF_ext = 0 componentwise to tolerance — reactions from §2.14's recovery formula; failure here has caught more assembly bugs than any unit test in the history of the field, which is why it is default-on.
Energy balance (transient): drift of E_kin + E_int − W_ext relative to peak energy, reported per step; the undamped average-acceleration case MUST hold it near machine precision (§2.17), so drift is diagnostic signal.
Spectral sanity (modal/buckling): count of near-zero/negative eigenvalues vs. expected rigid modes; negative λ_cr flagged with interpretation.

*Design decision.* Diagnostics are specified as *contract*, not as logging garnish, because in research software the run report is evidence: referees ask "how do you know the solver converged," and the answer is a file. *Limitation.* Condition estimation costs solves; the default estimator is deliberately cheap and labeled an estimate — precision costs are opt-in.

---

## SECTION 14 — Validation framework (`research/validation/`, amendment A3)

Distinct from `tests/` by question and by clock: tests ask *is the code right* in seconds and gate merges; validation asks *is the physics right against external evidence* in minutes-to-hours and gates releases and papers (nightly + pre-release runs).

**Taxonomy (directories).** beam/ · frame/ · plane/ · plate/ · shell/ · nonlocal/ · gradient/ · auxetic/ · multiphysics/ — plate/ and shell/ are declared forward slots (no plate/shell theory exists yet; empty directories carry a README pointing at the roadmap, consistent with the "tree is the roadmap" policy).

**The validation-case contract.** Every case provides: metadata (id, domain, theory keys, references with DOIs); a deterministic model builder (C-5); named quantities of interest with extractors (tip deflection, ω_n, λ_cr, ν_eff, …); evidence at one or more tiers; acceptance criteria per QoI per tier; and a generated scorecard (JSON + human summary) feeding an aggregated validation matrix (theory × case × tier status) published in the docs.

**Evidence tiers and acceptance.**
**T1 — Analytical.** Closed forms. Acceptance: machine-precision agreement where the discretization is exact (Hermite EB under nodal loads vs. PL³/3EI); otherwise agreement consistent with discretization error *plus a measured convergence rate* — a right answer at a wrong rate fails.
**T2 — Published literature.** Digitized tables/curves with provenance (source figure/table, digitization method). Acceptance: within stated precision of the source, typically ≤ 1–2 %. Anchor set: Reddy (2007) nonlocal beam deflection/frequency/buckling vs. e₀a; two-phase integral benchmarks (Romano–Barretta line); modified-couple-stress and strain-gradient microbeam size effects (Lam et al.); Gibson–Ashby / Masters–Evans re-entrant honeycomb ν_eff for auxetic/; piezoelectric bimorph actuator deflection for multiphysics/.
**T3 — Commercial FEM.** Reference values from Abaqus/ANSYS runs, stored as *data plus the archived input decks that produced them* (outputs and decks are ours to store; the software is not). Acceptance: within cross-code agreement typical for the problem class, stated per case.
**T4 — Experimental.** Measured data with uncertainty. Acceptance: prediction within the reported experimental band; where parameters were fitted (e₀a from tests), the fit and the validation MUST use disjoint data — the anti-circularity clause.

**Relation to the rest of the system.** A fast T1 subset is mirrored into `tests/verification/` (CI); the full framework runs under `research-nightly.yml`; each `research/papers/<slug>` manifest references validation-case ids rather than duplicating them, so a published figure traces to a scorecard traces to a tagged version.

*Limitation.* T3 and T4 evidence ages and licenses constrain redistribution of some datasets; provenance metadata and per-case licensing notes are therefore mandatory fields, and absent-evidence tiers are shown as gaps in the matrix, not silently skipped.

---

## SECTION 15 — Future research vision

The claim to defend: each capability below lands as new providers, theories, constitutive models, kernels, solvers, or orchestration — behind existing contracts, with zero changes to the frozen architecture. Per item: the absorbing seam, and the one honest caveat.

**Topology optimization.** Density field ρ(x) as a SpatialProperty; SIMP penalization inside a constitutive model; ∂K/∂ρ via SENSITIVITY(ρ) contributions (§11); the optimization loop is `analysis/optimization` orchestration; and the density *filter* is literally a kernel convolution — `kernels/` is reused verbatim. Caveat: large design-variable counts stress the adjoint bookkeeping, an orchestration problem, not an architectural one.

**Inverse design / parameter identification.** Adjoint sensitivities (§11) + optimization orchestration; flagship case: identifying e₀a and ξ₂ against MD or experimental dispersion/deflection data, with §14-T4's anti-circularity clause governing the evidence. Caveat: identifiability is a physics question the framework can expose (sensitivity magnitudes in the run report) but not solve.

**AI-assisted constitutive modelling.** A learned law is a ConstitutiveModel whose batched response delegates to a trained surrogate — the batched-array contract (C-8) is already the ML-native signature. The §5 consistency conditions become training-time constraints and test-time certification (the conformance kit runs identically on learned laws — tangent consistency via finite differences, objectivity under random rotations). Caveat: extrapolation honesty is the user's burden; the contract can require a declared validity envelope and a warning outside it.

**Physics-informed neural networks.** NanoFEM's stance is *interoperation, not reimplementation*: the operator library evaluates weak/strong residuals of candidate fields at quadrature points, and the framework exports (x_q, w_q, operator matrices) as training quadrature — NanoFEM as the residual oracle and verification referee for PINN studies. Caveat: no autodiff graph is offered; gradients of NanoFEM outputs come from §11 sensitivities, not backprop through the assembler.

**Reduced-order modelling.** Snapshot export from results → POD offline; Galerkin projection VᵀKV is a `numerics/linalg` addition; and hyper-reduction (ECSW/DEIM) *samples contributions* — the provider protocol enumerates exactly the units hyper-reduction selects, which is an unplanned but real dividend of ADR-001. Caveat: nonlocal PAIR terms complicate sampling estimators; a research topic the structure invites rather than blocks.

**GPU acceleration.** C-8 batching + tabulated shape data + stateless operators mean the hot path is already array programs over big batches; an array-backend swap inside `numerics/` (a future ADR) relocates those arrays. Caveat: PAIR assembly and sparse factorization are the hard 20 %; the honest plan is CELL-path first.

**Distributed computing.** The solver abstraction is the PETSc-shaped door (declared in v1 §1.4); providers partition naturally by element sets. Caveat: distributed assembly and ghost DOFs are genuinely new machinery inside `numerics/assembly` — architecture-compatible, effort-intensive, explicitly out of near-term scope.

**Adaptive refinement.** The pieces already named: E-11 error hooks + §2.19's ZZ indicator + E-12 state transfer + one decisive reuse — **hanging-node constraints are MultiPointConstraints**, so adaptivity's hardest bookkeeping rides the existing MPC machinery (the deal.II lesson, adopted). Caveat: mesh hierarchies and coarsening logic are a new `mesh/` responsibility to design when the ADR opens.

**Peridynamics.** Bond-based PD is PAIR contributions with an influence function that *is* a kernel, horizon queries that *are* `numerics/search`, and meshfree points that are a degenerate cell block. State-based PD adds neighborhood state — `state/` layouts already scale. Caveat: PD's boundary/surface-effect corrections are physics work; the plumbing is done.

**Micropolar / Cosserat.** Independent rotation field φ (FieldSpec — D3 built for this), relative-strain and wryness (curvature) operators added to §8, C⁰ continuity, one new physics subpackage. **Micromorphic** generalizes to a full micro-deformation tensor field (9 components in 3-D) — expensive, but only a *bigger field*, not a different architecture. Caveat: element technology for micromorphic problems (locking, stabilization) is genuine research.

**Higher-order continua generally.** The second-gradient operator, C¹ machinery, and derived-continuity rule (§8) are the general-purpose substrate; each named theory is a composition plus a property set.

**Quantum-informed constitutive models.** DFT/MD-derived elastic tensors import through §9's symmetry-class utilities into material frames; temperature- and position-dependent property tables ride §6; a multiscale handshake (constitutive response interpolated from a precomputed ab-initio database) is an ordinary ConstitutiveModel. Caveat: uncertainty propagation from ab-initio scatter is an open methods question; §14-T4 at least gives it an evidentiary home.

**Closing statement.** Fifteen capabilities, one sentence each, and not one required amending the folder tree beyond A1–A3 of §0.1. That sentence being true was the entire point of freezing the architecture before writing this document — and every future PR review is instructed to keep it true.

---

*End of the NanoFEM Software Design Specification. Implementation may begin: the phase-0 walking skeleton (a bar element exercising every contract herein, plus the import-linter rules and the conformance-kit scaffolding) is the first milestone against which this document itself gets debugged — an SDS is falsifiable, and phase 0 is its experiment.*
