# Developer notes — phase 0

Decisions made while turning the frozen SDS into an installable skeleton.
None contradicts the SDS; two refine seam placements and are recorded here
because phase 0 exists precisely to surface them.

**N-1 — `raise NotImplementedError` instead of bare `pass`.** mypy --strict
rejects an annotated function whose body cannot satisfy its return type; a
raise satisfies any signature. Stubs therefore raise with a
`TODO(phase-N: ...)` message. Abstract methods and Protocol methods keep
docstring-only bodies (mypy permits both).

**N-2 — Declarations are not implementation.** Frozen dataclass fields,
enum vocabularies (kinds, roles, locality, continuity), the exception tree,
and the Voigt convention tables encode SDS *contracts*; they contain zero
mathematics and exist so the import-linter contracts and integrity tests
have real edges and vocabulary to check.

**N-3 — setup.cfg and requirements.txt are deliberate mirrors.**
pyproject.toml (PEP 621) is the single source of truth; the other two exist
because the phase-0 requirements list names them and legacy tooling probes
them. Both say so in their first line.

**N-4 — Seam refinement: theory field declarations.** `core.FieldSpec` sits
above `physics` in the layer order, so `Theory.fields()` returns plain
`(name, n_components)` data and the Model materializes FieldSpec instances.
SDS Section 4 wording is compatible; the alternative (physics importing
core) would violate the layering the SDS itself mandates.

**N-5 — Seam refinement: the Continuity enum lives in numerics/operators.**
Both operators (declare what they need) and interpolation (declare what they
provide) share the vocabulary, and R2 forbids physics from importing
numerics.interpolation. numerics/operators is importable by both sides and
matches the SDS Section 8 rule that continuity is *derived* from operators.

**N-6 — Tool ownership.** black formats; isort orders imports (black
profile; ruff's import rules are deliberately not enabled to avoid dueling
sorters); ruff lints (E, F, W, B, UP); mypy types; import-linter enforces
architecture. CI and pre-commit run the same commands.

**N-7 — Coverage gate deferred.** Phase 0 bodies are raises; a fail_under
threshold would measure nothing. The 90% physics-core gate activates with
phase-1 numerics (TODO in pyproject).

**Open questions for phase 0.5** — the walking skeleton should answer:
exact Contribution vector-emission convention in practice (col_dofs=None);
Assembler return type (operator handle carrying role metadata, SDS Section
11); whether ElementDofSignature needs field/component structure beyond
name strings; run-report schema first cut (SDS Section 13).

## Phase 1 notes

**N-8 — Region-level binding is canonical.** `DomainDefinition(region -> theory, material[,
section])` is the single source of assignment truth; `Cell.material_id` / `Cell.geometry_id`
are optional convenience tags on the view object and are read by nothing in numbering or
validation. Mirrors deal.II's triangulation/DoF separation and keeps requirement 2's fields
present without creating a second binding mechanism.

**N-9 — Parallel-edit incident, reconstruction, and a defect found by tests.** Early in phase
1, three modules (`utils/validation.py`, `numerics/operators/base.py`,
`numerics/reference/cell.py`) were overwritten with fresh implementations while newer
collaborator versions already existed in the tree; with no VCS history, the originals were
unrecoverable. Expected APIs were reconstructed from every call site (`require_finite` is
scalar, `require_finite_array` for arrays, `require_non_negative`, `require_positive_int`;
`OPERATOR_CATALOG` is a frozenset of SDS Section 8 name strings) and validated by the new unit
suite plus the full gate. Two consequences: (1) the repository is now under git — initial
commit tags the phase-1 state; every future phase lands as commits, never as raw overwrites;
(2) the new round-trip tests exposed a genuine pre-existing defect — `DofHandler.
import_numbering` silently dropped per-field `VariableType`/`Continuity`, so exported and
reimported numberings had different fingerprints. Fixed; input-facing `assert`s there replaced
with `DofMappingError`.

**N-10 — Phase-1 line held.** `run()` raises on every analysis; time functions beyond
`ConstantTF` raise; section property computation raises; state checkpointing raises. The
object model is complete precisely up to, and not one line past, the no-mathematics boundary.

## Phase 2 notes

**N-11 — Two reference abstractions, deliberately.** `ReferenceCell` (phase 0, keyed by mesh
cell-type *name*: `"tri6"`) and `ReferenceElement` (the geometric domain, order-independent)
both exist and are bridged by `cell_type_of_name()`. `"tri3"` and `"tri6"` are the same
*domain* with different node counts. Merging them would drag interpolation order into the
geometry layer and violate R2; `cell.py` was left untouched since mesh/quadrature/interpolation
already depend on it.

**N-12 — The facet is a role, not a dimension.** A facet is codimension-1: a vertex for a line,
an edge for a 2-D cell, a face for a 3-D cell. So `EntityType` names absolute dimensions
(vertex/edge/face/cell) and facet-ness is reported per element via `facet_type` and
`facet_entity_type`, matching deal.II's face-vs-line accessor split. In 2-D
`edge_vertex_indices == facet_vertex_indices`; they stay separate properties because 3-D
separates them.

**N-13 — Orientation modeled only where it is known.** `Orientation` is {FORWARD, REVERSED},
which is complete for 1-D facets and therefore for every 2-D cell. The rotation/reflection
group of 2-D faces is *not* guessed ahead of the volume elements; `facet_orientations()` raises
for facet arity > 2 and the enum extends when tets/hexes land. Same discipline as N-4/N-5:
declare what is known, leave a named seam for what is not.

**N-14 — Every validation rule has a test that trips it.** Each rule in `validate()` is proven
to fire by a deliberately corrupted subclass (inward normal, duplicate vertices, bad index,
mixed arity, duplicate facet, repeated vertex, wrong count, dimension disagreement, non-finite
coordinates, measure mismatch, non-positive measure). An unexercised validation rule is an
untrusted one. Reference-layer coverage: 93-100%; the remaining misses are the 3-D
`NotImplementedError` branches, which is the intended state.

**N-15 — Version drift fixed.** `pyproject.toml` said 0.0.1 while `__version__` said 0.1.0
(pre-existing). Both are now 0.2.0. Worth a future CI check that the two agree.

## Phase 3 notes

**N-16 — The phase boundary is one matrix inverse.** The spec asked for partition-of-unity,
Kronecker, and linear-independence *verification* while forbidding shape functions, which reads
as a contradiction. It is not: all three reduce to properties of the polynomial space and the
DOF functionals. Build the generalized Vandermonde `M[k,j] = l_k(m_j)`; unisolvence is `M`
square and full rank; Kronecker is duality (how the basis is *defined*), so what can fail is
existence, which unisolvence settles; partition of unity is the constant lying in the space.
`M^-1 = C^T` is precisely the shape-function coefficient matrix, so this phase builds `M` and
never inverts it. Phase 4 inverts it. `ShapeFunctions` (phase-0 skeleton) was preserved
untouched in `base.py` as that seam; `Interpolation` sits beside it so the boundary reads in
one file.

**N-17 — Monomial calculus is not shape-function differentiation.** `DofFunctional.
apply_to_monomial` differentiates monomials (`D^a x^b`, elementary calculus on an exponent
tuple) to fill `M` for Hermite. No nodal basis exists at this layer to differentiate. Without
it, Hermite could not have its linear independence verified at all, which the spec required of
every family.

**N-18 — Node ordering follows the reference element, not gmsh.** `tri6` node 3 is the midpoint
of reference edge 0 = `(V1, V2)` at `(0.5, 0.5)`; gmsh numbers node 3 as the midpoint of
`(V0, V1)`. Ours is derived from the phase-2 topology, which the SDS makes the single source of
truth; adopting gmsh's would embed a second convention in the geometry core. Translation is the
mesh I/O adapter's job (P5). A regression test pins ours and states the divergence, so the
adapter's author meets the fact rather than discovering it.

**N-19 — The Hermite triangle is refused, with the reason.** The 10-dof cubic Hermite triangle
is unisolvent and the framework would build it, but it is only C0: an edge normal derivative is
quadratic (three conditions) while only two endpoint normal derivatives are shared. Labelling
it C1 would be false; labelling it C0 would ship a Hermite element that delivers nothing
Hermite is chosen for. C1 on simplices needs Argyris (quintic, 21 dofs) or an HCT macroelement.
Related: the BFS quadrilateral is C1 only on rectangular meshes, and says so on the class -
continuity is a property of the element *and its mesh*, not the element alone.

**N-20 — Placeholders are blocked structurally, not by scheduling.** Serendipity needs a
hand-specified `S_k` monomial set (no dimension rule yet); hierarchical needs moment dofs (an
integration rule = quadrature); spectral needs Gauss-Lobatto-Legendre nodes, which *are*
quadrature points, so that family cannot precede the quadrature layer. Each records its blocker
in `BLOCKED_BY`. The spectral motivation is already measurable:
`unisolvence_condition_number()` grows monotonically with order on equispaced nodes (tested) -
the Runge phenomenon, which is exactly why GLL nodes exist.

**N-21 — Finding: the cell registry has quad8 but not quad9.** `REFERENCE_CELLS` (phase 0)
holds the 8-node *serendipity* quad, while Lagrange order 2 on a quad needs `quad9`, which is
absent; `line4`, `tri10`, `quad16` likewise. Nothing is broken (nothing meshes those cells
yet), and the entries should land with the families that consume them - `quad8` with
serendipity. Asserted by a test so the gap stays visible rather than surprising a later phase.
`cell.py` was left untouched: mesh, quadrature, and interpolation already depend on it.

**N-22 — Every reachable validation rule has a test that trips it** (the N-14 discipline).
Interpolation-layer coverage 96-100%; the seven remaining misses in `base.py` are verified
defensive: degenerate-segment guards, an ABC `cell_type` default both families shadow, a
completeness branch unreachable because the degree is computed rather than declared, and three
guards that are unreachable for point functionals but protect the future moment-based family.
Notable among the trip tests: three *collinear* nodes on a triangle are distinct yet not
unisolvent (the Vandermonde's eta column vanishes) - rank deficiency without duplication, which
the duplicate-signature check cannot catch.

## Phase 4 notes

**N-23 — The seam was filled, not bypassed.** `ShapeFunctionFamily` implements the phase-0
`ShapeFunctions` ABC (SDS 2.4) rather than declaring a parallel contract, and the stubs
`LagrangeLine2` / `HermiteBeamLine2` are removed because `LagrangeShapeFunctions` covers them
generically - which is what those placeholders were placeholders for. One wrinkle recorded:
`ShapeFunctions.cell()` returns a `ReferenceCell`, a mesh-*name* record, a signature drafted in
phase 0 before phase 2 established `ReferenceElement` as the geometric domain. It is honored for
contract compatibility; `basis.reference_element` is the semantically correct accessor.

**N-24 — N-21 closed.** `cell()` needs `quad9`, `line4`, `tri10`, `quad16`, which phase 3 found
missing from `REFERENCE_CELLS` under the rule "entries land with the family that consumes them".
This is that family, so they were registered. The phase-3 test that asserted the gap was
rewritten to assert its closure rather than deleted. `quad8` stays registered and unused - the
serendipity cell, still waiting.

**N-25 — A verification that routes both sides through the same code proves nothing.** The
natural reproduction check `sum_i M[i,j] N_i(x) = m_j(x)` expands to `table C^T M`, and since
`C^T M = I` it collapses to `table` *identically* - it would pass for any tabulation, correct or
not, silently measuring only the accuracy of the solve. Two consequences, both applied: (1)
`verify_polynomial_reproduction` evaluates its right-hand side with `_naive_monomial_values`, a
deliberately independent power-product loop; (2) `verify_derivative_consistency` uses central
finite differences rather than the analytic identity, which is vacuous for derivatives by the
same collapse. A test makes it concrete: a family whose derivatives are scaled by 1.5 passes
Kronecker *and* reproduction, and only the finite-difference check catches it.

**N-26 — `verify_symmetry` is a structural guard, stated as such.** Both `H[a,b]` and `H[b,a]`
come from the same multi-index tabulation, so Hessian symmetry cannot fail here. It is kept to
pin the invariant for a future implementation that composes directional derivatives, and the
docstring says so. The independent confirmation is the finite-difference Hessian, whose two
routes to a mixed partial are computed separately.

**N-27 — The construction is family-agnostic, and that is the finding.** Nothing in `C = M^-T`
mentions Lagrange or Hermite; all family information lives in Sigma, which phase 3 captured. So
the concrete classes are thin - they validate the family and add its idiomatic verification
(classical `N_i(x_j) = delta_ij`; the Hermite value/slope pattern). The consequence worth
banking: **a new family needs zero shape-function code.** Serendipity and spectral get their
bases free once their `Interpolation` exists and is unisolvent.

**N-28 — Two bugs the toolchain caught that tests would not have.** (1) `black` reformatted an
f-string into a multi-line replacement field, which is Python 3.12+ syntax while the project
targets 3.11; `ruff` caught it. (2) mypy caught that `points: list[list[float]]` contradicted
the documented contract that a single point may be passed flat - the annotation was a lie, now
stated once as the `PointsLike` alias. Also: an exact-match `str.replace` silently no-op'd
because black had already rewrapped the target line. Assert on every scripted edit.

## Phase 5 notes

**N-29 — Absolute tolerances on det J are wrong, and for this package dangerously so.** The
obvious degeneracy check - "is the measure scaling near zero" - encodes an assumed element size,
because the scaling has units of length^d. A 1 nm triangle in SI units has an area scaling near
1e-18 and the first implementation rejected it as degenerate, in a library built for nanoscale
mechanics. The criterion is now `sigma_min / sigma_max` of J, which is dimensionless and
invariant under the uniform scaling that separates those cases. A test sweeps five decades of
element size and asserts identical validity, quality, and aspect ratio. This is the same
conditioning concern the architecture's `Nondimensionalizer` exists for, showing up early.

**N-30 — det(J^T J) squares the condition number; use singular values.** A rank-deficient map
leaves a Gram determinant near machine epsilon and a measure scaling near its *square root*,
about 1e-8 - far above any threshold a reader would think to write. The collinear triangle in the
tests demonstrates it. Same root cause, second instance: `inverse_jacobian` originally built
`(J^T J)^-1 J^T`, the textbook pseudo-inverse formula, and a merely awkward Jacobian produced a
numerically singular metric that leaked `LinAlgError` out of a geometry query. Both now go
through SVD.

**N-31 — Embedded elements were designed in, not deferred.** The Jacobian is
(embedding, topological), which is non-square for a bar in a plane - and the roadmap needs
exactly that for `Truss2D` and `Frame2D`. Writing every formula through the pseudo-inverse gives
one code path, since `J^+ = J^-1` when square. Where the mathematics genuinely does not extend,
the layer refuses with the reason instead of inventing: a tall Jacobian has no signed determinant
(a bar in a plane has no handedness), and the physical Hessian of a surface field is not
determined by the field alone (the second fundamental form enters).

**N-32 — Affineness is a property of the geometry, not the element type.** A `Q1` quadrilateral
is bilinear in general but affine when it is a parallelogram; a `P2` triangle with mid-side nodes
at the midpoints is affine despite a quadratic basis. Both are tested. `AffineMapping` derives
`(A, b)` by least squares and *verifies the fit is exact*, so the parallelogram rule falls out as
a residual rather than being hard-coded - and the error names `IsoparametricMapping` as the
answer.

**N-33 — A failed solve is not evidence about the element.** A diverging Newton iterate can
wander somewhere the map genuinely is degenerate; propagating that reports an element defect and
sends the user to inspect a sound mesh. `inverse_map` translates it. Related and tested:
`inverse_map` may return coordinates *outside* the reference cell, correctly - the map is a
polynomial defined beyond the cell, and containment is the caller's question
(`ReferenceElement.contains`), the same split deal.II's `transform_real_to_unit_cell` makes.

**N-34 — A test that passed against broken code.** The transposed-push-forward trip test
originally used a triangle whose Jacobian happened to be *symmetric*, which makes `J^T J^-1`
collapse to the identity - so the deliberately misindexed contraction was accidentally correct
and the test passed. Fixed with a non-symmetric Jacobian, and the requirement is now stated in
the test. A trip test that does not trip is worse than no test: it certifies the bug.

**N-35 — Scripted edits keep missing after black rewraps.** Third occurrence (see N-28). The
asserts caught every one, which is the point, but exact multi-line matching against
already-formatted source is brittle by construction. Line-indexed replacement with anchor asserts
is the reliable form.

## Phase 6 notes

**N-36 - Symmetry is reported, not imposed.** The first `verify()` rejected an anisotropic tensor
product (degree 5 in one axis, 1 in the other) because it is not invariant under the square's full
dihedral group - but that rule is perfectly legitimate, and swapping the axes swaps two unequal
factors, so it *cannot* be symmetric. Imposing symmetry was rejecting a valid rule. The fix
mirrors `has_positive_weights`: `declares_symmetry` (the claim), `invariant_symmetries` (the
subgroup actually respected, computed from the points), `is_symmetric` (whether they coincide),
and `verify_symmetry` fails only a rule that *claims* symmetry it lacks. Orbits are then built
from the invariant subgroup, so an anisotropic rule's orbits are the true ones. The symmetry
group itself is derived from vertex permutations rather than hard-coded per cell, so the
tetrahedron and hexahedron will fall out of the same loop.

**N-37 - Overtight precision assertions corrected to what the arithmetic delivers.** Two of my
own claims were tighter than machine precision allows. (1) The Dunavant barycentric sum: I claimed
`a + 2b = 1` "to the last bit", but recovering barycentric coordinates from the stored cartesian
points costs half an epsilon, so the real bound is <=0.5 eps (about 1.1e-16), not 1e-16 - deriving
`b` from `a` removes the *transcription* error, not the round-trip one, and the docstring now says
so. (2) Gauss on `exp` over `[-1,1]`: I asserted machine precision by order 9 (5 points), but it
reaches 8.2e-10 there and needs 8 points for <1e-14. Both the test and the claim were corrected to
the measured behaviour. A test asserting more precision than the arithmetic can deliver is a
latent failure, not a strong check.

**N-38 - The leaf constraint drove the module layout, not the reverse.** Quadrature must import
nothing from interpolation, because the spectral family (phase 3, blocked) has GLL nodes that
*are* quadrature points - so `interpolation` will import `quadrature`, and the reverse dependency
would make both packages unimportable. This is why `moments.py` writes its own monomial evaluation
and closed-form integrals instead of borrowing the phase-4 tabulator, and why `symmetry.py`
re-derives its affine fit instead of importing the phase-5 mapping (six lines against a forbidden
dependency). Proven by a subprocess import check and a source scan, not just asserted. Same
seam-filling discipline as phase 4: the phase-0 `QuadratureRule`/`QuadratureFactory` declarations
were filled, the `cell_name: str` draft reconciled to a `ReferenceElement` now that phase 2 owns
the domain.

## Phase 7 notes

**N-39 - `utils/exceptions.py`'s own docstring had already gone stale.** It states "no other
module defines exception types," but `numerics/mapping/errors.py` and
`numerics/quadrature/errors.py` both already define local exception families rooted at
`NanoFEMError`, and neither is listed in the central file. The precedent actually being followed
by the codebase, not the one written down, is the one to match: phase 7 adds
`tensors/errors.py` and `operators/errors.py` as local files, leaving `utils/exceptions.py`
untouched, exactly as phases 5 and 6 did. Documentation drift like this is itself worth noting -
the docstring should eventually be corrected, but that is a phase-7 finding, not a phase-7 change,
since touching it wasn't this phase's job.

**N-40 - Two packages built entirely of functions forced a real adaptation of the verification
pattern.** Every phase from 2 through 6 hung `verify()`/`is_valid()` off one rich, stateful class
and trip-tested by subclassing to break an invariant. `tensors` and `operators`' free functions
(the second-order/fourth-order algebra, the Voigt/Mandel converters) have no such class, matching
SDS's own "stateless recipe" language for operators. Verification for `tensors` therefore had to
become module-level (`verify_tensor_library()`), with trip tests constructing a deliberately
malformed array directly instead of subclassing. `operators`' recipes *do* each get a thin
`DiscreteOperator` subclass, so class-based `verify()`/`is_valid()` and subclass-style trip
testing still applied there - the split is not "tensors vs. operators" but "free function vs.
class," and it is worth remembering that the testing discipline itself needs a variant for a
phase built without a natural object.

**N-41 - `M_epsilon = M_sigma^-T` was made true by construction, not by two formulas.** The
naive approach - deriving the strain Bond matrix independently from the kinematic-Voigt basis,
the way `bond_matrix_stress` is built from the kinetic-Voigt basis - would leave two
hand-written formulas that are supposed to agree but are not forced to. Computing
`bond_matrix_strain` as `numpy.linalg.inv(bond_matrix_stress(q)).T` instead means the SDS Section 9
identity is a property of the code, not a claim checked after the fact. The same reasoning that
put `M_epsilon = M_sigma^-T` in the SDS as a design decision applies one level down: a single
formula plus a mechanical relationship beats two formulas plus a test.

**N-42 - The DOF axis stayed unflattened on purpose, and it costs something.** Every rank-2+
operator (`symmetric_gradient`, `divergence`, `curl`, `surface_gradient`) returns
`(n_qp, rows, n_fun, dim)` rather than the flattened `(rows, n_dof)` shape SDS Section 8's own
notation writes. Flattening requires a DOF ordering choice (node-major vs. component-major),
and that choice belongs to the not-yet-built `elements/` layer, which is the thing that actually
owns a DOF signature (SDS E-1). Committing to node-major now, before anything consumes it, would
be exactly the kind of ahead-of-a-consumer decision phase 0's audit column ("existing interfaces
touched: none") is designed to catch on review. The cost is real, though: every test and the
example script has to know the unflattened convention and contract it by hand with `einsum`
rather than a plain matrix-vector product, and `elements/` will need to reconcile this shape with
SDS Section 8's literal notation when it lands.

**N-43 - `nonlocal_integral` is blocked by two different layers, and the import-linter contract
proves it structurally rather than just narratively.** `kernels/` and `numerics/search/` are
both still stubs, which is the scheduling reason; but `kernels/` also sits *above* `numerics` in
the layer contract in `pyproject.toml`, which means `numerics/operators` could not import a
concrete `Kernel` even if one existed today. Recording both reasons matters: a future contributor
who fills in `kernels/` before touching `numerics/search/` might otherwise expect
`nonlocal_integral` to become buildable early: it is blocked on both fronts independently, and
the layer-contract half of the block does not go away until an ADR revisits where kernel
evaluation is allowed to be called from.

## Phase 8 notes

**N-44 - The `dim=1` isotropic-oracle degeneracy is real, not a test trick.**
`tensors.fourth_order.isotropic_stiffness(kappa, mu, dim=1)` reduces to exactly `kappa`
because its deviatoric projector `K = I - J` is built from a `dim x dim` identity contracted
against a rank-1 volumetric projector `J`; at `dim=1`, `J` *is* the identity, so `K` is
identically zero and the shear term drops out algebraically, leaving `C = 1*kappa*J = kappa`.
`test_bar_verification.py` exploits this as a genuine independent cross-check for the
constitutive tangent - the oracle was never touched by, and has no dependency on,
`IsotropicElasticConstitutive`'s own `sigma = E eps` implementation - but the reduction itself
is a property of the general 3-D formula at a specific dimension, not a coincidence engineered
for the test.

**N-45 - The composed-path verification test multiplies by area outside the pipeline, on
purpose.** SDS Section 1's pipeline (mapping -> interpolation -> quadrature -> operators ->
constitutive) carries no notion of cross-sectional area; a continuum `Theory` integrates its
weak form over the domain it is handed, which for `IsotropicElasticity` is the bar's length
only. Folding `area` into the quadrature loop inside `_composed_stiffness` would have made the
test *look* like it was proving something about the general pipeline that the pipeline does not
actually know - so `test_bar_verification.py` applies `area` as an explicit final
multiplication, matching how `Bar.local_stiffness()` itself treats it (an element-layer
constant, not a theory-layer one).

**N-46 - `Model` needed new read-only accessors because `core` cannot import `constraints` or
`elements`.** The layer contract in `pyproject.toml` places `core` below `constraints` and
`elements`; `elements/factory.py` and `analysis/static.py` sit above `core` and need to read a
built `Model`'s domains, materials, sections, and Dirichlet BCs to do their jobs. Rather than
grow `Model` a bespoke method per future consumer, five accessors were added that mirror the
existing `theories` property exactly (`dict[str, T]` copies, or a `ModelError`-on-miss lookup
for `load_case`) - the smallest surface that let two new upward layers read the model without
`core` importing anything new downward.

**N-47 - The two test-loop edits were surgical, not blanket relaxations.** Both
`test_materials_geometry.py`'s "every section still raises" loop and
`test_model_and_analysis.py`'s "every analysis still raises `NotImplementedError`" loop
existed specifically to catch phase drift - a class quietly becoming real without a
corresponding test update. Excluding `CircularSection` and `LinearStaticAnalysis` by name
(rather than, say, wrapping the whole loop in a broader `try/except`) keeps that drift
detector live for every class that is *still* a stub: `RectangularSection` and
`HollowCircularSection` still raise, and the loop still catches it if one of them
stops doing so silently.

**N-48 - The full verification gate caught a v0.7.0 gap, not a v0.8.0 one.**
`test_package_integrity.py::test_every_package_states_responsibilities_and_todo` failed on
`numerics.math` and `numerics.tensors`, both untouched by this phase's file list. Their
`__init__.py` docstrings had a "Future modules" section listing implemented items but no
"TODO" heading, a gap introduced when v0.7.0 filled those packages' stubs. Fixed in place
(added a one-line `TODO(phase-N): ...` naming the next real gap in each) rather than deferred,
since a green gate is the acceptance bar this project holds itself to, regardless of which
phase introduced the regression.

## Phase 9 notes

**N-49 - "Phase 9" here is a coincidence of two independent numbering schemes, not a
collision.** This file's own "Phase N notes" convention tracks the *version* a note belongs to
(Phase 7 notes = v0.7.0, Phase 8 notes = v0.8.0); the user's separate Stage-3 directive
independently calls this same increment (element integration) "Phase 9." The two happen to
agree here (this is v0.9.0), but that is luck, not a rule - `0 Roadmap/PROJECT_PLAN.md` already
documents why the charter's own phase numbers stopped tracking package versions after Phase 6,
and this is the same situation one level down.

**N-50 - The DOF flatten really is a bare `reshape`, and that fact is worth distrusting once.**
It is tempting to assume node-major-then-component flattening needs a transpose or an
interleaving `einsum` - it does not, precisely because `symmetric_gradient_matrix` already
returns its trailing axes in `(n_fun, dim)` order (node, then component), which is the same
order a C-contiguous array's last two axes merge in under `reshape`. Had the operator returned
`(dim, n_fun)` instead (component-major), the same flatten would silently produce the *wrong*
DOF order with no shape error to catch it - the correctness of `_flatten_b_matrix` depends on an
invariant of `numerics/operators` that this module does not itself enforce. Worth remembering if
`symmetric_gradient_matrix`'s axis order is ever "cleaned up" without checking every consumer.

**N-51 - `np.kron(A, I_dim)` is the same block-expansion trick used twice, not two designs.**
`local_mass`'s `rho * kron(mass_term(...), I_dim)` and `local_body_force`'s
`kron(shape_integral, density)` look like different operations (one produces a matrix, one a
vector) but are the identical node-major-then-component block expansion `_flatten_b_matrix`
also relies on - `kron`'s second-operand axis is always the per-node component axis. Any future
`ContinuumElement` capability that produces per-node/per-component data (damping, a consistent
nodal traction, geometric stiffness) should reach for `kron` against the same convention rather
than inventing a fourth ordering scheme.

**N-52 - Building the mass matrix eagerly and gating only `rho`'s lookup was a compromise, not
the "obviously right" design.** The plan's original sketch cached `local_stiffness`/`local_mass`
as `functools.cached_property`, which reads nicely but silently turns them into *attributes*
(`element.local_stiffness`, no parens) - inconsistent with `Bar.local_stiffness()`, a plain
method, and with every test file that already calls it that way. Rewritten to plain methods with
manual caching (`self._k_e` set eagerly in `__init__`, since `E` is always required regardless of
`MASS`; `self._m_e` set lazily on first `local_mass()` call, since `rho` is only required if
`MASS` is actually assembled) - more code than one `@cached_property` line, but it keeps every
element's public surface a set of callables, not a mix of methods and properties a caller has to
remember which is which for.

## Phase 10 notes

**N-53 - The composed-path oracle caught a real bug before it was ever shipped, not after.**
Building `EulerBernoulliBeam`'s verification test the same way `Bar`'s was built (raw
`HermiteShapeFunctions.hessian()` -> `AffineMapping.physical_hessian()` ->
`second_gradient_tensor()` -> assemble `K`) reproduced a stiffness matrix off by exactly the
Jacobian `J` on every rotation-associated row/column and `J^2` on the rotation-rotation term -
not noise, an exact ratio, confirmed by direct numerical comparison against the closed form
before any test file existed. Root cause: `HermiteInterpolation`'s "derivative" DOFs are
`dw/dxi` (reference-coordinate), because that is what the family's own unisolvence construction
is defined against, but this element's actual global DOF is `theta = dw/dx` (physical rotation).
Nothing about that mismatch is visible from either piece's own tests in isolation - `Hermite`'s
tests check its reference-space math is internally consistent, `AffineMapping`'s tests check its
pull-forward math is internally consistent - the mismatch only exists at the seam between "what
a DOF's derivative order means" and "what a physical global DOF means," which is exactly the kind
of thing ADR-002/E-5's composed-path equivalence requirement exists to catch. This is the
strongest evidence yet, in this project's own history, for why that requirement is not
bureaucratic overhead.

**N-54 - The rigid-body null-space test almost shipped with the wrong null vector.** The first
draft checked `K @ [0,1,0,1] ~= 0` for "rigid rotation," modeled on how `[1,0,1,0]` (both nodes
translated by the same amount, no rotation) is obviously a rigid translation. But a beam rotating
rigidly about node 1 does not leave node 2's transverse position unchanged - to first order,
node 2 sweeps through `w2 = L * theta`, so the correct null vector is `[0, 1, L, 1]`, not
`[0, 1, 0, 1]`. The wrong vector was caught by hand-computing `K @ [0,1,0,1]` and getting a
visibly nonzero result before writing it into a test, not by the test failing after the fact -
worth recording because the "obvious-looking" analog of an already-correct check (translation)
is not automatically correct itself, and beam/plate rigid-body modes in general are a place this
kind of sign/coupling error recurs (the same caution will apply to `Frame2D`'s and any future
plate element's rigid-body checks).

**N-55 - The bending constitutive law's functional identity to the axial law was tempting to
exploit, and it would have been a bad trade.** `EulerBernoulliBendingConstitutive`'s
`M_per_I = E * kappa` and `IsotropicElasticConstitutive`'s `sigma = E * eps` are the same code
shape - broadcast multiply, broadcast tangent. Parameterizing one class by "what the generalized
strain physically means" and reusing it for both would have satisfied "avoid duplicated code"
literally at the cost of a class whose name, docstring, and error messages are all specific to
axial elasticity silently also meaning curvature/moment. Chose the ~10 lines of duplication;
`docs/design/BEAM_ELEMENT.md` Section 3 records the reasoning in full since this exact tension
(a later reviewer proposing to "simplify" by merging them) is likely to recur.

## Phase 11 notes

**N-56 - A memorized closed-form formula was wrong, and only numerical verification caught
it before it shipped.** The natural plan for `TimoshenkoBeam` was to mirror `Bar`/
`EulerBernoulliBeam`: use the widely-known "exact" Timoshenko stiffness matrix, parametrized
by `Phi = 12EI/(GA_sL^2)`, recalled from general FEM knowledge. Before writing any production
code, this was checked against a from-scratch selective-reduced-integration (SRI) composed-path
calculation - the shear-rigid limit gave a clean, exact `0.75` ratio discrepancy, not noise.
The root cause: the "exact" formula belongs to a *different* element formulation entirely
(ODE-derived cubic-`w`/quadratic-`theta` shape functions), not the SRI-linear element SDS
clause E-3 actually names ("Timoshenko declares selective-reduced integration of the shear
term"). Building the ODE-derived shape-function family would have meant a genuinely new
`Interpolation` subfamily - exactly the kind of new abstraction this development phase is
meant to avoid - so the SRI formulation was the right choice on architectural grounds
independent of the discovery, but the discovery is what caught a formula that would otherwise
have been transcribed confidently and wrongly. Recorded here as the strongest evidence yet for
why ADR-002/SDS E-5's composed-path equivalence requirement is load-bearing, not bureaucratic:
this is the second time in two beam elements it has caught a real, non-obvious error (N-53 was
the first, for `EulerBernoulliBeam`'s Hermite reference-derivative scaling).

**N-57 - Mesh convergence, not single-element exactness, is the right correctness criterion
for this specific formulation - and that had to be verified, not assumed.** `Bar` and
`EulerBernoulliBeam` are both exact for a single element under a tip load, which set an
implicit expectation that `TimoshenkoBeam` would be too. It is not: a single SRI element gives
only 75% of the exact cantilever tip deflection. This was initially mistaken for a bug (further
investigation chased a rank-deficiency theory, then an ill-conditioning theory, before a
mesh-refinement sweep showed clean monotonic convergence - 0.751 -> 0.938 -> 0.984 -> 0.996 ->
0.999 -> 0.9998 for 1/2/4/8/16/32 elements). The lesson generalizes: a new element's
"correctness criterion" (single-element exactness vs. mesh convergence vs. something else
entirely) is a property of its specific formulation, not something to assume by analogy with
the last element built, however similar the two look on the surface.

**N-58 - The Timoshenko constitutive law is the first genuinely multi-component one, and it
exposed that `ConstitutiveModel`'s contract was already ready for it.** Every constitutive law
before this one (`IsotropicElasticConstitutive`, `EulerBernoulliBendingConstitutive`) had
`response_components() == 1` - a broadcast scalar problem. `TimoshenkoBeamConstitutive` needed
a real `(..., 2, 2)` tangent with a non-trivial (if diagonal) structure. No change to
`physics/base.py`'s `ConstitutiveModel` ABC was needed - `respond_batch`'s declared shape
contract (`strains: (..., n_eps)`, `tangent: (..., n_eps, n_eps)`) already generalized
correctly; only the *implementation* needed the `np.stack`/explicit-diagonal-assignment
pattern instead of the broadcast-scalar one. Worth recording because it confirms the phase-1
contract design (`docs/design/...` era) anticipated multi-component generalized strains
correctly, years before any theory actually needed one.

## Phase 12 notes

**N-59 - Extending `IsotropicElasticity` to `dim=2`, rather than writing a parallel Theory
class, is the first real payoff of the Theory/ConstitutiveModel split.** Plane stress and plane
strain differ only in their constitutive law, not their kinematics - both use the same
2-component displacement field and the same `symmetric_gradient` strain measure. Recognizing
that let the fix be a two-line change to an existing, already-`dim`-parametrized class (whose
own docstring had named exactly this gap since v0.8.0) instead of a new `PlaneElasticityTheory`
duplicating `IsotropicElasticity`'s kinematics declarations. This is worth recording as the
concrete first instance of what the project's own current directive asks for - "the
constitutive law can later be replaced... without modifying the solver architecture" - since a
future `EringenDifferentialConstitutive` slots into exactly the same seam
`PlaneStressConstitutive`/`PlaneStrainConstitutive` just did.

**N-60 - `symmetric_gradient_matrix` needed zero changes for 2-D, because `VOIGT_ORDER[2]`
was already there.** It would have been easy to assume the v0.7.0 operator library was built
and verified only against the 1-D case every consumer so far (`Bar`, `EulerBernoulliBeam`,
`TimoshenkoBeam`) had actually exercised, and that extending to 2-D might need new operator
work. Checking first (reading `numerics/tensors/conventions.py`'s `VOIGT_ORDER` table, then
running the real operator against a genuine 2-D case before writing any new physics code)
confirmed the dimension-generic design already worked. Recorded because "confirm the generic
code path actually generalizes, rather than assume it because the type hints say `dim: int`"
is a cheap check that would have caught a real gap had one existed.

**N-61 - The rigid-body null-space tolerance needed to scale with the matrix, and the earlier
beam elements' fixed `atol` doesn't generalize.** `Bar`/`EulerBernoulliBeam`/`TimoshenkoBeam`'s
rigid-body checks all used `atol=1e-6`, which worked because their stiffness entries happened
to be at a scale where that's comfortably above floating-point noise. A plane-stress T3 with
steel-scale `E ~ 2e11` Pa has `K` entries of order `1e11` - noise on a supposedly-exact-zero
product is itself `~1e-5`, which the old fixed tolerance would have failed. The general,
correct form (`atol` scaled to `K.max()`) should probably have been the pattern from the first
beam element onward; recorded here so it is the pattern from here on, not something re-derived
per element.

**N-62 - T3 needed no new element class, and confirming that saved real duplication.** Before
writing anything, `AffineMapping`'s own fit-residual check was read to confirm a 3-node
triangle is *always* exactly affine (a geometric fact, not an approximation specific to the
test cases used) - this is why `ContinuumElement`, already hardcoded to `AffineMapping` since
v0.9.0, needed zero changes to correctly become "T3." The same check showed precisely why Q4
is different in kind, not just untested: `AffineMapping` correctly *raises* for a real
quadrilateral rather than silently computing a wrong stiffness, confirming Q4's gap is a real,
separate architecture need (`IsoparametricMapping` support) and not something a quick patch to
`ContinuumElement` could paper over.

## Phase 13 notes

**N-63 - `IsoparametricMapping` needed a fallback wire-up, not new mapping code, because it
had already been built and verified four phases earlier.** `numerics/mapping/isoparametric.py`
existed complete since v0.5.0 - Newton inversion, curvature-corrected physical Hessian, its own
`is_affine` detection - but no element had ever constructed one; `ContinuumElement` was
hardcoded to `AffineMapping` alone since v0.9.0. Q4 therefore needed exactly one change: catch
`AffineMapping`'s own `NonAffineError` and build an `IsoparametricMapping` from the same
shape-function basis the field already tabulated. Every downstream quantity
(`physical_gradient`, `volume_scale`) is read through `GeometricMapping`'s shared interface, so
nothing past the try/except needed to change or even know which concrete mapping is in play -
confirmed by `Bar`/T3's existing test suites passing unmodified after the change.

**N-64 - The constant-strain patch test passing on a strongly distorted, genuinely non-affine
quadrilateral is a theorem, not a lucky test case, and it was verified numerically before the
fallback was written.** A bilinear Q4's shape functions have partition of unity and reproduce
any linear field exactly at the nodes regardless of how irregular the quadrilateral is - this is
the classical isoparametric consistency/patch-test guarantee. Checking it on a deliberately ugly
quadrilateral (vertices `(0,0),(4,0.5),(3.2,3),(-0.3,2.4)`, nowhere near a parallelogram) before
writing any production code confirmed the patch-test energy matches the analytical value to
`rel err ~1e-16`, and the rigid-body translation/rotation residuals sit at the same
floating-point-noise scale N-61 already found for T3 - so the v0.12.0 scale-relative tolerance
convention (`atol = 1e-9 * K.max()`) carried over with no re-derivation needed.

## Phase 14 notes

**N-65 - A theory's own DOF-signature documentation and its actual global numbering had silently
diverged, for two whole versions, because nothing ever drove them through the same code path.**
`EulerBernoulliBeam`/`TimoshenkoBeam`'s `dof_signature()` has said `u.y`/`r.z` since v0.10.0 -
correct, pinned by tests, matching SDS C-2's own worked example. But `Model.field_specs()` (the
thing that actually numbers global DOFs) names every field's components positionally
(`x`,`y`,`z`,...), which silently produces `u.x`/`r.x` for a 1-component field regardless of what
the element's own signature says - invisible until this phase, because no beam theory had ever
been registered on a `Model` and driven through `DofHandler.generate()` before; every existing
beam test built `global_dofs` by hand. The lesson generalizes past this one mismatch: two pieces
of the codebase can each be internally correct and individually well-tested, and still disagree
with each other, if nothing ever actually composes them - which is exactly why wiring every
element family through the real `Model`/factory path (not just verifying each in isolation) was
worth doing as its own increment, not assumed safe because each piece already had its own tests.

**N-66 - An eager top-level re-export turned a lazy dependency into a leak, and five unrelated
tests caught it immediately.** Adding `ContinuumElement` to `elements/factory.py`'s import list
seemed harmless - it is a real, already-verified dependency of the T3/Q4 dispatch branch - but
`nanofem/__init__.py` eagerly imports `analysis.static`, which imports this factory at module
scope, which means *any* `import nanofem.anything` now transitively imported
`numerics.quadrature` too. Five independence-proof tests across unrelated `numerics` leaf
packages (`mapping`, `operators`, `quadrature` itself, `shape_functions`, `tensors`) failed
immediately on the next full gate run - each one asserting, via a subprocess, that importing
*only* its own layer does not pull in a forbidden sibling. `Bar`/`EulerBernoulliBeam`/
`TimoshenkoBeam` never had this problem (confirmed by reading their own import lists: none of the
three imports quadrature), so the fix was narrow: defer `ContinuumElement`'s import to inside the
one function that actually needs it. Worth recording because the failure was caught by tests that
have nothing to do with elements or factories at all - proof that the existing independence-proof
suite (built package by package, phase by phase, since v0.3.0) is still doing real work four
increments after the packages it protects were declared "frozen."

## Phase 15 notes

**N-67 - A 2-D reduction still needs the full 3-D tensor for anything invariant-based, and each
of the two reductions is missing a *different* component.** It would have been easy to embed the
recovered 2x2 in-plane stress/strain directly into a 3x3 with a zero third row/column and call it
done - von Mises and principal values would run without error, just silently wrong for one of the
two laws. Plane stress's zero is `sigma_zz` (by the law's own definition); plane strain's zero is
`eps_zz`. The *other* member of each pair is not zero and has its own closed form
(`eps_zz = -nu/(1-nu)(eps_xx+eps_yy)` for plane stress; `sigma_zz = nu(sigma_xx+sigma_yy)` for
plane strain) - recovered here from the in-plane state and Poisson's ratio, dispatched on which
constitutive class built the element. Verified against a uniaxial-tension closed form where both
quantities are hand-computable before trusting the general code.

**N-68 - Direct recovery, not SPR/Zienkiewicz-Zhu, was the right amount of machinery for this
increment - and the existing `recovery.py` stub's own docstring had already said so.** The
phase-0 stub named "least-squares extrapolation, then volume-weighted nodal averaging" as its
eventual scope, which reads like superconvergent patch recovery (SPR). Building that now would
be real, nontrivial numerics (patch assembly, a local least-squares solve per node) in service of
an accuracy improvement nothing yet consumes - the ZZ error indicator this package has named
since v0.1.0 is still future adaptivity work, not a Stage-3 requirement. The classical *direct*
method (evaluate the constitutive response at each quadrature point, average) is exact for T3
(constant strain, so every quadrature point already agrees) and an honest, standard approximation
for Q4 - sufficient for reporting stress/strain/von-Mises and for a future VTK export, which is
what Stage 3 actually asks for.

## Phase 16 notes

**N-69 - A frozen-sounding docstring turned out to be exactly the right place to look for the
next real gap, not a wall.** `mesh/region.py`'s docstring has said, since v0.1.0, that facet/edge
regions are "refused until then rather than half-supported" - easy to read as "this package is
closed for extension." It wasn't a design decision to leave permanently unaddressed, just an
honest statement of phase-1 scope; nothing about the mesh layer's actual architecture blocked
facet regions, and the reference layer (`facet_vertex_indices`, complete since v0.2.0) had
already anticipated exactly this need. Confirmed with the project owner before building (the
scope was large enough, and touched a package frequently described as settled, that guessing
felt like the wrong call) rather than either skipping the gap or redesigning past it.

**N-70 - The same independence-leak class of bug recurred in a second, independently-written
module, and was caught the same way.** `constraints/traction.py`'s first draft imported
`numerics.interpolation`/`numerics.mapping`/`numerics.quadrature` at module scope, exactly the
mistake N-66 already diagnosed for `elements/factory.py` one increment earlier - and it broke the
identical five tests. The fix (defer the three imports into the functions/cached builders that
actually need them) is now a recognizable pattern, not a one-off: any module reachable from
`nanofem/__init__.py`'s eager top-level re-exports (which, transitively, is nearly every module
under `elements`/`constraints`/`analysis`) needs to think about this before adding a new
`numerics.interpolation`/`mapping`/`quadrature` import, not just Add It And See. Worth recording
as a checklist item for the next such module (a future `FacetBodyForceProvider`, an
`IntegralNonlocalOperator`, ...), since it will keep recurring until `nanofem/__init__.py`'s own
eager-import design is revisited - which is out of scope for now.

**N-71 - A facet needed a genuinely new record, not a reused one, and recognizing that early
avoided a worse design.** The tempting shortcut was encoding a facet as a single integer (a
global facet id, or a packed `cell_id * max_facets + local_index`) so it could reuse `Region`'s
existing `entity_ids: tuple[int, ...]` field unchanged. Both would have worked for this phase's
tests and both would have been the wrong kind of clever: a packed integer is a magic-number
encoding the directive's own "no hard-coded numerical constants" rule warns against, and a global
facet id needs a deduplication pass across shared interior facets that boundary-only traction
loads never actually need. `FacetRegion.facets: tuple[tuple[int, int], ...]` costs one new small
dataclass and says exactly what it is.

## Phase 17 notes

**N-72 - Wrapping scipy's `cg` instead of hand-rolling the recurrence was the same call
`SparseDirectSolver` already made, and worth stating explicitly rather than re-deciding.** The
directive's own "implement a Conjugate Gradient solver" reads like an invitation to write the
recurrence by hand (the classical `alpha_k`/`beta_k`/`r_k`/`p_k` loop), and a from-scratch
implementation would demonstrate the algorithm more visibly. But `SparseDirectSolver` never
hand-rolled LU either - it wraps `scipy.sparse.linalg.spsolve` and adds NanoFEM's own
non-finite-solution diagnostic on top. Consistency with that precedent, plus "avoid unnecessary
complexity" applying as much to solver internals as to element formulations, settled it the same
way: wrap `scipy.sparse.linalg.cg`, add the Jacobi preconditioner, the true-residual convergence
tracking via callback, and the fail-loud non-convergence check.

**N-73 - The *true* residual, not scipy's internal (preconditioned) one, is what
`residual_history` needed to record, and the two are not the same number.** Preconditioned CG's
internal stopping test operates on `||M^-1(b - A x_k)||`-like quantities, which scipy does not
expose directly through `callback` - `callback(xk)` hands back the iterate, not a residual at
all. Recomputing `||b - A x_k||` explicitly inside the callback, once per iteration, costs one
sparse matrix-vector product per step (cheap relative to the solve itself) and guarantees
`residual_history` means what a caller reading "convergence monitoring" would expect: the actual
equation residual, not an internal proxy quantity that happens to correlate with it.

## Phase 18 notes

**N-74 - A module's own docstring predicted the right shape but the layer contract still had to
be checked, not assumed from the prose.** `io/meshio_adapter.py`'s docstring has said
"Bidirectional meshio <-> nanofem.Mesh conversion" since v0.0.1, which reads as license to have
`MeshIOAdapter.to_meshio` accept a `Mesh` directly - and the first draft did exactly that.
`lint-imports` rejected it immediately: `io` sits *below* `mesh` in `pyproject.toml`'s layer list,
so `io` importing `mesh.Mesh` is backwards regardless of what the docstring's prose implied. The
docstring was describing the *feature* ("convert between the two representations"), not the
*module boundary* it would have to respect - those turned out to be different questions, and only
running the actual gate (not just planning from the written intent) caught the difference. Fixed
by moving the `Mesh`-aware extraction into `postprocess.export.VTKExporter` (which legally imports
both layers) and narrowing `MeshIOAdapter` to plain-primitive geometry data - the fourth time this
session an existing docstring turned out to describe an aspiration rather than a constraint
(N-69's facet-region docstring was the inviting kind; this one was the misleading kind), which is
itself worth remembering: a docstring is evidence about intent, not a substitute for running the
check that actually enforces the contract.

## Phase 19 notes

**N-75 - A "clean, monotonic, converged" result is not the same as a "correct" result, and both
had to be checked separately before trusting the plate-with-hole benchmark.** The first
implementation (`W/a = 4`) converged beautifully under mesh refinement - smoothly, monotonically,
to a stable asymptote - which is exactly the signature a reviewer would look for as evidence of a
correct implementation. The asymptote was ~3.58, not the expected 3.0. Trusting "it converged
cleanly" alone would have shipped a benchmark that silently checked the wrong thing. What actually
resolved it was checking the physics independently: a finite-width plate's true stress
concentration factor really is higher than the infinite-plate value (Peterson's charts), and a
relatively large hole (`a/W = 0.25` at `W/a = 4`) pushes it up substantially - confirmed by
widening the plate (`W/a = 10`, `20`) and watching the converged value move back toward 3, which a
genuine bug producing a fixed wrong answer would not have done. Clean convergence is necessary but
not sufficient evidence of correctness; it has to be convergence to the right place, checked
against an independent expectation, not just internal self-consistency.

**N-76 - Mesh grading was the second thing that looked like it might be a bug and wasn't -
uniform spacing was just the wrong tool for a localized-gradient problem.** Before discovering the
finite-width issue above, an even earlier symptom was uniform-spacing meshes not approaching 3 at
all even at 64x128 elements, still visibly climbing. This is the standard, well-known reason mesh
grading exists: a stress concentration's gradient is steepest right at the feature (the hole
boundary) and decays away from it, so uniform element density wastes resolution far from where it
is needed and under-resolves where it matters. Quadratic radial grading (`t = (i/n_r)^2`) fixed it
in one change, converging to within a few percent by a much coarser element count than uniform
spacing needed. Recorded because it is the second time in two notes that "this doesn't look
right" was resolved by understanding why a numerical scheme behaves the way it does, rather than
by tweaking parameters until the number looked plausible.

## Phase 20 notes

**N-77 - The pre-approved plan assumed a constitutive-law swap; the derivation showed a load
correction was the actually-correct minimal implementation, and the plan was updated rather than
forced.** The Stage-4 kickoff plan (confirmed with the project owner) expected
`NonlocalBarTheory`/`EringenDifferentialConstitutive` built around `helmholtz_matrix`, mirroring
the "constitutive law swaps in, backbone unchanged" story the project's own directive states as
mandatory. Working the strong-form math by hand first (per the established discipline) showed
that story doesn't literally apply to a statically-determinate 1-D bar: `sigma` eliminates in
closed form from equilibrium's own first-order ODE, collapsing the entire nonlocal effect into
the load vector, with the classical bar stiffness exactly unchanged. Implementing the originally
planned `ConstitutiveModel`-based version anyway - once the math showed it wasn't what this
specific problem needed - would have been building unnecessary machinery to match a plan rather
than the physics. The general 2-D/3-D case (where equilibrium is a divergence condition with no
closed-form stress inversion) genuinely will need the mixed/`helmholtz_matrix` treatment; that
remains real, correctly-deferred future work, not abandoned.

**N-78 - A textbook "obvious" nonlocal benchmark (cantilever, tip point load) would have been a
false positive machine.** Eringen's differential model has a documented property (the
"Peddieson paradox," Peddieson/Buchanan/McNitt 2003): a concentrated end load or a spatially
uniform distributed load leaves the model's displacement field *identical* to the classical
local solution - the nonlocal parameter drops out of the strong-form ODE entirely whenever
`q''(x) = 0`. A benchmark built the way every prior Stage-3 element benchmark was (mirror the
classical cantilever-under-tip-load case) would have "passed" regardless of whether the nonlocal
load term was implemented correctly, present at all, or silently wrong - it discriminates
nothing. The sinusoidal-load, fixed-fixed benchmark was chosen specifically because `q''(x) != 0`
everywhere and because both-ends-fixed avoids a separate, genuinely unresolved literature debate
(what "nonlocal boundary condition" a free/cantilevered end needs) that a first, minimal
benchmark had no reason to wade into. The uniform-load case was kept as its own test precisely
*because* it should show zero nonlocal effect - a passing "paradox" test is itself evidence the
implementation is doing the right calculus, not a case being skipped.

## Phase 21 notes

**N-79 - Asking "can the solver actually do X end to end" surfaced a real gap that "each piece
has its own tests" had been quietly hiding.** Every element family had its own verified test
file; `elements/factory.py` had a per-family dispatch test since v0.14.0; `LinearStaticAnalysis`
had passing benchmarks. It would have been easy to read that as "the pipeline works for all four
elements." A direct, systematic check (grep every test file that calls `LinearStaticAnalysis`,
read what element family and what post-solve step each one actually exercises) found
`EulerBernoulliBeam` had *never* been solved through `Model` at all - the v0.14.0 factory test
only compares stiffness matrices, never runs a real BC+load solve - and neither `Bar` nor
`EulerBernoulliBeam` had any stress/moment recovery method, while T3/Q4 had both since
v0.14.0-v0.15.0. This is the same lesson N-65 drew from the DOF-naming mismatch, one level up:
individually-correct, individually-tested pieces can still leave a real seam untested if nothing
ever exercises the *whole chain* for a given combination - "each piece has tests" is not the same
claim as "the pipeline works for this piece," and only checking the latter directly (not
inferring it from the former) caught the gap before it was declared closed.

**N-80 - The beam curvature-recovery formula got the same two-independent-checks treatment as
N-53/N-56, and for the same reason: this exact element has already produced two sign/scaling
bugs from formulas that looked right by inspection.** Before writing
`EulerBernoulliBeam.curvature_response`, the classical cubic-Hermite curvature formula (in terms
of the *physical* rotation DOFs this element's `local_stiffness()` already uses, not the
reference-parametrized `dw/dxi` N-53 warned about) was checked against the classical cantilever
result (`M(fixed end) = P*L` for a tip load `P`, using the already-verified stiffness matrix to
solve for the tip displacement/rotation first) *and* independently against a from-scratch
finite-difference curvature of the raw Hermite polynomial - not reusing the analytic formula
being checked. Both matched immediately, so no bug was caught this time, but the point of the
discipline was never "catch a bug every time" - it is "don't find out about the bug from a user."

## Phase 22 notes

**N-81 - A verification method can be internally correct and still be structurally blind to an
entire class of bug, and only a full end-to-end solve exposed it.** Building the mixed (u, e*)
element for general 2-D Eringen differential elasticity, the first round of verification
(symmetry of `K_eu`/`K_ue`; the Schur complement `K_ue @ K_ee^-1 @ K_eu^T` matching the classical
T3 stiffness exactly at `mu=0`; a constant-strain field recovering `e* = eps(u)` exactly for any
`mu`) all passed - using `+K_ee` in the assembled block matrix, which turned out to be wrong. The
reason every one of those checks passed anyway: each one solves for `e*` *given* `u` by direct
substitution (`e* = K_ee^-1 K_eu u`), which never assembles the monolithic
`[[0,K_ue],[K_eu,?K_ee]]` matrix at all - the sign of the diagonal block is simply invisible to a
substitution-based check, correct or not. A full `Model -> LinearStaticAnalysis` solve of the
same uniaxial-tension plate T3 already used (`test_static_t3_plate_analytical.py`'s own problem)
reproduced the closed-form tip displacement with the right magnitude and the wrong sign - the
first symptom that something was off. Re-deriving the residual equations from scratch
(`K_eu.u - K_ee.e* = 0`, matched term by term against the matrix-row convention) showed the
correct diagonal block is `-K_ee` - the standard KKT/Stokes-type saddle-point sign pattern
(`[[0,B^T],[B,-C]]`), which should have been recognized from the start given the matrix's own
zero-diagonal-block structure on `u`. Recorded as the single strongest instance yet of why a
*monolithic, full-pipeline* solve is not a redundant final formality once the "real" checks pass
- it is checking something the other checks structurally cannot.

**N-82 - `e0a` already existed in the material property vocabulary since v0.1.0, and using it
instead of inventing a new key was worth pausing to check.** The first draft used a new
material property, `nonlocal_parameter`, for `mu = (e0*a)^2`. `Material`'s own canonical key
list (`materials/material.py::_BOUNDS`) already has `e0a` - "0 = local limit, deliberately
legal (SDS Section 6)" - present since the very first phase-1 object model, long before any
nonlocal theory existed to consume it. Reusing it (storing the length `e0a` itself, squaring it
internally to get `mu`) rather than adding a parallel property is a small thing, but it is
exactly the kind of "check whether this already exists before adding a new name for the same
concept" discipline the project has tried to hold since N-21 (the `quad9`/`quad8` registry gap)
- and in this instance, the existing name was chosen well before the feature existed to use it,
which is itself a small piece of evidence the phase-1 object model's vocabulary was built with
real foresight, not guessed at.

**N-83 - The layer-contract gate caught a real violation the moment it ran, precisely because
`voigt_component_names` reached for the "obvious" helper instead of the low-layer-safe one.**
`eringen_differential.py`'s first draft imported `core.fields.component_names` to build Voigt
pair names generically - the natural thing to reach for, and exactly what `IsotropicElasticity`
would use if it needed axis names computed rather than positionally defaulted. `import-linter`
failed immediately: `physics` may not import `core` (R2, physics is discretization- *and*
core-free). `EulerBernoulliBendingTheory`/`TimoshenkoBeamTheory` had already solved this same
problem by hard-coding their one or two axis literals directly rather than importing the helper
- a fix this file's first draft could have matched from the start had the precedent been
checked before writing the import, not after the gate caught it. Fixed by duplicating the three
axis-letter constant locally (`_AXES`), the same trade-off N-55 named for the bending
constitutive law: a few lines of duplication is worth it against introducing a forbidden
dependency edge.

## Phase 23 notes

**N-84 - A recalled shear/moment sign convention was wrong, and re-deriving it from the
codebase's own already-verified weak form (not from memory) caught it before any code was
written.** Building the nonlocal Euler-Bernoulli beam's strong form needed beam equilibrium's
sign (relating `M''` to `q`). The "obvious" approach - recall the standard `dV/dx=-q`,
`dM/dx=V` textbook relations - gave `M''=-q`, and combined with this codebase's own
`kappa=w''`/`M=EI*kappa` convention (no minus sign) produced a solution with the wrong sign
relative to the applied load direction. Textbook shear/moment sign conventions are notoriously
inconsistent across sources (differing on which direction is "positive shear," which way `q` is
assumed positive, etc.) - exactly the kind of ambiguity that bit N-53/N-54 for this same
element's Hermite DOF parametrization and rigid-body mode, respectively. The fix: derive
`M''=q` directly from this codebase's *own* weak form (`integral(M*delta_w'')=integral(q*delta_w)`,
which the already-verified `K=EI/L^3[...]` matrix is itself built from) via integration by
parts, rather than trusting an externally-recalled convention that was never guaranteed to
match this codebase's specific sign choices. Recorded as the third time this specific element
has required resolving a convention ambiguity by deriving from first principles/existing
verified code rather than from memory - a real pattern for beam/plate elements generally, not a
one-off.

**N-85 - N-53's fix was incomplete, and a beam element's own classical superconvergence
property is what exposed it.** `_reference_derivative_scale` (N-53, v0.10.0) was written to
correct the curvature B-matrix for Hermite's reference-vs-physical rotation DOF mismatch. Built
the nonlocal beam's load vector using the *same* shape functions' raw *values* (uncorrected) for
the classical consistent-load term, reasoning that "a shape function's value, not just its
derivative, is unaffected by which derivative order the DOF represents." That reasoning was
wrong: interpolation is `w(x) = sum_a N_a(x) d_a_ref`, and since `d_a_ref = scale_a *
d_a_physical`, the shape function that correctly multiplies a *physical* DOF is `N_a(x)*scale_a`
everywhere, not only inside a bilinear (stiffness) form - a load-vector integral is exactly as
much "a shape function used against physical DOFs" as a stiffness integral is. The bug would
have been easy to miss without a strong independent expectation to check against: a numerical
`mu=0` sanity check gave a `~6.7e-5` residual, which could plausibly be mistaken for ordinary
discretization error - except cubic Hermite beam elements have their *own* classical
superconvergence property (nodally exact `w`/`theta` for the consistent Galerkin load, the same
kind of result N-50/bar-superconvergence already relied on), so a non-machine-precision
residual at `mu=0` was recognized as a genuine red flag rather than dismissed as expected
approximation error. Fixed by applying `scale` to the shape values too; verified by the
idealized (exact-`q`-at-quadrature) check reaching `2.2e-14`, from a previously non-converging
`~1e-3` residual.

## Phase 24 notes

**N-86 - The mixed (u, e*) formulation's "e0a=0 recovers classical elasticity" claim is exact
for a single element/constant-strain field, but only mesh-*convergent* for a general 2-D field
- discovered while building the cantilever benchmark, before its tolerances were chosen.**
Building `test_nonlocal_cantilever_benchmark.py`, a direct comparison between the mixed system
at `e0a=0` and a classical FEM solve on the *same* multi-element mesh showed a large discrepancy
(185.8% on a 4x2 mesh) - alarming at first, since the element-level Schur-complement equivalence
had already been proven exactly (`test_nonlocal_continuum_element.py`). Root cause, found by
narrowing the reproduction down from a full cantilever to two triangles sharing one edge: the
element-level proof is a *local* identity (one element's own `K_ue @ K_ee^-1 @ K_eu^T` equals
its own classical stiffness), but `e*` is a globally-shared, C0-continuous nodal field - global
elimination of a shared field is **not** the sum of independent local eliminations
(`(A+B)^-1 != A^-1 + B^-1` in general), so the local identity does not, by itself, imply a
global one. It only *happens* to hold globally too when the true classical strain field is
itself continuous across element boundaries - true only in the constant-strain case
(`test_static_nonlocal_plate.py`'s uniaxial tension patch test), never true in general (a T3's
own classical strain is piecewise-constant, generically discontinuous between elements, while
`e*` cannot represent a discontinuous field at all). Verified this is a mesh-convergent, not a
per-mesh-exact, property by refining: the discrepancy shrank monotonically (185.8% -> 55.8% ->
14.7% -> 3.8% at 4x2 -> 8x4 -> 16x8 -> 32x16) - not a bug, but the well-documented, accepted
behavior of implicit-gradient-type mixed regularization models generally (the same mathematical
structure as Peerlings-style gradient-enhanced damage/plasticity), confirmed by checking the
literature's own treatment of this model class before concluding it was expected rather than a
defect. `docs/design/ERINGEN_DIFFERENTIAL_CONTINUUM.md` Section 7 records the full derivation
of *why*, and corrects every benchmark's classical-recovery claim to state mesh-convergence
explicitly rather than exactness, for any field that is not constant. Recorded as the clearest
instance yet of this project's own repeated lesson (N-53, N-56, N-81, N-84, N-85): a property
proven true in one specific, narrower case (here: single-element, or constant-strain) does not
automatically generalize to the case that matters (here: a real, multi-element, non-constant
mesh) - and the only way to find out is to actually build and check that more general case, not
assume the narrower proof extends.
