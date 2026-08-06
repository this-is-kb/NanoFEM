# NanoFEM Facet Regions + Traction Loads (v0.16.0)

**Status:** implemented and tested. Companion to `docs/design/ELEMENT_FACTORY.md` (the `Model`
plumbing this reuses) and `docs/design/ELEMENT_INTEGRATION.md` (`ContinuumBodyForceProvider`,
whose one-dimension-down pattern this mirrors).

**Scope discipline.** `TractionLoad` (already declared, SDS 2.14) gets a real
`ContributionProvider`; `mesh/region.py` gains the smallest facet-identity record a boundary
traction needs. `NeumannBC`/`RobinBC` (the separate, `Model`-level flux/BC objects) remain
declared-but-unconsumed - this phase's job was the load-case-level surface load, matching
`NodalLoad`'s existing pattern, not the whole BC vocabulary.

---

## 1. The blocker: facet regions didn't exist, and why that's a real gap, not an oversight

`mesh/region.py`'s own docstring named this explicitly: "Facet/edge regions... arrive with the
phase-2 orientation machinery... and are refused until then rather than half-supported."
`Mesh.__init__` enforced it - any `Region` with `dimension` other than `0` (node) or
`self.dimension` (cell) raised. A rigorous traction integral needs facet geometry (which two
nodes bound an edge, its arc length, its outward normal) that a node-set region cannot express.

Confirmed with the project owner before building: extend the mesh layer with the minimal facet
concept needed (recommended), rather than deferring traction loads or working around the gap
with hand-split nodal loads indefinitely (the workaround every prior 2-D test used, and still a
correct answer for the simple straight-edge case - see Section 4).

## 2. `FacetRegion`: a new record, not a `Region` variant

A facet has no single global id the way a node or cell does - two cells can share one interior
facet - but every *boundary* facet (the only kind a traction or Neumann flux needs) belongs to
exactly one cell. So `(cell_id, local_facet_index)` is a sufficient, unambiguous identity,
`local_facet_index` resolved against `ReferenceElement.facet_vertex_indices` (`numerics.reference`,
complete since v0.2.0 - no new reference-layer code needed). `FacetRegion` is a new, small,
frozen dataclass (`mesh/facet_region.py`), not a new `Region.dimension` value: facet identity
genuinely needs two integers where node/cell identity needs one, so reusing `Region.entity_ids:
tuple[int, ...]` would have meant either a lossy encoding scheme or silently accepting only
single-cell-boundary facets under a false pretense of generality. `Mesh.__init__` gained an
optional `facet_regions` parameter (default `()`, so every existing `Mesh(...)` call is
unaffected) and four small query methods (`facet_region`, `facet_region_names`,
`facets_in_region`, `facet_node_ids`) mirroring the existing node/cell region query style exactly.

## 3. `TractionLoadProvider`: one dimension down from `ContinuumBodyForceProvider`

For the T3/Q4 minimal element library (Stage 3's own scope - linear elements only), a facet is
always a 2-node line, and a P1/Q1 element's shape functions restricted to one of its facets are
*exactly* the 2-node line's own linear Lagrange basis (every other node's shape function vanishes
identically on a facet it is not part of - a standard fact about linear elements, not something
specific to this implementation). So the traction integral `integral N_a(s) t_c dS` needs no
restriction operator: build a fresh, independent `LagrangeInterpolation(LINE, 1)` on the facet's
own two nodes, map it with `AffineMapping(LINE, coords)` - already supporting a 2-node line
embedded in 2-D physical space since v0.5.0's "bar embedded in a plane" case, so the arc-length
Jacobian (`volume_scale`) needed no new mapping code - and integrate with the existing line
quadrature. `TractionLoadProvider` (`constraints/traction.py`) composes exactly these existing
pieces, the same way `ContinuumBodyForceProvider` composes cell-level pieces for a body force.

`analysis/static.py`'s load-case dispatch, previously `NodalLoad`-only, now also accepts
`TractionLoad` entries.

### The independence-leak lesson recurred, and was caught the same way

Importing `numerics.interpolation`/`numerics.mapping`/`numerics.quadrature` at `traction.py`'s
module scope broke the same five `numerics`-leaf independence tests v0.14.0's `ContinuumElement`
import already broke once (dev note N-66) - `analysis.static` is in `nanofem/__init__.py`'s eager
import chain, so any module-level heavy import there leaks into every `import nanofem.anything`.
Fixed the same way: defer the three imports into the functions/cached builders that need them
(`_facet_basis`, `_facet_quadrature`, `TractionLoadProvider.contributions`), confirmed by
re-running the exact previously-broken tests before trusting the fix.

---

## 4. Verification

Numerically checked before writing the facet integral: a uniform traction on a straight 2-node
edge integrates to the classical consistent load `traction * length / 2` at each end node -
exactly the hand-split value `test_static_t3_plate_analytical.py`/`test_postprocess_recovery.py`
already used for the same plate geometry.

`tests/unit/test_traction_load_provider.py` (5 tests): `FacetRegion` construction/validation and
its out-of-range-cell rejection; `Mesh.facet_node_ids`/`facets_in_region` resolve the right edge
correctly; `TractionLoadProvider`'s assembled block matches the hand-derived consistent load
exactly (`rtol=1e-12`); a full `LinearStaticAnalysis` solve using a real `TractionLoad` reproduces
- to floating-point precision - the identical displacement and reaction state the hand-split
`NodalLoad` version already produced, which is the real end-to-end proof that the facet integral
is wired correctly through assembly, not just correct in isolation.
