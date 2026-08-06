# NanoFEM Reference Element Library

**Status:** implemented and tested (phase 2). Companion to ARCHITECTURE_v2.md, the SDS, and
OBJECT_MODEL.md. This document records the reference geometry layer: the canonical domains,
their numbering and orientation conventions, the topological relationships derived from them,
and why the module is shaped the way it is.

**Scope discipline.** This layer is purely geometric and topological. It contains no shape
functions, no quadrature, no Jacobians, no interpolation, no mapping, and no constitutive
mathematics. Those live in later phases and *consume* this layer; nothing here anticipates
their implementation beyond leaving the right seams.

---

## 1. Reference domains

| Element | Domain | Vertices | Edges | Faces | Facets | Facet shape | Measure |
|---|---|---|---|---|---|---|---|
| `ReferenceLine` | `xi ∈ [-1, 1]` | 2 | 1 | 0 | 2 | vertex | 2 |
| `ReferenceTriangle` | unit right triangle | 3 | 3 | 1 | 3 | line | 1/2 |
| `ReferenceQuadrilateral` | `[-1, 1]²` | 4 | 4 | 1 | 4 | line | 4 |
| `ReferenceTetrahedron` | *(placeholder)* | 4 | 6 | 4 | 4 | triangle | 1/6 |
| `ReferenceHexahedron` | *(placeholder)* | 8 | 12 | 6 | 6 | quadrilateral | 8 |
| `ReferencePrism` | *(placeholder)* | 6 | 9 | 5 | 5 | mixed | — |
| `ReferencePyramid` | *(placeholder)* | 5 | 8 | 5 | 5 | mixed | — |

These match the SDS C-3 conventions frozen in phase 0 and the `ReferenceCell` records already
used by the mesh layer. The bi-unit line and square with the *unit* (not bi-unit) triangle is
the standard pairing used by deal.II, MFEM, and FEniCS: it keeps tensor-product elements
symmetric about the origin while giving simplices the barycentric-friendly corner at the origin.

### Vertex numbering

```
Line                    Triangle                  Quadrilateral
                                                  
0-----------1           V2 (0,1)                  V3(-1,1)-------V2(1,1)
-1          +1          |  .                        |               |
                        |     .                     |       .       |
                        |        .                  |    (0,0)      |
                        V0------V1                   |               |
                        (0,0)   (1,0)              V0(-1,-1)------V1(1,-1)
```

Vertices are listed counterclockwise for the 2-D shapes (positive orientation), starting from
the origin corner for the triangle and from `(-1, -1)` for the quadrilateral.

### Facet numbering

**Line** — facets are the two vertices: facet 0 = vertex 0 (`xi = -1`), facet 1 = vertex 1
(`xi = +1`). Outward normals `[-1]` and `[+1]`.

**Triangle** — *facet `i` is opposite vertex `i`*, each ordered so its normal points outward:

| Facet | Vertices | Description | Outward normal |
|---|---|---|---|
| 0 | (1, 2) | hypotenuse | `(1, 1)/√2` |
| 1 | (2, 0) | left edge | `(-1, 0)` |
| 2 | (0, 1) | bottom edge | `(0, -1)` |

**Quadrilateral** — facets ordered bottom, right, top, left:

| Facet | Vertices | Description | Outward normal |
|---|---|---|---|
| 0 | (0, 1) | bottom | `(0, -1)` |
| 1 | (1, 2) | right | `(1, 0)` |
| 2 | (2, 3) | top | `(0, 1)` |
| 3 | (3, 0) | left | `(-1, 0)` |

In 2-D the edges and facets coincide (`edge_vertex_indices == facet_vertex_indices`); they are
exposed as separate properties because they diverge in 3-D, where edges are dimension-1 and
facets are dimension-2.

---

## 2. Orientation conventions

**Outward normals.** For a 2-D cell, the outward normal of edge `(a, b)` is its tangent rotated
clockwise: `t = v_b - v_a`, `n = (t_y, -t_x) / |t|`. Combined with counterclockwise vertex
ordering, this makes every facet normal point out of the cell — a property `validate()`
*checks* rather than assumes, by requiring `n_i · (c_i - c) > 0` for every facet centroid `c_i`
and cell centroid `c`. For a 1-D cell the facet is a vertex and its normal points away from the
centroid.

**Sub-entity orientation.** A shared facet is seen in opposite orders by the two cells that
share it. `Orientation` names the two possibilities for a 1-D facet — `FORWARD` (canonical) and
`REVERSED` (flipped) — and `permute_facet(i, orientation)` returns the reordered vertex tuple.
`Orientation.sign` (+1/-1) is the multiplier a tangent or normal picks up under the flip.

A vertex facet admits only `FORWARD`, so `ReferenceLine().facet_orientations()` is a
one-element tuple and requesting `REVERSED` raises. The richer orientation group of 2-D faces
(rotations × reflections, deal.II's `combined_face_orientation`) is deliberately *not* modeled
yet: guessing its encoding before the volume elements exist would be speculative. `Orientation`
extends when they arrive.

---

## 3. Topological relationships

The boundary lattice is `vertex ⊂ edge ⊂ face ⊂ cell`, and `EntityType` names the roles by
absolute dimension (vertex 0, edge 1, face 2, cell contextual). The **facet** — the
codimension-1 entity, which is what a flux, a traction, or a facet integral acts on — is a
*role*, not a fixed dimension: it is a vertex for a line, an edge for a 2-D cell, a face for a
3-D cell. It is therefore reported per element (`facet_type`, `facet_entity_type`) rather than
being an `EntityType` member. This is the same distinction deal.II draws between `face` and
`line`/`vertex` accessors.

Incidence is stored once, as vertex-index tuples per facet and per edge, and every other
relationship is derived:

- entity counts from the array shapes and incidence lengths,
- facet centroids and normals from the incidence and coordinates,
- vertex-facet incidence by inversion (each 2-D vertex lies on exactly two facets — asserted in
  the tests),
- boundary extraction as `facet_vertex_indices` itself.

---

## 4. Design rationale

**Data-driven base class.** `ReferenceElement` is an ABC whose subclasses declare only *data*:
`cell_type`, `facet_type`, `vertex_coordinates`, `facet_vertex_indices`, `edge_vertex_indices`,
`reference_measure`. Every operation — counts, centroid, bounding box, edge lengths, diameter,
tangents, normals, containment, distance, permutation, validation, serialization — is a generic
algorithm in the base class. Consequences: each concrete shape is ~30 lines of pure declaration;
a new shape gets the entire query and validation surface for free; and the conventions live in
exactly one place, which is what makes the SDS's "no module may embed its own copies" rule
enforceable. This mirrors deal.II's `ReferenceCell`/`GeometryInfo` centralization.

**Value semantics.** A reference element is *canonical per shape*: all triangles are the same
triangle. So the concrete classes are frozen dataclasses with no fields, and `__eq__`/`__hash__`
are defined once on the base by `cell_type`. `ReferenceTriangle() == ReferenceTriangle()` holds,
and reference elements work as dict keys and set members — which matters because the future
tabulation caches (shape-function tables per (cell, rule)) will key on exactly this. Coordinate
arrays are module-level singletons with `setflags(write=False)`, so the canonical geometry
cannot be mutated through a returned reference; derived arrays are freshly computed copies.

**Validation is a first-class operation, not a comment.** `validate()` checks dimension
consistency, vertex count and uniqueness, incidence validity/arity/uniqueness, outward
orientation of every normal, unit length of normals and tangents, and agreement between the
declared measure and the measure recomputed from the vertices (shoelace in 2-D). Every one of
these rules has a test that *trips* it via a deliberately corrupted subclass — an unexercised
validation rule is an untrusted one. `is_valid()` is the non-raising form.

**Two abstractions, one bridge.** `ReferenceCell` (`cell.py`, from phase 0) is a lightweight
record keyed by a mesh cell-type *name* (`"tri6"`), used by mesh/quadrature/interpolation as a
quick lookup. `ReferenceElement` is the full geometric domain, independent of interpolation
order. They are related by `cell_type_of_name()`, which strips the order suffix: `"tri3"` and
`"tri6"` are the same *domain* with different node counts. Keeping them separate is deliberate —
the reference domain does not change when you raise the polynomial order, and conflating them
would force the geometry layer to know about interpolation, violating rule R2.

**Errors.** `ReferenceElementError` → `TopologyError`, `OrientationError`, all rooted at
`NanoFEMError` so one catch clause still works. These signal an *internal* inconsistency in an
element definition (a library bug, surfaced by `validate()`), as distinct from
`InputValidationError`, which reports a bad argument from a caller (a query point of the wrong
dimension). The distinction tells a user immediately whose fault a failure is.

**Independence.** The layer imports only `numpy` and the shared exception base — nothing from
`mesh`, `core`, `physics`, or any other package. A test enforces this by running a subprocess
that imports only `nanofem.numerics.reference` and builds, queries, validates, and serializes a
triangle. This is what makes it a *foundation*: every later layer may depend on it, and it
depends on nothing that could churn.

**What is deliberately absent.** No Jacobian, no shape functions, no quadrature points, no
mapping, no interpolation. `embedding_dimension` equals `topological_dimension` here by design:
embedding a bar in a plane or a shell in 3-D is a property of the *physical* element and its
mapping, a later layer, not of the canonical domain.

---

## 5. Future extension strategy

**3-D elements.** `ReferenceTetrahedron`, `ReferenceHexahedron`, `ReferencePrism`, and
`ReferencePyramid` exist as declared placeholders: they satisfy the interface, raise
`NotImplementedError` on construction, and carry a `PROVISIONAL_TOPOLOGY` mapping recording
their intended counts and measures so the eventual implementation has a fixed target. Building
them requires four things this phase defers, each with its seam already visible:

1. **Volume measure** — `_computed_measure()` raises for `dim == 3`; needs the tetrahedral
   decomposition.
2. **Face enumeration** — `num_faces` raises for `dim == 3`; needs a `face_vertex_indices`
   declaration alongside the existing edge/facet ones.
3. **Facet normals via cross products** — `reference_normals()` raises for `dim == 3`.
4. **Face orientation groups** — `facet_orientations()` raises for facet arity > 2; `Orientation`
   grows from {FORWARD, REVERSED} to the rotation/reflection group.

**Mixed-facet shapes.** Prisms and pyramids have facets of two different shapes, which the
current single-valued `facet_type` cannot express. When they land, `facet_type` becomes
per-facet (`facet_type(i)`), with the scalar property retained where the shape is uniform. This
is recorded now so the change is a planned evolution, not a surprise.

**Higher-order geometry.** Curved (isoparametric) cells do not change the *reference* domain —
they change the mapping. Nothing in this layer needs to move for them.

**Registry.** New shapes register in `REFERENCE_ELEMENTS`; `reference_element()` and
`reference_element_for_name()` pick them up automatically, and `CellType.is_implemented` starts
reporting `True` without any edit to the enum.
