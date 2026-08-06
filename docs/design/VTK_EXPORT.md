# NanoFEM VTU Export (v0.18.0)

**Status:** implemented and tested. Companion to `docs/design/POSTPROCESSING.md` (the recovered
fields this is meant to export) and `docs/design/ELEMENT_FACTORY.md`/`TRACTION_LOADS.md` (the
`Model`-driven solves that produce them).

**Scope discipline.** `postprocess.export.VTKExporter` and `io.meshio_adapter.MeshIOAdapter` fill
in the two stubs SDS 2.19/rule R3 already declared, wrapping the existing `meshio` dependency -
the last item in the Stage-3 success criteria's own list ("export results... VTK output").
`TimeSeriesWriter` (XDMF transient export) stays a stub - transient analysis is explicitly out of
scope for Stage 3 (the directive's own "DO NOT IMPLEMENT... Dynamics" list).

---

## 1. A layer-contract violation caught before it shipped

The first draft put the nanofem-`Mesh` -> `meshio.Mesh` conversion inside
`io.meshio_adapter.MeshIOAdapter.to_meshio(mesh)`, reading naturally given the module's own
"Bidirectional meshio <-> nanofem.Mesh conversion" docstring. `lint-imports` immediately rejected
it: the layer contract places `io` *below* `mesh` (`...geometry, mesh, numerics, io, utils`), so
`io` cannot import `mesh.Mesh` at all - "higher may import lower," and `io` is lower. This
matches rule R3's actual wording ("only mesh and postprocess may import *it* [io]") - the
dependency was always meant to run the other way.

Fixed by moving the `Mesh`-aware extraction (points, one `(meshio_cell_type, connectivity)` pair
per cell block) up into `postprocess.export.VTKExporter.write()`, which legally imports both
`mesh` and `io` (it sits above both). `MeshIOAdapter` was narrowed to a genuinely mesh-agnostic
`build_meshio_mesh(points, cells)` working in plain primitives - a real architecture correction,
caught by the same import-linter gate this project has run after every increment, not a redesign
of the layer contract itself.

## 2. What each piece does

- `MeshIOAdapter.build_meshio_mesh(points, cells)`: pads points to 3-D (VTK's own convention;
  nanofem meshes are 1-D or 2-D) and builds a `meshio.Mesh` from plain arrays. `.read()` (importing
  an external mesh file) remains a stub - a separate, larger undertaking (physical-group/region
  reconstruction) than "write VTK output" needs, and would belong in `mesh/` if built later (the one
  layer above `io` that can both import it and construct a `Mesh`).
- `VTKExporter.write(path, mesh, point_data=..., cell_data=...)`: resolves each cell block's type
  through a small, explicit `{"line2": "line", "tri3": "triangle", "quad4": "quad"}` map (the
  Stage-3 minimal element library's three cell types - not a general gmsh-name bridge), validates
  point/cell data shapes against the mesh's own node/cell counts, splits a flat per-cell array
  across meshio's per-block `cell_data` structure, and writes the `.vtu` file.

---

## 3. Verification

`tests/unit/test_postprocess_export.py` (6 tests): `build_meshio_mesh` preserves geometry and pads
2-D points to 3-D correctly (checked against the mesh's own coordinates, not re-derived);
malformed point-array shapes are rejected; an unmapped cell type raises with the supported list
named; a genuine write-then-read round trip through a real `.vtu` file (not just constructing a
`meshio.Mesh` object and trusting it) confirms point data and cell data both survive intact;
wrong-shaped point/cell data are rejected before any file is written.
