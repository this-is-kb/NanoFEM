"""nanofem.mesh.

Domain topology and geometry data: nodes, cell blocks, regions, facet tags, quality.

Responsibilities
----------------
- Immutable node/cell/region storage and queries (SDS 2.1)
- Import via io adapters; gmsh geometry builder behind a lazy import (rule R3)

Future modules
--------------
- node.py (implemented: Node)
- mesh.py (implemented: Mesh, CellBlock, Cell; region()/cells_in_region()/nodes_in_region() real)
- readers.py (import adapters: not yet implemented)
- gmsh_builder.py (lazy-imported gmsh geometry builder: not yet implemented)
- quality.py (mesh quality metrics: not yet implemented)

TODO
----
- TODO(phase-1): facet enumeration consistent with C-3
"""
