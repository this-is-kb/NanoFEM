"""nanofem.io.

External format conversion; never imported by the physics core (rule R3).

Responsibilities
----------------
- meshio adapter both directions; result writer dispatch

Future modules
--------------
- meshio_adapter.py (implemented: MeshIOAdapter.to_meshio - nanofem Mesh -> meshio.Mesh, the
  write path VTKExporter needs; .read (import) remains deferred)
- writers.py

TODO
----
- TODO(future): meshio -> Mesh import path (MeshIOAdapter.read); ResultWriter dispatch
"""
