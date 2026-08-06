"""nanofem.postprocess.

Derived quantities and export from immutable results (SDS 2.19).

Responsibilities
----------------
- Gauss->nodal recovery with region-aware averaging; sampling; member diagrams; export; plots

Future modules
--------------
- recovery.py (implemented: element/nodal stress-strain recovery, principal values, von Mises,
  strain energy - plane stress/strain continuum elements only)
- export.py (implemented: VTKExporter - Mesh + point/cell data to .vtu via meshio; TimeSeriesWriter
  still a stub)
- sampling.py
- diagrams.py
- plotting.py
- pyvista_view.py

TODO
----
- TODO(future): beam diagrams; superconvergent patch recovery for adaptivity; XDMF time series
"""
