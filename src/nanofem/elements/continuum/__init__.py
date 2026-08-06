"""nanofem.elements.continuum.

Dimension-generic continuum elements parameterized by ReferenceCell (ADR-011).

Responsibilities
----------------
- Composition of interpolation, quadrature, mapping, theory, material

Future modules
--------------
- continuum.py (implemented: ContinuumElement, ContinuumBodyForceProvider - single-field
  theories; geometry via AffineMapping, falling back to IsoparametricMapping for a
  non-parallelogram Q4; verified for the 1-D bar, T3, and Q4 cases)

TODO
----
- TODO(phase-3+): 3-D continuum elements (Tet, Hex); curved (higher-order isoparametric)
  geometry support
"""
