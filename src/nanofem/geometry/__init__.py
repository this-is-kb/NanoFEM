"""nanofem.geometry.

Member and section geometry (NOT the meshed domain, which lives in nanofem.mesh).

Responsibilities
----------------
- CrossSection property models: A, I, J_t vs I_p, shear area, warping, centroid, shear center
  (SDS 2.2)
- PlaneGeometry thickness for 2-D continua

Future modules
--------------
- base.py
- standard.py
- custom.py
- plane.py
- section_solver.py (future ADR)

TODO
----
- TODO(phase-1): closed forms for standard shapes
- TODO(future): Saint-Venant section solver
"""
