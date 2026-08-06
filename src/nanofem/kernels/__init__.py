"""nanofem.kernels.

Attenuation kernels alpha(r; length scale) for nonlocal theories (ADR-009).

Responsibilities
----------------
- Kernel ABC: evaluation, support radius, dimension-aware normalization (SDS Section 8 row:
  nonlocal)
- Boundary-truncation/renormalization policies as explicit objects

Future modules
--------------
- base.py
- standard.py
- user.py
- normalization.py

TODO
----
- TODO(phase-4b): Helmholtz family, Gaussian, bi-exponential, polynomial
"""
