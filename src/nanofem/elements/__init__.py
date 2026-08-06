"""nanofem.elements.

Discretization only: compose numerics + physics + geometry + materials into providers (SDS
Section 3).

Responsibilities
----------------
- Element ABC honoring clauses E-1..E-13; DOF signatures (C-2)
- Structural closed-form family (ADR-002) and dimension-generic continuum family (ADR-011)

Future modules
--------------
- base.py
- factory.py (implemented: build_elements - dispatches Model domains to element instances)
- structural/ (Bar implemented; the walking skeleton)
- continuum/

TODO
----
- TODO(phase-1): continuum/ family (ADR-011); remaining structural suite
"""
