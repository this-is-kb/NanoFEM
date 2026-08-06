"""nanofem.materials.

Validated property records only; zero constitutive mathematics (ADR-004).

Responsibilities
----------------
- Material property store with canonical keys and bounds validation (SDS Section 6)
- SpatialProperty and FGM grading laws; temperature dependence

Future modules
--------------
- material.py
- properties.py
- grading.py

TODO
----
- TODO(phase-1): property key registry + bounds
- TODO(phase-2+): grading law evaluation
"""
