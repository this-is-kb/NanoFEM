"""nanofem.symbolics.

Independent SymPy derivations; optional extra; NEVER imported by runtime modules (ADR-013).

Responsibilities
----------------
- Symbolic cells/shape functions/operators sharing no code with numerics (deliberate
  duplication)
- Test oracles for tests/symbolic; code generation with provenance headers

Future modules
--------------
- cells.py
- interpolation.py
- operators.py
- integration.py
- codegen.py

TODO
----
- TODO(phase-1): Hermite oracle
- NOTE: a runtime import of this package is a CI failure
"""
