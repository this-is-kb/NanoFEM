"""nanofem.numerics.

Approximation and algebra, mechanics-free by import-linter rule R1 (ADR-007).

Responsibilities
----------------
- Reference cells, interpolation, quadrature, mapping, operators, tensors
- Assembly, sparsity, linear/eigen solvers, time integration, spatial search

Future modules
--------------
- reference/
- interpolation/
- quadrature/
- mapping/
- operators/
- tensors/
- assembly/
- linalg/
- timeintegration/
- search/
- math/

TODO
----
- TODO(phase-0.5): assembler scatter path
- NOTE: importing any mechanics package here is a CI failure
"""
