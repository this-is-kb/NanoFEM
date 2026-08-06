"""nanofem.numerics.assembly.

Contribution protocol, sparsity, scatter, global/reduced systems (SDS 2.12, 2.13, Section 10).

Responsibilities
----------------
- ContributionKind {CELL, FACET, PAIR, EDGE, VERTEX}; OperatorRole vocabulary (SDS Section 11)
- Assembler as the only writer of global operators

Future modules
--------------
- contributions.py
- sparsity.py
- assembler.py
- system.py

TODO
----
- TODO(phase-0.5): COO->CSR scatter with precomputed pattern
"""
