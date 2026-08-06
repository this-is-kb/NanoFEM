"""nanofem.numerics.search.

Spatial queries: KD-tree horizon searches for nonlocal pairs (moved from mesh per ADR-007).

Responsibilities
----------------
- Neighbor pairs within a kernel support radius; unordered pairs incl. self-pairs (SDS Section
  10 PAIR)

Future modules
--------------
- neighbor.py

TODO
----
- TODO(phase-4b): KD-tree pair enumeration
"""
