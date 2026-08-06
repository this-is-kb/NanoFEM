"""nanofem.constraints.

Constraint and load descriptions plus DOF partitioning; never mutates operators (SDS 2.14).

Responsibilities
----------------
- Dirichlet, loads, load cases, time functions, multipoint constraints
- ConstraintHandler: free/constrained partition, MPC transformation

Future modules
--------------
- dirichlet.py
- loads.py
- load_case.py
- time_functions.py
- mpc.py
- handler.py

TODO
----
- TODO(phase-0.5): partition + elimination reduction (ADR-003)
"""
