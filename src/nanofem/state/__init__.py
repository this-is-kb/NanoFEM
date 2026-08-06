"""nanofem.state.

Quadrature-point state: layouts, trial/committed lifecycle, checkpointing (ADR-010, SDS Section
7).

Responsibilities
----------------
- StateLayout declarations; ModelState structure-of-arrays storage
- commit/revert transitions; restart fingerprinting (SDS C-5)

Future modules
--------------
- layout.py
- model_state.py
- quadrature_state.py
- history.py

TODO
----
- TODO(phase-3+): allocation from layouts
- TODO(future): HDF5 checkpointing via io/
"""
