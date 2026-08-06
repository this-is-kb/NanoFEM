"""nanofem.physics.eringen.

Eringen nonlocal elasticity: differential (Helmholtz) and integral (two-phase) forms.

Responsibilities
----------------
- Differential form: modified MASS/GEOMETRIC/FORCE operators, K stays local (SDS 4.2)
- Integral two-phase form: CELL + PAIR contributions via kernels (SDS 4.3)

Future modules
--------------
- differential.py
- integral.py
- beam_theories.py

TODO
----
- TODO(phase-4a): differential form
- TODO(phase-4b): two-phase integral form
"""
