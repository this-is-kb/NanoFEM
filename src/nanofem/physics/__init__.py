"""nanofem.physics.

Every theory lives here: kinematics, constitution, weak-form integrands (ADR-006).

Responsibilities
----------------
- Theory ABC: fields, continuity, roles, kinds, locality (SDS Section 4)
- ConstitutiveModel ABC: batched response contract (SDS Section 5, C-8)
- KinematicOperator ABC; discretization-free by rule R2

Future modules
--------------
- base.py
- elasticity/
- eringen/
- strain_gradient/
- couple_stress/
- surface/
- piezoelectric/
- thermoelastic/

TODO
----
- TODO(phase-2): LocalElasticity theory
- TODO(phase-4a/4b): Eringen differential/integral
"""
