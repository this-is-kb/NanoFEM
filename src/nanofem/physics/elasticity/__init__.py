"""nanofem.physics.elasticity.

Local small-strain elasticity: the phase-2 default theory (SDS 4.1).

Responsibilities
----------------
- Isotropic/orthotropic laws; plane stress/strain reductions; small-strain operator

Future modules
--------------
- local_elasticity.py
- isotropic.py (implemented: IsotropicElasticity - dim=1 axial or dim=2 plane kinematics;
  IsotropicElasticConstitutive - dim=1 axial law only)
- euler_bernoulli.py (implemented: EulerBernoulliBendingTheory, EulerBernoulliBendingConstitutive
  - 1-D bending only)
- timoshenko.py (implemented: TimoshenkoBeamTheory, TimoshenkoBeamConstitutive - 1-D bending+shear,
  selective-reduced-integration formulation)
- plane.py (implemented: PlaneStressConstitutive, PlaneStrainConstitutive - the two dim=2
  constitutive laws that pair with IsotropicElasticity(dim=2)'s kinematics)
- orthotropic.py

TODO
----
- TODO(phase-2): 3-D reduction for IsotropicElasticity; orthotropic.py
"""
