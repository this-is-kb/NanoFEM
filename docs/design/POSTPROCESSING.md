# NanoFEM Post-Processing: Stress/Strain Recovery (v0.15.0)

**Status:** implemented and tested. Companion to `docs/design/PLANE_ELASTICITY.md`/
`Q4_QUADRILATERAL.md` (`ContinuumElement`, whose cached data this reuses) and
`ELEMENT_FACTORY.md` (the `Model`-driven solve this consumes).

**Scope discipline.** Plane-continuum elements (T3/Q4 under `IsotropicElasticity(dim=2)`) only -
this is the only real 2-D physics in the codebase, and Eringen nonlocal elasticity's own
constitutive laws (differential and integral) operate on exactly this kind of stress/strain
field, which is why this increment was prioritized ahead of the remaining Stage-3 gaps (Neumann
BCs, the CG solver, VTK export). Structural elements (`Bar`/`EulerBernoulliBeam`/
`TimoshenkoBeam`) are untouched - their "stress" (axial force, bending moment, shear) is already
a direct output of their own closed-form constitutive law, a different, already-solved concern.

---

## 1. What recovery means here, and what it deliberately isn't

`ContinuumElement` gained two small new methods (`quadrature_point_response`,
`measure`) reusing data it already tabulates and caches for `local_stiffness()` - no
re-derivation, no new quadrature rule, no new B-matrix. `postprocess/recovery.py` composes
that per-element data into:

1. **Element fields** (`recover_element_fields`): the quadrature-weighted average strain/stress
   over each element, as the classical *direct* recovery method - not superconvergent patch
   recovery (SPR) or a Zienkiewicz-Zhu least-squares extrapolation, both of which remain future
   work for adaptivity (the ZZ error-indicator hook this package's docstring has named since
   v0.1.0). Direct recovery is simpler, exact for T3 (constant strain, so every quadrature point
   already agrees), and an honest, standard approximation for Q4.
2. **Nodal fields** (`recover_nodal_fields`): a measure-weighted average of element fields across
   every element sharing a node - explicitly *not* automatic across an entire model, since SDS
   2.19 requires averaging never cross a material interface; the caller passes one
   material/domain's elements at a time.
3. **Strain energy** (`strain_energy`): `sum_e 0.5 (stress_e : strain_e) measure_e`, independent
   of `0.5 u^T K u` - a genuine physical cross-check, not a restatement.

Reactions were already available (`StaticResult.reactions`, walking skeleton v0.8.0) and needed
no new code here.

---

## 2. The out-of-plane term: why a 2-D reduction needs a 3x3 tensor

Von Mises stress and principal values are defined on the *full* 3-D stress/strain state. Plane
stress and plane strain are each a specific, named reduction of that 3-D state - not a genuinely
2-D physics - so naively embedding the recovered 2x2 in-plane tensor into a 3x3 with a zero
third row/column would be **wrong for one of the two laws in each case**:

- **Plane stress** (`sigma_zz = 0` by the law's own definition): the strain's out-of-plane term
  is *not* zero - `eps_zz = -nu/(1-nu) (eps_xx + eps_yy)`, the free lateral contraction/expansion
  a plane-stress body is defined to allow.
- **Plane strain** (`eps_zz = 0` by the law's own definition): the stress's out-of-plane term is
  *not* zero - `sigma_zz = nu (sigma_xx + sigma_yy)`, the confining stress the zero-strain
  constraint produces.

`_out_of_plane_terms` names this explicitly, dispatching on which constitutive law
(`PlaneStressConstitutive`/`PlaneStrainConstitutive`) built the element - the only two
constitutive laws this reduction is defined for; `RecoveryInput.__post_init__` enforces that
invariant fail-fast rather than discovering it deep inside a tensor computation.

---

## 3. Verification

Numerically checked against a clean, hand-derivable case: a two-triangle rectangular plate under
a consistent-nodal-load uniaxial tension (the same model `test_static_t3_plate_analytical.py`
solves) produces the classical uniaxial stress state `sigma_xx = P/(H t)`, `sigma_yy = tau_xy =
0` exactly - a state whose principal stresses (`[0, 0, sigma_xx]`), von Mises stress
(`sigma_xx`, the well-known uniaxial-stress identity), and out-of-plane strain
(`eps_zz = eps_yy` for this specific case, since both reduce to `-nu * eps_xx`) are all
independently hand-checkable.

`tests/unit/test_postprocess_recovery.py` (6 tests): element-field stress/strain/principal/von
Mises against the closed forms above; nodal recovery reproduces the (here, uniform) element field
at every node; strain energy checked against Clapeyron's theorem (`U = 0.5 F.u` for a
linear-elastic system loaded from zero) - a formula independent of `0.5 u^T K u`, giving a real,
non-circular cross-check; a plane-strain case checked against hand-derived `sigma_zz`; the
unsupported-constitutive and wrong-shape error paths.
