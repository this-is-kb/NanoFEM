# NanoFEM Tensors

**Status:** implemented and tested (v0.7.0). Companion to ARCHITECTURE_v2.md, the SDS
(Section 9), and OPERATORS.md, which is the first consumer of this layer.

**Scope discipline.** Small dense tensor algebra over batched arrays with arbitrary leading
axes — a single tensor is a batch of shape `()`. There is no mesh, no material, no element, no
constitutive model, and no anisotropic material-frame storage: this package is pure linear
algebra on square matrices and Voigt/Mandel vectors, mechanics-free, matching the `numerics`
leaf contract (rule R1).

---

## 1. What lives here, and what does not

A tensor here is a plain `numpy` array; the package's job is the algebra on it, not the
physics it will eventually represent. `second_order.py` and `fourth_order.py` operate on full
tensor form (`(..., d, d)` and `(..., d, d, d, d)`); `voigt.py` is the *only* sanctioned bridge
to the compacted Voigt and Mandel representations; `rotations.py` and `invariants.py` build on
both. Every function accepts arbitrary batch axes and no function anywhere in the package
mentions strain, stress, displacement, or any other mechanics name in its logic — those names
appear only in docstrings, as the motivating use case.

---

## 2. Second-order algebra

`second_order.py` provides the plain linear-algebra vocabulary: symmetric/skew parts, trace,
determinant, inverse (raising `TensorError` rather than a bare `LinAlgError` on a singular
input), the deviator `dev(A) = A - (tr(A)/d) I`, the Frobenius norm, outer products
(`u ⊗ v`, and its symmetrization), and single/double contraction (`A·v`, `A:B`). Every function
is a thin, direct `einsum` or `numpy.linalg` call — there is no algorithm here, only a single
place each operation is spelled correctly once.

---

## 3. Fourth-order algebra and the isotropic oracle

`fourth_order.py` represents a fourth-order tensor as a full `(..., d, d, d, d)` array (no
Voigt compaction — that is `voigt.py`'s job) and provides the identity `I_ijkl = δ_ik δ_jl`,
the symmetrizer `Ībar_ijkl = 1/2(δ_ik δ_jl + δ_il δ_jk)`, and the resolution of identity
`Ibar = J + K` into the volumetric projector `J = 1/d I2 ⊗ I2` and the deviatoric projector
`K = Ibar - J`.

`isotropic_stiffness(kappa, mu, dim) = 3κJ + 2μK` is the canonical closed form for isotropic
elasticity (SDS Section 9) — and it is a **verification oracle only**. It carries no knowledge
of plane-stress/strain reductions, no unit system, and no material record; it exists so that a
constitutive law built anywhere else in the framework can be checked against a second,
independently-derived formula (`C:ε = λ tr(ε) I + 2με` with `λ = κ - 2μ/d`), which is a
stronger check than comparing a formula to itself.

`has_major_symmetry`/`has_minor_symmetry` classify a fourth-order tensor's symmetry from its
values rather than from how it was constructed — the same "report, not impose" discipline the
quadrature layer's symmetry module established (QUADRATURE.md §6).

---

## 4. Voigt: the wire format

`voigt.py` implements SDS C-1 exactly: strain vectors are **kinematic** Voigt (engineering
shear, `γ_ij = 2ε_ij` for `i ≠ j`); stress vectors are **kinetic** Voigt (plain components).
This split is what makes `σᵀ ε_voigt = σ:ε` hold with no hidden factor — the single fact this
whole module exists to protect, and the one a hand-written cross-developer implementation gets
wrong most often. `strain_to_voigt`/`stress_to_voigt` and their inverses key off
`conventions.VOIGT_ORDER`/`VOIGT_LENGTH` and never re-derive the index pairing.

---

## 5. Mandel: the internal form

Mandel form uses a single, symmetric `√2` weighting on the off-diagonal components for *both*
stress and strain — there is no kinematic/kinetic split here, because Mandel exists precisely
so that contractions, norms, and eigenproblems are plain matrix algebra with no factor
bookkeeping (SDS Section 9). Every Voigt↔Mandel conversion in this package routes through the
full tensor form (`voigt_to_strain`/`voigt_to_stress` then `full_to_mandel`, and the reverse) —
there is exactly one place the kinematic factor of 2 is applied and exactly one place the
Mandel `√2` is applied, rather than two independent formulas that must be kept in sync.
`fourth_order_to_mandel`/`mandel_to_fourth_order` extend the same weighting to fourth-order
tensors, turning a major- and minor-symmetric `C` into a plain symmetric matrix.

---

## 6. Rotation: `SO(d)` and the Bond transformation pair

`rotations.py` rotates vectors (`v' = Qv`) and second-order tensors (`A' = QAQᵀ`) directly, and
builds the Bond matrices `M_σ`, `M_ε` for rotating a Voigt-form stiffness
(`D'_voigt = M_σ D M_σᵀ`). `M_σ` is built **column by column** from the rotation's action on the
Voigt basis tensors — reusing `voigt_to_stress`/`stress_to_voigt` as the single source of truth
for the convention — rather than hand-derived per dimension, which is the classic source of a
convention bug in this corner of the literature. `M_ε` is then computed as `M_σ⁻ᵀ` by
inversion, never re-derived independently, so the identity `M_ε = M_σ⁻ᵀ` holds *by
construction*. `is_rotation` checks `QᵀQ = I` and `det(Q) = +1`; every rotation function raises
`NotRotationError` for anything else (a reflection, a non-orthogonal matrix) rather than
silently producing a physically meaningless result.

---

## 7. Invariants and spectra

`invariants.py` provides `I1, I2, I3` (trace, the standard second invariant, determinant),
their deviatoric counterparts `J2, J3`, `von Mises = √(3 J2)`, and batched symmetric
eigendecomposition (`principal_values`/`principal_directions`, via `eigvalsh`/`eigh` — exact
and stable for symmetric input, unlike the general complex-valued `eig`).

---

## 8. A leaf of the numerics layer

`tensors` imports nothing beyond `numpy` and `nanofem.utils.exceptions` — no mesh, no
materials, no physics, no elements, no interpolation, no mapping, no quadrature, no kernels, no
search. This is proven the same way every prior numerics phase proved it: a subprocess test
imports the package fresh, exercises every real function, and asserts none of those package
names appear in `sys.modules`; a source scan strips comments and docstrings and asserts no
`shape_function`, `stiffness_matrix`, `assembl`, or `DofHandler` term appears in executable
code.

---

## 9. Verification

Unlike every prior numerics phase, `tensors` has no single rich, stateful class to hang
`verify()`/`is_valid()` off — it is a library of pure functions, matching SDS's own "stateless
recipe" language. Verification is therefore **module-level**:
`verification.verify_tensor_library()` runs every check below in sequence and raises on the
first failure; `is_tensor_library_valid()` wraps it as a boolean.

| check | catches |
|---|---|
| `verify_round_trip` | a broken Voigt/Mandel/full conversion, for every supported dimension |
| `verify_work_conjugacy` | a lost or mis-placed engineering-shear or `√2` factor |
| `verify_projector_algebra` | `J + K ≠ Ibar`, non-idempotent projectors, or `J:K ≠ 0` |
| `verify_isotropic_oracle` | `isotropic_stiffness` disagreeing with the independent Lamé form |
| `verify_symmetry_classification` | `has_major_symmetry`/`has_minor_symmetry` missing a break |
| `verify_rotation_consistency` | `M_ε ≠ M_σ⁻ᵀ`, rotation breaking frame-invariance of work |
| `verify_invariants` | von Mises or `I1`/`I3` disagreeing with the principal-value routes |
| `verify_eigendecomposition` | a reconstruction `V diag(λ) Vᵀ ≠ A` |

Trip tests for the free-function checks construct a deliberately malformed array directly
(a strain Voigt vector missing its factor of 2, a reflection instead of a rotation) rather than
subclassing, since there is no class to subclass — this is the one adaptation the project's
testing discipline needed for a phase built entirely of pure functions.

---

## 10. Examples

```python
from nanofem.numerics.tensors import (
    strain_to_voigt, stress_to_voigt, full_to_mandel,
    isotropic_stiffness, apply, von_mises, verify_tensor_library,
)

eps = ...  # a strain tensor from somewhere upstream
sigma = apply(isotropic_stiffness(kappa=5.0, mu=2.0, dim=3), eps)
work = stress_to_voigt(sigma) @ strain_to_voigt(eps)   # == sigma:eps, no hidden factors
von_mises(sigma[None])                                  # equivalent stress
verify_tensor_library()                                 # every identity in §9
```

The success-criterion chain, executable in `examples/ex07_operators_and_tensors.py`: a
`ShapeFunctionFamily` + `IsoparametricMapping` produce physical gradients →
`numerics.operators` builds a strain-displacement operator → the resulting strain is converted
to Voigt and Mandel form → work conjugacy is checked → both this package's and `operators`'
verification suites run.

---

## 11. Future extension strategy

**Anisotropic material-frame storage and symmetry-class utilities** (SDS Section 9's own
"future-facing" clause) — storing an anisotropic stiffness in a declared material frame plus
one rotation to global per element block, and symmetry-class utilities (isotropic → cubic →
transversely isotropic → orthotropic → triclinic) for validating and importing first-principles
(DFT) elastic tensors. No real consumer exists yet: the first anisotropic `physics/elasticity`
theory is what would drive this, and building it now would be exactly the kind of
ahead-of-a-consumer work this project's phase discipline forbids. `rotate_stiffness_voigt`
already provides the one rotation this future work would need; what is missing is the
*storage* of a declared material frame, which belongs with `materials/` or `physics/`, not
here.

**A future array backend.** SDS Section 9's own limitation note: batched `einsum`-style
algebra in pure `numpy` is memory-bandwidth-bound. This module's purity (no state, no mesh, no
materials) makes it the natural first beneficiary of a future compiled or GPU backend — the
same "single swap point" argument ARCHITECTURE_v2 D-4/D-12 makes for `numerics` as a whole.
