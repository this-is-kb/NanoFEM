"""nanofem.numerics.tensors.

Tensor algebra: Voigt wire format, Mandel internal form, rotations, invariants (SDS Section 9,
ADR-016).

Responsibilities
----------------
- Conventions of SDS C-1 encoded once; converters as the only representation bridge
- Second- and fourth-order tensor algebra, SO(d)/Bond rotations, invariants and spectra
  (SDS Section 9), batched over arbitrary leading axes

Future modules
--------------
- conventions.py (implemented: VOIGT_ORDER, VOIGT_LENGTH)
- errors.py (implemented: TensorError, RepresentationError, NotRotationError, TensorSymmetryError)
- second_order.py (implemented: symmetric/skew parts, trace, det, inverse, deviator, norm,
  outer products, contractions)
- fourth_order.py (implemented: identity, symmetrizer, volumetric/deviatoric projectors,
  the isotropic oracle, symmetry classification)
- voigt.py (implemented: Voigt/Mandel/full converters - the only representation bridge)
- rotations.py (implemented: SO(d) rotation, Bond transformation pair)
- invariants.py (implemented: I1/I2/I3, J2/J3, von Mises, principal values/directions)
- verification.py (implemented: module-level verify_tensor_library / is_tensor_library_valid)

Explicitly out of scope (SDS Section 9's own "future-facing" clause)
----------------------------------------------------------------------
- Anisotropic material-frame storage and symmetry-class utilities - no real consumer
  exists yet; deferred until a materials/physics phase needs them.

TODO
----
- TODO(phase-2+): anisotropic material-frame storage and symmetry-class utilities,
  once a physics theory consumes them
"""

from __future__ import annotations

from nanofem.numerics.tensors.conventions import VOIGT_LENGTH, VOIGT_ORDER
from nanofem.numerics.tensors.errors import (
    NotRotationError,
    RepresentationError,
    TensorError,
    TensorSymmetryError,
)
from nanofem.numerics.tensors.fourth_order import (
    apply,
    deviatoric_projector,
    has_major_symmetry,
    has_minor_symmetry,
    identity_fourth,
    isotropic_stiffness,
    symmetrizer,
    volumetric_projector,
)
from nanofem.numerics.tensors.invariants import (
    deviatoric_second_invariant,
    deviatoric_third_invariant,
    first_invariant,
    principal_directions,
    principal_values,
    second_invariant,
    third_invariant,
    von_mises,
)
from nanofem.numerics.tensors.rotations import (
    bond_matrix_strain,
    bond_matrix_stress,
    is_rotation,
    rotate_second_order,
    rotate_stiffness_voigt,
    rotate_vector,
)
from nanofem.numerics.tensors.second_order import (
    determinant,
    deviator,
    double_contraction,
    frobenius_norm,
    inverse,
    is_symmetric,
    outer,
    single_contraction,
    skew_part,
    symmetric_outer,
    symmetric_part,
    trace,
)
from nanofem.numerics.tensors.verification import is_tensor_library_valid, verify_tensor_library
from nanofem.numerics.tensors.voigt import (
    VoigtKind,
    fourth_order_to_mandel,
    full_to_mandel,
    mandel_to_fourth_order,
    mandel_to_full,
    mandel_to_voigt,
    strain_to_voigt,
    stress_to_voigt,
    voigt_to_mandel,
    voigt_to_strain,
    voigt_to_stress,
)

__all__ = [
    # conventions
    "VOIGT_ORDER",
    "VOIGT_LENGTH",
    # errors
    "TensorError",
    "RepresentationError",
    "NotRotationError",
    "TensorSymmetryError",
    # second order
    "symmetric_part",
    "skew_part",
    "trace",
    "determinant",
    "inverse",
    "deviator",
    "frobenius_norm",
    "outer",
    "symmetric_outer",
    "single_contraction",
    "double_contraction",
    "is_symmetric",
    # fourth order
    "identity_fourth",
    "symmetrizer",
    "volumetric_projector",
    "deviatoric_projector",
    "isotropic_stiffness",
    "has_major_symmetry",
    "has_minor_symmetry",
    "apply",
    # voigt / mandel
    "VoigtKind",
    "strain_to_voigt",
    "stress_to_voigt",
    "voigt_to_strain",
    "voigt_to_stress",
    "full_to_mandel",
    "mandel_to_full",
    "voigt_to_mandel",
    "mandel_to_voigt",
    "fourth_order_to_mandel",
    "mandel_to_fourth_order",
    # rotations
    "is_rotation",
    "rotate_vector",
    "rotate_second_order",
    "bond_matrix_stress",
    "bond_matrix_strain",
    "rotate_stiffness_voigt",
    # invariants
    "first_invariant",
    "second_invariant",
    "third_invariant",
    "deviatoric_second_invariant",
    "deviatoric_third_invariant",
    "von_mises",
    "principal_values",
    "principal_directions",
    # verification
    "verify_tensor_library",
    "is_tensor_library_valid",
]
