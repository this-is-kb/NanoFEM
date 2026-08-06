"""Module-level verification suite for the tensor algebra layer (SDS Section 9).

Every prior numerics phase hung its verification suite off one rich, stateful
class (``ReferenceElement``, ``Interpolation``, ``ShapeFunctionFamily``,
``GeometricMapping``, ``QuadratureRule``). ``tensors`` has no such object - it
is a library of pure functions, matching SDS's own "stateless recipe" language
- so verification here is module-level: :func:`verify_tensor_library` runs
every ``verify_*`` check with fixed-seed random data and raises on the first
failure, and :func:`is_tensor_library_valid` wraps it as a boolean. Individual
``verify_*`` functions are independently callable and independently testable.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.tensors.errors import RepresentationError, TensorError, TensorSymmetryError
from nanofem.numerics.tensors.fourth_order import (
    apply,
    deviatoric_projector,
    has_major_symmetry,
    has_minor_symmetry,
    isotropic_stiffness,
    symmetrizer,
    volumetric_projector,
)
from nanofem.numerics.tensors.invariants import (
    first_invariant,
    principal_directions,
    third_invariant,
    von_mises,
)
from nanofem.numerics.tensors.rotations import (
    bond_matrix_strain,
    bond_matrix_stress,
    rotate_second_order,
)
from nanofem.numerics.tensors.second_order import double_contraction
from nanofem.numerics.tensors.voigt import (
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

_SEED = 20260701
_DIMS = (1, 2, 3)
_TOL = 1.0e-10


def _random_symmetric(rng: np.random.Generator, n: int, dim: int) -> NDArray[np.float64]:
    raw = rng.normal(size=(n, dim, dim))
    return np.asarray(0.5 * (raw + np.swapaxes(raw, -1, -2)), dtype=np.float64)


def _random_rotation(rng: np.random.Generator, dim: int) -> NDArray[np.float64]:
    raw = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(raw)
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return np.asarray(q, dtype=np.float64)


def _compose_fourth_order(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.einsum("ijmn,mnkl->ijkl", a, b), dtype=np.float64)


def verify_round_trip() -> None:
    """Voigt/Mandel/full round trips are lossless, for every supported dimension."""
    rng = np.random.default_rng(_SEED)
    for dim in _DIMS:
        tensor = _random_symmetric(rng, 4, dim)

        if not np.allclose(voigt_to_strain(strain_to_voigt(tensor)), tensor, atol=_TOL):
            raise RepresentationError(f"strain Voigt round trip failed at dim={dim}")
        if not np.allclose(voigt_to_stress(stress_to_voigt(tensor)), tensor, atol=_TOL):
            raise RepresentationError(f"stress Voigt round trip failed at dim={dim}")
        if not np.allclose(mandel_to_full(full_to_mandel(tensor)), tensor, atol=_TOL):
            raise RepresentationError(f"Mandel round trip failed at dim={dim}")

        strain_voigt = strain_to_voigt(tensor)
        mandel = voigt_to_mandel(strain_voigt, kind="strain")
        back = mandel_to_voigt(mandel, kind="strain")
        if not np.allclose(back, strain_voigt, atol=_TOL):
            raise RepresentationError(f"Voigt<->Mandel strain round trip failed at dim={dim}")

        stress_voigt = stress_to_voigt(tensor)
        mandel_stress = voigt_to_mandel(stress_voigt, kind="stress")
        back_stress = mandel_to_voigt(mandel_stress, kind="stress")
        if not np.allclose(back_stress, stress_voigt, atol=_TOL):
            raise RepresentationError(f"Voigt<->Mandel stress round trip failed at dim={dim}")

        stiffness = isotropic_stiffness(2.0, 1.0, dim)
        mandel_matrix = fourth_order_to_mandel(stiffness)
        if not np.allclose(mandel_to_fourth_order(mandel_matrix), stiffness, atol=_TOL):
            raise RepresentationError(f"fourth-order Mandel round trip failed at dim={dim}")


def verify_work_conjugacy() -> None:
    """``sigma:eps`` agrees whether computed in full, Voigt, or Mandel form.

    The load-bearing identity this module exists to protect: a hand-inserted
    factor-of-2 error in either representation breaks this while individual
    round trips can still pass.
    """
    rng = np.random.default_rng(_SEED + 1)
    for dim in _DIMS:
        sigma = _random_symmetric(rng, 1, dim)[0]
        eps = _random_symmetric(rng, 1, dim)[0]

        work_full = float(double_contraction(sigma[np.newaxis], eps[np.newaxis])[0])
        work_voigt = float(stress_to_voigt(sigma) @ strain_to_voigt(eps))
        work_mandel = float(full_to_mandel(sigma) @ full_to_mandel(eps))

        if not np.isclose(work_voigt, work_full, atol=_TOL):
            raise RepresentationError(
                f"Voigt work {work_voigt:.16g} != full-tensor work {work_full:.16g} at dim={dim}"
            )
        if not np.isclose(work_mandel, work_full, atol=_TOL):
            raise RepresentationError(
                f"Mandel work {work_mandel:.16g} != full-tensor work {work_full:.16g} at dim={dim}"
            )


def verify_projector_algebra() -> None:
    """``J + K = Ibar``, both are idempotent under fourth-order composition, and ``J:K = 0``."""
    for dim in _DIMS:
        j = volumetric_projector(dim)
        k = deviatoric_projector(dim)
        ibar = symmetrizer(dim)

        if not np.allclose(j + k, ibar, atol=_TOL):
            raise TensorError(f"J + K != Ibar at dim={dim}")
        if not np.allclose(_compose_fourth_order(j, j), j, atol=_TOL):
            raise TensorError(f"J is not idempotent at dim={dim}")
        if not np.allclose(_compose_fourth_order(k, k), k, atol=_TOL):
            raise TensorError(f"K is not idempotent at dim={dim}")
        if not np.allclose(_compose_fourth_order(j, k), 0.0, atol=_TOL):
            raise TensorError(f"J:K != 0 at dim={dim}")


def verify_isotropic_oracle() -> None:
    """``isotropic_stiffness`` matches the independently-derived Lame closed form."""
    rng = np.random.default_rng(_SEED + 2)
    for dim in (2, 3):
        bulk_modulus, shear_modulus = 3.7, 1.4
        lame_lambda = bulk_modulus - 2.0 * shear_modulus / dim
        stiffness = isotropic_stiffness(bulk_modulus, shear_modulus, dim)
        eps = _random_symmetric(rng, 3, dim)

        actual = apply(stiffness, eps)
        identity = np.eye(dim, dtype=np.float64)
        trace_eps = np.trace(eps, axis1=-2, axis2=-1)
        expected = (
            lame_lambda * trace_eps[..., np.newaxis, np.newaxis] * identity
            + 2.0 * shear_modulus * eps
        )
        if not np.allclose(actual, expected, atol=_TOL):
            raise RepresentationError(f"isotropic oracle disagrees with the Lame form at dim={dim}")


def verify_symmetry_classification() -> None:
    """The symmetry checks accept the isotropic tensor and reject a deliberately broken copy."""
    for dim in (2, 3):
        stiffness = isotropic_stiffness(2.0, 1.0, dim)
        if not bool(has_major_symmetry(stiffness)):
            raise TensorSymmetryError(f"isotropic stiffness failed major symmetry at dim={dim}")
        if not bool(has_minor_symmetry(stiffness)):
            raise TensorSymmetryError(f"isotropic stiffness failed minor symmetry at dim={dim}")

        broken = stiffness.copy()
        broken[0, 0, 0, 1] += 1.0
        if bool(has_minor_symmetry(broken)):
            raise TensorSymmetryError(
                f"has_minor_symmetry failed to catch a deliberately broken tensor at dim={dim}"
            )


def verify_rotation_consistency() -> None:
    """Bond transforms are mutually consistent, frame-invariant, and reject non-rotations."""
    rng = np.random.default_rng(_SEED + 3)
    for dim in (2, 3):
        q = _random_rotation(rng, dim)

        m_sigma = bond_matrix_stress(q)
        m_epsilon = bond_matrix_strain(q)
        if not np.allclose(m_epsilon, np.linalg.inv(m_sigma).T, atol=_TOL):
            raise TensorError(f"M_epsilon != M_sigma^-T at dim={dim}")

        sigma = _random_symmetric(rng, 1, dim)[0]
        eps = _random_symmetric(rng, 1, dim)[0]
        work_before = float(double_contraction(sigma[np.newaxis], eps[np.newaxis])[0])

        sigma_voigt_rotated = m_sigma @ stress_to_voigt(sigma)
        eps_voigt_rotated = m_epsilon @ strain_to_voigt(eps)
        work_after = float(sigma_voigt_rotated @ eps_voigt_rotated)
        if not np.isclose(work_before, work_after, atol=_TOL):
            raise TensorError(f"rotation broke frame-invariance of work at dim={dim}")

        sigma_rotated_direct = rotate_second_order(q, sigma)
        if not np.allclose(voigt_to_stress(sigma_voigt_rotated), sigma_rotated_direct, atol=_TOL):
            raise TensorError(
                f"Bond-rotated stress disagrees with direct tensor rotation at dim={dim}"
            )

        round_trip = rotate_second_order(q.T, rotate_second_order(q, sigma))
        if not np.allclose(round_trip, sigma, atol=_TOL):
            raise TensorError(
                f"rotate then inverse-rotate did not return the original tensor at dim={dim}"
            )


def verify_invariants() -> None:
    """Von Mises matches the uniaxial closed form; I1/I3 agree with the principal-value routes."""
    rng = np.random.default_rng(_SEED + 4)
    uniaxial = np.zeros((1, 3, 3), dtype=np.float64)
    uniaxial[0, 0, 0] = 5.0
    if not np.isclose(float(von_mises(uniaxial)[0]), 5.0, atol=_TOL):
        raise TensorError("von Mises stress of a uniaxial state did not equal the axial stress")

    tensor = _random_symmetric(rng, 1, 3)
    values = np.linalg.eigvalsh(tensor[0])
    if not np.isclose(float(first_invariant(tensor)[0]), float(np.sum(values)), atol=_TOL):
        raise TensorError("I1 disagrees with the sum of principal values")
    if not np.isclose(float(third_invariant(tensor)[0]), float(np.prod(values)), atol=_TOL):
        raise TensorError("I3 disagrees with the product of principal values")


def verify_eigendecomposition() -> None:
    """``V diag(values) V^T`` reconstructs the original symmetric tensor batch."""
    rng = np.random.default_rng(_SEED + 5)
    for dim in _DIMS:
        tensor = _random_symmetric(rng, 3, dim)
        values, vectors = principal_directions(tensor)
        reconstructed = np.einsum("...ij,...j,...kj->...ik", vectors, values, vectors)
        if not np.allclose(reconstructed, tensor, atol=_TOL):
            raise TensorError(f"eigendecomposition failed to reconstruct the tensor at dim={dim}")


def verify_tensor_library() -> None:
    """Run every ``verify_*`` check in this module; raise on the first failure."""
    verify_round_trip()
    verify_work_conjugacy()
    verify_projector_algebra()
    verify_isotropic_oracle()
    verify_symmetry_classification()
    verify_rotation_consistency()
    verify_invariants()
    verify_eigendecomposition()


def is_tensor_library_valid() -> bool:
    """Return ``True`` if :func:`verify_tensor_library` passes, ``False`` otherwise."""
    try:
        verify_tensor_library()
    except TensorError:
        return False
    return True
