"""Gauss -> nodal stress/strain recovery, principal values, von Mises, strain energy (SDS 2.19).

Scope: plane-continuum elements (``ContinuumElement`` under
``IsotropicElasticity(dim=2)`` - T3/Q4, the only 2-D physics this codebase
has) only. A structural element's "stress" is a different, already-solved
concern (`EulerBernoulliBeam`/`TimoshenkoBeam` report internal moment/shear
directly from their own constitutive law, not from a recovered field), so
this module does not touch them.

Recovery here is the classical *direct* method - evaluate the constitutive
response at each quadrature point (``ContinuumElement.quadrature_point_
response``), take the quadrature-weighted element average as that element's
representative field, then a measure-weighted average across every element
sharing a node for the nodal field. This is deliberately simpler than
superconvergent patch recovery (SPR) or a full Zienkiewicz-Zhu
least-squares extrapolation - both remain future work for adaptivity, not
needed to recover a field for stress/strain/von-Mises reporting or export.
Averaging is per call, not automatic across an entire model: pass only the
elements of one material/domain at a time to respect SDS 2.19's "never
crosses material interfaces" rule.

Plane stress/strain are 2-D reductions of a 3-D tensor state, not a genuinely
2-D one: von Mises and principal values are only physically meaningful on
the full 3x3 tensor, which needs the *out-of-plane* term (``sigma_zz`` for
plane stress is exactly 0 by the reduction's own definition; ``eps_zz`` for
plane strain is exactly 0; the other of each pair is recovered from the
in-plane state and Poisson's ratio). ``_out_of_plane_terms`` names this
explicitly rather than silently embedding a 2x2 tensor into a 3x3 with a
zero corner, which would be wrong for plane strain's stress and for plane
stress's strain alike.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.elements.continuum.continuum import ContinuumElement
from nanofem.numerics.tensors.invariants import principal_values, von_mises
from nanofem.numerics.tensors.voigt import voigt_to_strain, voigt_to_stress
from nanofem.physics.base import ConstitutiveModel
from nanofem.physics.elasticity.plane import PlaneStrainConstitutive, PlaneStressConstitutive
from nanofem.utils.exceptions import PhysicsError


@dataclass(frozen=True)
class RecoveryInput:
    """One element plus the constitutive law/Poisson's ratio recovery needs for it.

    ``constitutive`` and ``poisson_ratio`` are supplied explicitly rather than
    read back off the element: ``ContinuumElement`` holds them privately (an
    implementation cache for stiffness, not a public query surface), and the
    caller already has both on hand from the ``Model`` domain that built the
    element.
    """

    element: ContinuumElement
    constitutive: ConstitutiveModel
    poisson_ratio: float

    def __post_init__(self) -> None:
        if not isinstance(self.constitutive, (PlaneStressConstitutive, PlaneStrainConstitutive)):
            raise PhysicsError(
                f"stress/strain recovery's out-of-plane reduction is defined for "
                f"PlaneStressConstitutive/PlaneStrainConstitutive only, got "
                f"{type(self.constitutive).__name__}"
            )


@dataclass(frozen=True)
class FieldResult:
    """A recovered field at one location: full 3x3 tensors plus derived scalars."""

    strain: NDArray[np.float64]
    stress: NDArray[np.float64]
    principal_strains: NDArray[np.float64]
    principal_stresses: NDArray[np.float64]
    von_mises_stress: float


@dataclass(frozen=True)
class ElementFieldResult:
    """One element's quadrature-weighted-average recovered field."""

    element_id: int
    measure: float
    field: FieldResult


def _out_of_plane_terms(
    constitutive: ConstitutiveModel,
    strain_voigt: NDArray[np.float64],
    stress_voigt: NDArray[np.float64],
    nu: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(eps_zz, sigma_zz)`` batches for a plane stress/strain reduction.

    ``constitutive`` is one of the two plane laws by construction -
    ``RecoveryInput.__post_init__`` is the single place that invariant is
    enforced, so it is not re-checked here.
    """
    if isinstance(constitutive, PlaneStressConstitutive):
        eps_zz = -nu / (1.0 - nu) * (strain_voigt[..., 0] + strain_voigt[..., 1])
        return eps_zz, np.zeros_like(eps_zz)
    sigma_zz = nu * (stress_voigt[..., 0] + stress_voigt[..., 1])
    return np.zeros_like(sigma_zz), sigma_zz


def _embed_plane(in_plane: NDArray[np.float64], zz: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(..., 2, 2)`` in-plane tensor batch + ``(...,)`` zz batch -> ``(..., 3, 3)``."""
    tensor = np.zeros(in_plane.shape[:-2] + (3, 3), dtype=np.float64)
    tensor[..., :2, :2] = in_plane
    tensor[..., 2, 2] = zz
    return tensor


def _full_field(
    strain_voigt: NDArray[np.float64],
    stress_voigt: NDArray[np.float64],
    constitutive: ConstitutiveModel,
    nu: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Kinematic/kinetic Voigt batches -> full ``(..., 3, 3)`` strain/stress tensor batches."""
    eps_zz, sigma_zz = _out_of_plane_terms(constitutive, strain_voigt, stress_voigt, nu)
    strain = _embed_plane(voigt_to_strain(strain_voigt), eps_zz)
    stress = _embed_plane(voigt_to_stress(stress_voigt), sigma_zz)
    return strain, stress


def _field_result(strain: NDArray[np.float64], stress: NDArray[np.float64]) -> FieldResult:
    """Full ``(3, 3)`` strain/stress tensors -> the derived-quantity bundle."""
    return FieldResult(
        strain=strain,
        stress=stress,
        principal_strains=principal_values(strain),
        principal_stresses=principal_values(stress),
        von_mises_stress=float(von_mises(stress)),
    )


def recover_element_fields(
    inputs: Sequence[RecoveryInput], displacements: NDArray[np.float64]
) -> tuple[ElementFieldResult, ...]:
    """Each element's quadrature-weighted-average recovered field, from a global solve."""
    results = []
    for item in inputs:
        element = item.element
        local_u = displacements[element.global_dofs]
        strain_voigt, stress_voigt, point_measure = element.quadrature_point_response(local_u)
        strain, stress = _full_field(
            strain_voigt, stress_voigt, item.constitutive, item.poisson_ratio
        )
        avg_strain = np.average(strain, axis=0, weights=point_measure)
        avg_stress = np.average(stress, axis=0, weights=point_measure)
        results.append(
            ElementFieldResult(
                element_id=element.cell_id,
                measure=element.measure(),
                field=_field_result(avg_strain, avg_stress),
            )
        )
    return tuple(results)


def recover_nodal_fields(
    inputs: Sequence[RecoveryInput], element_results: Sequence[ElementFieldResult]
) -> dict[int, FieldResult]:
    """Measure-weighted average of each element's field across the nodes it touches.

    Pass only elements from one material/domain at a time (SDS 2.19: nodal
    averaging must never cross a material interface).
    """
    strain_accum: dict[int, NDArray[np.float64]] = {}
    stress_accum: dict[int, NDArray[np.float64]] = {}
    weight_accum: dict[int, float] = {}
    for item, result in zip(inputs, element_results, strict=True):
        weight = result.measure
        for node_id in item.element.node_ids:
            if node_id not in weight_accum:
                strain_accum[node_id] = np.zeros((3, 3), dtype=np.float64)
                stress_accum[node_id] = np.zeros((3, 3), dtype=np.float64)
                weight_accum[node_id] = 0.0
            strain_accum[node_id] = strain_accum[node_id] + weight * result.field.strain
            stress_accum[node_id] = stress_accum[node_id] + weight * result.field.stress
            weight_accum[node_id] += weight
    return {
        node_id: _field_result(strain_accum[node_id] / total, stress_accum[node_id] / total)
        for node_id, total in weight_accum.items()
    }


def strain_energy(element_results: Sequence[ElementFieldResult]) -> float:
    """``sum_e 0.5 (stress_e : strain_e) measure_e`` - independent of ``0.5 u^T K u``."""
    return float(
        sum(
            0.5 * float(np.tensordot(r.field.stress, r.field.strain)) * r.measure
            for r in element_results
        )
    )
