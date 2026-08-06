"""Isotropic local elasticity, 1-D axial or 2-D plane kinematics (SDS Section 4.1).

The dim=1 constitutive law here is the plain axial one, ``sigma = E epsilon``
- not a reduction of ``numerics.tensors.fourth_order.isotropic_stiffness``,
which is explicitly documented as a verification-oracle-only 3-D-generalized
tensor with no knowledge of any physical dimensional reduction. That oracle
is used only in this theory's verification tests, as an independent
cross-check (it happens to collapse to ``E`` exactly at ``dim=1``, since its
deviatoric part vanishes identically there - a genuine coincidence of the
general formula, not a hidden dependency).

``IsotropicElasticity`` declares only *kinematics* - a displacement field of
``dim`` components, the ``symmetric_gradient``/``voigt_map`` strain measure -
which stays identical regardless of which constitutive law is injected.
Plane stress vs. plane strain is purely a **constitutive-law** distinction
(both use the same 2-D strain measure), which is why ``dim=2`` is handled
here (kinematics only) while the two actual 2-D laws,
``PlaneStressConstitutive``/``PlaneStrainConstitutive``, live separately in
``physics/elasticity/plane.py`` - this Theory/ConstitutiveModel split is
exactly what lets a future ``EringenDifferentialConstitutive`` swap in
against the same kinematics later, with no solver-architecture change.

3-D reduction remains a real gap, honestly named: ``dim`` still rejects
anything other than 1 or 2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.assembly.contributions import ContributionKind, OperatorRole
from nanofem.numerics.operators.base import Continuity, derived_continuity
from nanofem.physics.base import ConstitutiveModel, Locality, Theory
from nanofem.state.layout import StateLayout
from nanofem.utils.exceptions import PhysicsError

_OPERATORS_USED: tuple[str, ...] = ("symmetric_gradient", "voigt_map")


class IsotropicElasticity(Theory):
    """Local isotropic elasticity kinematics, axial (dim=1) or plane (dim=2) (SDS Section 4.1)."""

    def __init__(self, dim: int = 1) -> None:
        if dim not in (1, 2):
            raise PhysicsError(
                f"IsotropicElasticity currently supports dim=1 (axial) or dim=2 (plane "
                f"stress/strain); dim={dim} needs a 3-D reduction, not yet implemented"
            )
        self._dim = dim

    @property
    def dim(self) -> int:
        """The spatial dimension this instance was constructed with (1 or 2)."""
        return self._dim

    def fields(self) -> tuple[tuple[str, int], ...]:
        """A displacement field ``u`` with ``dim`` components."""
        return (("u", self._dim),)

    def continuity_requirements(self) -> dict[str, Continuity]:
        """Derived from the operators this theory composes (SDS Section 8 rule)."""
        return {"u": derived_continuity(self.operators_used())}

    def required_properties(self) -> tuple[str, ...]:
        """``E`` alone for the dim=1 axial law; ``E`` and ``nu`` for the dim=2 plane laws."""
        return ("E",) if self._dim == 1 else ("E", "nu")

    def required_state(self) -> StateLayout:
        """Elastic law: no history, zero memory (SDS Section 7)."""
        return StateLayout(())

    def operators_used(self) -> tuple[str, ...]:
        """The SDS Section 8 operators this theory composes."""
        return _OPERATORS_USED

    def operator_roles(self) -> tuple[OperatorRole, ...]:
        """Declared per SDS 4.1; only STIFFNESS/FORCE are exercised by ``Bar``/``ContinuumElement``
        this phase."""
        return (
            OperatorRole.STIFFNESS,
            OperatorRole.MASS,
            OperatorRole.GEOMETRIC_STIFFNESS,
            OperatorRole.FORCE,
        )

    def contribution_kinds(self) -> tuple[ContributionKind, ...]:
        """Local elasticity is cell-only."""
        return (ContributionKind.CELL,)

    def locality(self) -> Locality:
        """Local elasticity is LOCAL, not PAIRWISE."""
        return Locality.LOCAL


class IsotropicElasticConstitutive(ConstitutiveModel):
    """``sigma = E * eps``, ``D = [[E]]`` - the real 1-D axial constitutive law."""

    def __init__(self, dim: int = 1) -> None:
        if dim != 1:
            raise PhysicsError(f"IsotropicElasticConstitutive supports dim=1 only; got {dim}")
        self._dim = dim

    def required_properties(self) -> tuple[str, ...]:
        """Only ``E``."""
        return ("E",)

    def state_layout(self) -> StateLayout:
        """Elastic law: no state."""
        return StateLayout(())

    def response_components(self) -> int:
        """A single generalized stress/strain component in 1-D."""
        return 1

    def respond_batch(
        self,
        strains: NDArray[np.float64],
        properties: dict[str, NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``stress = E * strains``; ``tangent`` is ``E`` broadcast to ``(..., 1, 1)``.

        ``strains`` has shape ``(..., 1)``; ``properties["E"]`` broadcasts
        against ``strains``'s leading batch axes.
        """
        e = np.asarray(properties["E"], dtype=np.float64)
        stress = e[..., np.newaxis] * strains
        tangent = np.broadcast_to(e[..., np.newaxis, np.newaxis], strains.shape + (1,)).copy()
        return stress, tangent
