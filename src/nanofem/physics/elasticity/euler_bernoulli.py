"""Local Euler-Bernoulli bending theory, 1-D only (SDS Section 4.1, ADR-002's E-5 companion).

**Sign convention** (stated here to satisfy SDS clause E-10's "document sign
conventions for V and M diagrams" - no normative convention for this exists
anywhere else in the design docs, so this is the first place it is written
down): local axis ``x`` runs node 1 (``x=0``) to node 2 (``x=L``); transverse
deflection ``w`` is positive in the local ``+y`` direction; rotation
``theta = dw/dx`` is positive by the right-hand rule about local ``+z``
(matching the ``r.z`` DOF component name below); curvature
``kappa = d2w/dx2``; bending moment ``M = E I kappa``, sagging positive. This
reproduces the universal textbook stiffness
``K = (EI/L^3)[[12,6L,-12,6L],[6L,4L^2,-6L,2L^2],[-12,-6L,12,-6L],
[6L,2L^2,-6L,4L^2]]`` in DOF order ``(w1, theta1, w2, theta2)``. Moment/shear
*recovery* (post-processing) is out of scope here.

The constitutive law here, ``M_per_I = E * kappa``, is the same functional
form as ``IsotropicElasticConstitutive``'s axial law but is deliberately a
separate class: curvature is not axial strain, and the second moment of
area ``I`` (not cross-sectional area ``A``) is the element-layer multiplier
applied afterward, exactly as ``Bar`` applies ``area`` - see
``elements/structural/beam_eb.py``. Reusing the axial class would either
require rewriting its axial-specific docstring/errors or silently stretch
its documented meaning; a dedicated ~15-line class keeps both physical laws
honest about what they actually model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.assembly.contributions import ContributionKind, OperatorRole
from nanofem.numerics.operators.base import Continuity, derived_continuity
from nanofem.physics.base import ConstitutiveModel, Locality, Theory
from nanofem.state.layout import StateLayout

_OPERATORS_USED: tuple[str, ...] = ("second_gradient",)


def _curvature_from_hessian(physical_hessians: NDArray[np.float64]) -> NDArray[np.float64]:
    """``kappa = H[..., 0, 0]`` from a ``(..., 1, 1)`` physical Hessian batch.

    A squeeze, not a projection, at the 1-D embedding this theory is scoped
    to. ``second_gradient_tensor`` deliberately stops at the raw Hessian and
    defers this contraction to a theory (its own docstring); this is that
    contraction, owned here because it is bending physics, not a
    numerics-layer concern (rule R1: numerics stays mechanics-free).
    """
    return physical_hessians[..., 0, 0]


def _reference_derivative_scale(
    dof_derivative_orders: tuple[int, ...], jacobian: float
) -> NDArray[np.float64]:
    """Per-shape-function scale: Hermite reference-derivative DOFs -> physical-derivative DOFs.

    A cubic Hermite family's "derivative" DOFs are values of ``dw/dxi`` (the
    *reference*-coordinate derivative), not ``dw/dx`` (the physical rotation
    a beam element's global DOF actually is). Composing the raw
    ``physical_hessian``/``second_gradient_tensor`` output directly against
    physically-parametrized rotation DOFs reproduces a stiffness matrix off
    by exactly the Jacobian ``J`` on every rotation-associated row/column and
    ``J^2`` on the rotation-rotation term - confirmed by direct computation
    while building this module. ``scale = jacobian ** derivative_order``
    corrects it: general to any Hermite/C1 family used with physically
    -parametrized derivative DOFs, not specific to bending, but this is its
    only consumer today.
    """
    return np.array([jacobian**order for order in dof_derivative_orders], dtype=np.float64)


class EulerBernoulliBendingTheory(Theory):
    """Local Euler-Bernoulli bending, 1-D only (no ``dim`` parameter - there is one case)."""

    def fields(self) -> tuple[tuple[str, int], ...]:
        """Transverse displacement ``u.y`` and rotation ``r.z`` (SDS C-2's own worked example)."""
        return (("u", 1), ("r", 1))

    def field_component_names(self) -> dict[str, tuple[str, ...]]:
        """``u`` is the *y*-component, ``r`` is the *z*-component - not the positional default."""
        return {"u": ("y",), "r": ("z",)}

    def continuity_requirements(self) -> dict[str, Continuity]:
        """``second_gradient`` forces ``Continuity.C1`` (SDS Section 8 rule) on both fields."""
        continuity = derived_continuity(self.operators_used())
        return {"u": continuity, "r": continuity}

    def required_properties(self) -> tuple[str, ...]:
        """Only ``E`` - the second moment of area is an element-layer multiplier, not a material."""
        return ("E",)

    def required_state(self) -> StateLayout:
        """Elastic law: no history, zero memory (SDS Section 7)."""
        return StateLayout(())

    def operators_used(self) -> tuple[str, ...]:
        """The SDS Section 8 operator this theory composes."""
        return _OPERATORS_USED

    def operator_roles(self) -> tuple[OperatorRole, ...]:
        """MASS/GEOMETRIC_STIFFNESS excluded: modal/buckling analysis are out of scope."""
        return (OperatorRole.STIFFNESS, OperatorRole.FORCE)

    def contribution_kinds(self) -> tuple[ContributionKind, ...]:
        """Local bending is cell-only."""
        return (ContributionKind.CELL,)

    def locality(self) -> Locality:
        """Local bending is LOCAL, not PAIRWISE."""
        return Locality.LOCAL


class EulerBernoulliBendingConstitutive(ConstitutiveModel):
    """``M_per_I = E * kappa``, ``D = [[E]]`` - moment-per-unit-second-moment vs. curvature."""

    def required_properties(self) -> tuple[str, ...]:
        """Only ``E``."""
        return ("E",)

    def state_layout(self) -> StateLayout:
        """Elastic law: no state."""
        return StateLayout(())

    def response_components(self) -> int:
        """A single generalized moment/curvature component in 1-D bending."""
        return 1

    def respond_batch(
        self,
        strains: NDArray[np.float64],
        properties: dict[str, NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``stress = E * strains``; ``tangent`` is ``E`` broadcast to ``(..., 1, 1)``.

        Here ``strains`` *is* curvature ``kappa``, shape ``(..., 1)``;
        ``properties["E"]`` broadcasts against ``strains``'s leading batch
        axes.
        """
        e = np.asarray(properties["E"], dtype=np.float64)
        stress = e[..., np.newaxis] * strains
        tangent = np.broadcast_to(e[..., np.newaxis, np.newaxis], strains.shape + (1,)).copy()
        return stress, tangent
