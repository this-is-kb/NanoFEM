"""Local Timoshenko beam theory, 1-D only (SDS Section 4.1, ADR-002's E-5 companion).

Timoshenko theory relaxes the Euler-Bernoulli (Kirchhoff) assumption:
transverse deflection ``w(x)`` and rotation ``theta(x)`` are **independent**
fields, not ``theta = dw/dx``. This introduces a shear strain
``gamma = dw/dx - theta`` alongside the bending curvature
``kappa = d(theta)/dx`` - a **first** derivative of the independent rotation
field, not a second derivative of ``w`` as in Euler-Bernoulli. This is the
precise kinematic reason this theory needs only ``Continuity.C0`` (via the
plain ``gradient`` operator), not Euler-Bernoulli's ``C1``/``second_gradient``.

**Sign convention** (SDS clause E-10, matching ``euler_bernoulli.py``'s):
local axis ``x`` runs node 1 (``x=0``) to node 2 (``x=L``); ``w`` positive
local ``+y``; ``theta`` positive right-hand-rule about local ``+z``
(matching the ``r.z`` DOF component name); ``kappa = d(theta)/dx``,
``gamma = dw/dx - theta``; ``M = E I kappa``, ``V = G A_s gamma``.

**This is a selective-reduced-integration (SRI) element, not the "exact"
Phi-parametrized Timoshenko element** many textbooks and commercial codes
use (which is derived from the governing ODEs' cubic-``w``/quadratic-
``theta`` closed-form solution - a different shape-function family this
codebase's ``Interpolation`` framework does not have, and building one would
be exactly the new abstraction this project's current phase avoids). SDS
clause E-3 names the remedy this element actually implements directly:
*"Timoshenko declares selective-reduced integration of the shear term"* -
full (2-point Gauss) integration of the bending term, reduced (1-point
Gauss) integration of the shear term, avoiding shear locking without a new
shape-function family. See ``elements/structural/beam_timoshenko.py`` and
``docs/design/TIMOSHENKO_BEAM.md`` for the full derivation, the composed-path
equivalence proof, and the mesh-convergence data confirming this
formulation's correctness (a single element is *not* exact for a cantilever,
unlike ``Bar``/``EulerBernoulliBeam`` - convergence under mesh refinement is
this formulation's actual, textbook-correct guarantee).

The constitutive law here is genuinely 2-component - ``[kappa, gamma]``, SDS
Section 5's "composed generalized strain" pattern (the same pattern named
for strain-gradient/couple-stress theories, applied here for bending+shear)
- with an uncoupled diagonal tangent ``diag(E, G)``. ``I`` (second moment of
area) and ``A_s`` (shear area) are both element-layer multipliers applied
afterward, exactly as ``Bar``'s ``area``/``EulerBernoulliBeam``'s
``second_moment`` - except here there are two independent multipliers, one
per generalized-strain component, since bending and shear are physically
distinct.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.assembly.contributions import ContributionKind, OperatorRole
from nanofem.numerics.operators.base import Continuity, derived_continuity
from nanofem.physics.base import ConstitutiveModel, Locality, Theory
from nanofem.state.layout import StateLayout

_OPERATORS_USED: tuple[str, ...] = ("gradient",)


class TimoshenkoBeamTheory(Theory):
    """Timoshenko bending+shear, 1-D only; ``w``/``theta`` independent (SDS 4.1)."""

    def fields(self) -> tuple[tuple[str, int], ...]:
        """Transverse displacement ``u.y`` and rotation ``r.z`` (SDS C-2's own worked example)."""
        return (("u", 1), ("r", 1))

    def field_component_names(self) -> dict[str, tuple[str, ...]]:
        """``u`` is the *y*-component, ``r`` is the *z*-component - not the positional default."""
        return {"u": ("y",), "r": ("z",)}

    def continuity_requirements(self) -> dict[str, Continuity]:
        """``gradient`` (order 1) only requires ``Continuity.C0`` on both fields."""
        continuity = derived_continuity(self.operators_used())
        return {"u": continuity, "r": continuity}

    def required_properties(self) -> tuple[str, ...]:
        """``E`` and ``G`` - shear modulus is a real, independent material property here."""
        return ("E", "G")

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
        """Local bending+shear is cell-only."""
        return (ContributionKind.CELL,)

    def locality(self) -> Locality:
        """Local bending+shear is LOCAL, not PAIRWISE."""
        return Locality.LOCAL


class TimoshenkoBeamConstitutive(ConstitutiveModel):
    """``[M_per_I, V_per_As] = diag(E, G) @ [kappa, gamma]`` - the composed 2-component law."""

    def required_properties(self) -> tuple[str, ...]:
        """``E`` and ``G``."""
        return ("E", "G")

    def state_layout(self) -> StateLayout:
        """Elastic law: no state."""
        return StateLayout(())

    def response_components(self) -> int:
        """Two generalized components: curvature ``kappa`` and shear strain ``gamma``."""
        return 2

    def respond_batch(
        self,
        strains: NDArray[np.float64],
        properties: dict[str, NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``stress = [E*kappa, G*gamma]``; ``tangent`` is the ``(..., 2, 2)`` ``diag(E, G)``.

        Here ``strains`` is ``[kappa, gamma]``, shape ``(..., 2)``;
        ``properties["E"]``/``["G"]`` broadcast against ``strains``'s leading
        batch axes. The two responses are uncoupled - bending and shear do
        not interact in this theory - so the tangent's off-diagonals are
        exactly zero, not merely small.
        """
        e = np.asarray(properties["E"], dtype=np.float64)
        g = np.asarray(properties["G"], dtype=np.float64)
        stress = np.stack([e * strains[..., 0], g * strains[..., 1]], axis=-1)
        tangent = np.zeros(strains.shape[:-1] + (2, 2), dtype=np.float64)
        tangent[..., 0, 0] = e
        tangent[..., 1, 1] = g
        return stress, tangent
