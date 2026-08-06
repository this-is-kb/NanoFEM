"""Plane stress / plane strain constitutive reductions, 2-D isotropic elasticity (SDS 4.1).

Both laws pair with the same kinematics, ``IsotropicElasticity(dim=2)``
(`physics/elasticity/isotropic.py`) - plane stress vs. plane strain is purely
a *constitutive-law* distinction, not a kinematic one, which is exactly why
these are two small, separate ``ConstitutiveModel`` classes rather than a
change to the Theory: a future nonlocal (Eringen) constitutive law swaps in
against the same kinematics the same way.

**Convention**: generalized strain is the kinematic Voigt vector
``[eps_xx, eps_yy, gamma_xy]`` (``gamma_xy`` the engineering shear,
``VOIGT_ORDER[2]`` in ``numerics/tensors/conventions.py`` - the same
convention ``symmetric_gradient_matrix`` already produces for ``dim=2``, so
no operator-layer change was needed to support either law). Thickness is an
element-layer multiplier (``PlaneGeometry.thickness``, ``geometry/plane.py``),
applied via ``ContinuumElement``'s existing ``section_measure`` parameter -
exactly as ``Bar``'s ``area``/``EulerBernoulliBeam``'s ``second_moment`` are
applied - not part of either law here.

**Plane stress** (thin, out-of-plane-stress-free): ``sigma_zz = 0``,
``eps_zz`` free.
``D = E/(1-nu^2) * [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]]``

**Plane strain** (thick/constrained, out-of-plane-strain-free): ``eps_zz = 0``,
``sigma_zz`` free.
``D = E/((1+nu)(1-2nu)) * [[1-nu, nu, 0], [nu, 1-nu, 0], [0, 0, (1-2nu)/2]]``

Both verified this session via a constant-strain patch test through
``ContinuumElement`` (a unit right triangle, prescribed linear displacement
field, strain energy ``0.5 u^T K u`` matched against the analytical
``0.5 (eps^T D eps) Area thickness`` to ``rtol=1e-9`` for both a normal-strain
and a pure-shear case) - see ``docs/design/PLANE_ELASTICITY.md``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nanofem.physics.base import ConstitutiveModel
from nanofem.state.layout import StateLayout


class PlaneStressConstitutive(ConstitutiveModel):
    """``D = E/(1-nu^2) * [[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]]`` - the plane stress reduction."""

    def required_properties(self) -> tuple[str, ...]:
        """``E`` and ``nu``."""
        return ("E", "nu")

    def state_layout(self) -> StateLayout:
        """Elastic law: no state."""
        return StateLayout(())

    def response_components(self) -> int:
        """Three Voigt components: ``eps_xx``, ``eps_yy``, ``gamma_xy``."""
        return 3

    def respond_batch(
        self,
        strains: NDArray[np.float64],
        properties: dict[str, NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``stress = D @ strains``; ``strains`` has shape ``(..., 3)``."""
        e = np.asarray(properties["E"], dtype=np.float64)
        nu = np.asarray(properties["nu"], dtype=np.float64)
        c = e / (1.0 - nu**2)
        tangent = np.zeros(strains.shape[:-1] + (3, 3), dtype=np.float64)
        tangent[..., 0, 0] = c
        tangent[..., 0, 1] = c * nu
        tangent[..., 1, 0] = c * nu
        tangent[..., 1, 1] = c
        tangent[..., 2, 2] = c * (1.0 - nu) / 2.0
        stress = np.einsum("...ij,...j->...i", tangent, strains)
        return stress, tangent


class PlaneStrainConstitutive(ConstitutiveModel):
    """``D = E/((1+nu)(1-2nu)) [[1-nu,nu,0],[nu,1-nu,0],[0,0,(1-2nu)/2]]`` - plane strain."""

    def required_properties(self) -> tuple[str, ...]:
        """``E`` and ``nu``."""
        return ("E", "nu")

    def state_layout(self) -> StateLayout:
        """Elastic law: no state."""
        return StateLayout(())

    def response_components(self) -> int:
        """Three Voigt components: ``eps_xx``, ``eps_yy``, ``gamma_xy``."""
        return 3

    def respond_batch(
        self,
        strains: NDArray[np.float64],
        properties: dict[str, NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``stress = D @ strains``; ``strains`` has shape ``(..., 3)``."""
        e = np.asarray(properties["E"], dtype=np.float64)
        nu = np.asarray(properties["nu"], dtype=np.float64)
        c = e / ((1.0 + nu) * (1.0 - 2.0 * nu))
        tangent = np.zeros(strains.shape[:-1] + (3, 3), dtype=np.float64)
        tangent[..., 0, 0] = c * (1.0 - nu)
        tangent[..., 0, 1] = c * nu
        tangent[..., 1, 0] = c * nu
        tangent[..., 1, 1] = c * (1.0 - nu)
        tangent[..., 2, 2] = c * (1.0 - 2.0 * nu) / 2.0
        stress = np.einsum("...ij,...j->...i", tangent, strains)
        return stress, tangent
