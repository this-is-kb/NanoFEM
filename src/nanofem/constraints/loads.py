"""Load descriptions (SDS 2.14; Section 10): body, surface, point, distributed.

SDS naming with the phase-1 aliases requested by the object model:
NodalLoad = point load (a VERTEX FORCE provider in phase 2), TractionLoad =
surface load (FACET), LineLoad = distributed member load, BodyForce = body
load (CELL). ``NonlocalAxialLoad``/``NonlocalTransverseLoad`` (Stage 4) are narrower kinds: the
distributed-load corrections Eringen's differential nonlocal elasticity adds for a 1-D bar and
Euler-Bernoulli beam respectively - see ``constraints/nonlocal_load.py`` and
``docs/design/ERINGEN_DIFFERENTIAL_BAR.md``/``ERINGEN_DIFFERENTIAL_BEAM.md`` for the derivations.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.core.dof_handler import DofHandler
from nanofem.mesh.mesh import Mesh
from nanofem.numerics.assembly.contributions import Contribution, ContributionKind, OperatorRole
from nanofem.utils.exceptions import InputValidationError
from nanofem.utils.validation import require_finite_array, require_identifier, require_non_negative


def _validated_vector(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 1:
        raise InputValidationError(f"{name} must be a 1-D component vector, got shape {arr.shape}")
    require_finite_array(arr, name)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, eq=False)
class NodalLoad:
    """Concentrated generalized forces on the nodes of a node region (point load)."""

    region: str
    field: str
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        require_identifier(self.region, "NodalLoad region")
        require_identifier(self.field, "NodalLoad field")
        object.__setattr__(self, "values", _validated_vector(self.values, "NodalLoad values"))


@dataclass(frozen=True, eq=False)
class NodalLoadProvider:
    """A ``NodalLoad`` as a VERTEX FORCE ``ContributionProvider`` (SDS Section 10).

    ``field_components`` names, in order, which components ``load.values``
    supplies (matching the theory's field declaration) - the load itself
    carries only a bare vector, so the provider is what resolves it against a
    specific DOF numbering. Satisfies ``ContributionProvider`` structurally
    (the protocol is ``@runtime_checkable``); no inheritance is needed.
    """

    load: NodalLoad
    mesh: Mesh
    dof_handler: DofHandler
    field_components: tuple[str, ...]
    factor: float = 1.0

    def contributions(self, role: OperatorRole) -> Iterator[Contribution]:
        """Yield one VERTEX FORCE block per node in the load's region."""
        if role is not OperatorRole.FORCE:
            return
        block = self.factor * self.load.values
        for node_id in self.mesh.nodes_in_region(self.load.region):
            dofs = np.array(
                [
                    self.dof_handler.global_dof(node_id, self.load.field, component)
                    for component in self.field_components
                ],
                dtype=np.int64,
            )
            yield Contribution(ContributionKind.VERTEX, role, dofs, None, block)


@dataclass(frozen=True, eq=False)
class LineLoad:
    """Distributed load intensity on structural members of a cell region."""

    region: str
    field: str
    intensity: NDArray[np.float64]

    def __post_init__(self) -> None:
        require_identifier(self.region, "LineLoad region")
        require_identifier(self.field, "LineLoad field")
        intensity = _validated_vector(self.intensity, "LineLoad intensity")
        object.__setattr__(self, "intensity", intensity)


@dataclass(frozen=True, eq=False)
class TractionLoad:
    """Surface/edge traction on a boundary region (surface load)."""

    region: str
    field: str
    traction: NDArray[np.float64]

    def __post_init__(self) -> None:
        require_identifier(self.region, "TractionLoad region")
        require_identifier(self.field, "TractionLoad field")
        traction = _validated_vector(self.traction, "TractionLoad traction")
        object.__setattr__(self, "traction", traction)


@dataclass(frozen=True, eq=False)
class BodyForce:
    """Domain force density on a cell region (body load)."""

    region: str
    field: str
    density: NDArray[np.float64]

    def __post_init__(self) -> None:
        require_identifier(self.region, "BodyForce region")
        require_identifier(self.field, "BodyForce field")
        object.__setattr__(self, "density", _validated_vector(self.density, "BodyForce density"))


@dataclass(frozen=True, eq=False)
class NonlocalAxialLoad:
    """A spatially-varying distributed axial load, plus Eringen's nonlocal length parameter.

    Eringen's differential nonlocal elasticity for a 1-D bar (``sigma - mu*sigma'' = E*u'``,
    ``mu = (e0*a)^2``, combined with axial equilibrium ``A*sigma' = -q(x)``) eliminates to a
    single 2nd-order ODE in ``u`` alone: ``A*E*u'' = mu*q''(x) - q(x)`` (see
    ``docs/design/ERINGEN_DIFFERENTIAL_BAR.md`` for the full derivation, verified two
    independent symbolic ways plus a numerical mesh-convergence check before this class was
    written). Its weak form needs *no change* to the classical bar stiffness - the nonlocal
    effect is entirely a load correction, ``mu * integral(dN_a/dx * q'(x)) dx``, added to the
    classical consistent load ``integral(N_a * q(x)) dx``. That is exactly what
    ``NonlocalAxialLoadProvider`` (``constraints/nonlocal_load.py``) computes.

    ``q(x)`` is supplied by *nodal sample*, not as a callable (frozen dataclasses hold no
    functions, matching every other load in this module) - ``nodal_intensity[node_id]`` is
    ``q`` at that global node's coordinate, interpolated element-wise via the same linear
    shape functions the field itself uses (the standard, serializable FEM convention for
    distributed load data). ``nonlocal_intensity`` must have one entry per mesh node.

    A uniform or purely-linear ``q(x)`` (``q'' = 0`` identically) produces *zero* nonlocal
    correction by this formula - not a bug, but a well-documented property of Eringen's
    differential model for 1-D members (the "Peddieson paradox": concentrated end loads and
    uniform distributed loads leave the differential-model displacement field identical to
    the classical local solution). A discriminating benchmark needs ``q''(x) != 0``.
    """

    region: str
    field: str
    nodal_intensity: NDArray[np.float64]
    nonlocal_parameter: float

    def __post_init__(self) -> None:
        require_identifier(self.region, "NonlocalAxialLoad region")
        require_identifier(self.field, "NonlocalAxialLoad field")
        intensity = np.asarray(self.nodal_intensity, dtype=np.float64)
        if intensity.ndim != 1 or intensity.size < 1:
            raise InputValidationError(
                f"NonlocalAxialLoad nodal_intensity must be a 1-D array, got shape "
                f"{intensity.shape}"
            )
        require_finite_array(intensity, "NonlocalAxialLoad nodal_intensity")
        intensity.setflags(write=False)
        object.__setattr__(self, "nodal_intensity", intensity)
        require_non_negative(self.nonlocal_parameter, "NonlocalAxialLoad nonlocal_parameter")


@dataclass(frozen=True, eq=False)
class NonlocalTransverseLoad:
    """A spatially-varying distributed transverse load, plus Eringen's nonlocal parameter.

    The Euler-Bernoulli-beam analogue of ``NonlocalAxialLoad``: Eringen's differential nonlocal
    relation applied to bending (``M* - mu*M*'' = E*I*kappa``, ``kappa = w''``), combined with
    beam equilibrium (``M*'' = q(x)``, this codebase's own sign convention - see
    ``physics/elasticity/euler_bernoulli.py``), eliminates to ``E*I*w'''' = q(x) - mu*q''(x)``
    (see ``docs/design/ERINGEN_DIFFERENTIAL_BEAM.md`` for the full derivation, verified two
    independent symbolic ways plus a numerical weak-form check against the real Hermite
    interpolation stack before this class was written). Its weak form needs *no change* to the
    classical ``EulerBernoulliBeam`` stiffness - the nonlocal effect is entirely a load
    correction, ``mu * integral(dN_a/dx * q'(x)) dx`` (using the *physical*-rotation-corrected
    shape function derivatives, exactly as the classical stiffness itself does), added to the
    classical consistent load ``integral(N_a * q(x)) dx``. That is exactly what
    ``NonlocalTransverseLoadProvider`` (``constraints/nonlocal_load.py``) computes.

    ``q(x)`` is supplied by nodal sample (one ``w``-conjugate value per mesh node; the rotation
    -conjugate entries of the underlying Hermite interpolation are taken as zero, matching
    ``NonlocalAxialLoad``'s "nodal sample, not a callable" convention).

    A uniform or purely-linear ``q(x)`` (``q'' = 0``) produces *zero* nonlocal correction - the
    beam analogue of the 1-D bar's Peddieson-paradox null effect (v0.20.0). A discriminating
    benchmark needs ``q''(x) != 0`` (e.g. a sinusoidal load).
    """

    region: str
    field: str
    nodal_intensity: NDArray[np.float64]
    nonlocal_parameter: float

    def __post_init__(self) -> None:
        require_identifier(self.region, "NonlocalTransverseLoad region")
        require_identifier(self.field, "NonlocalTransverseLoad field")
        intensity = np.asarray(self.nodal_intensity, dtype=np.float64)
        if intensity.ndim != 1 or intensity.size < 1:
            raise InputValidationError(
                f"NonlocalTransverseLoad nodal_intensity must be a 1-D array, got shape "
                f"{intensity.shape}"
            )
        require_finite_array(intensity, "NonlocalTransverseLoad nodal_intensity")
        intensity.setflags(write=False)
        object.__setattr__(self, "nodal_intensity", intensity)
        require_non_negative(self.nonlocal_parameter, "NonlocalTransverseLoad nonlocal_parameter")
