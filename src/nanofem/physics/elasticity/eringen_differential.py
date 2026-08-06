"""Eringen differential nonlocal elasticity, dimension-generic (Stage 4).

Eringen's differential nonlocal constitutive law replaces the classical pointwise
``sigma = C:eps(u)`` with a Helmholtz-regularized relation between a *nonlocal strain* field
``e*`` and the classical (local) strain ``eps(u)``::

    e*(x) - mu * laplacian(e*(x)) = eps(u)(x)      (mu = (e0*a)^2, the nonlocal parameter)
    sigma*(x) = C : e*(x)                          (the classical law, applied to e* not eps(u))

**Why nonlocal *strain*, not nonlocal *stress*.** Eringen's own papers state the relation on
stress: ``sigma - mu*laplacian(sigma) = C:eps(u)``. For a spatially uniform (homogeneous) ``C``,
these are exactly equivalent: ``(I - mu*laplacian)(C:e*) = C:(I - mu*laplacian)e*`` since ``C``
is constant and commutes through differentiation. Choosing strain as the auxiliary field lets it
reuse the *exact* Voigt convention, shape functions, and ``PlaneStress/StrainConstitutive``
matrices `ContinuumElement` already has, with zero new tensor algebra - not an approximation,
an exact reformulation for this material class.

**Why this needs a second field, not just a new ``ConstitutiveModel``.** The 1-D nonlocal bar
(v0.20.0, ``docs/design/ERINGEN_DIFFERENTIAL_BAR.md``) eliminated the nonlocal stress in closed
form because a 1-D bar's equilibrium is a first-order ODE with a closed-form integral. General
2-D/3-D equilibrium is a divergence condition on a tensor field with no such closed-form
inversion, so ``e*`` must be carried as a genuine, independently-discretized field, coupled to
``u`` - this is a real, additive extension (a new element family), not a redesign of the
single-field ``ContinuumElement``/backbone.

**Why the weak form needs a specific ("energy-conjugate") test-function weighting.** Testing
equilibrium with ``delta_u`` and the Helmholtz relation with a *bare* ``delta_e*`` gives a
mixed system whose two off-diagonal coupling blocks are not transposes of each other (one
carries a ``C`` factor from the constitutive substitution, the other doesn't) - not symmetric,
and a non-symmetric matrix silently breaks ``ConjugateGradientSolver`` (which requires SPD).
Testing the Helmholtz relation with ``C:delta_e*`` instead (i.e. weighting the constraint
equation by the material stiffness) makes the two coupling blocks exact transposes of each
other, verified both algebraically and numerically (against the real NanoFEM quadrature stack)
before this module was written. The Helmholtz block's own *sign* in the assembled monolithic
matrix is a separate, easy-to-miss subtlety - see ``elements/continuum/nonlocal_continuum.py``'s
module docstring and ``docs/dev/notes.md`` for the full account of a sign bug caught only by a
full-pipeline solve, not by the transpose check alone. This module only declares the theory and
the constitutive law; the weak form is assembled in ``nonlocal_continuum.py``.

**Dimension independence.** Every formula here is expressed through ``VOIGT_LENGTH``/
``VOIGT_ORDER`` (``numerics/tensors/conventions.py``), never a hard-coded "3" - the same
mechanism ``IsotropicElasticity`` already uses for its own ``dim`` parameter. Only ``dim=2`` is
currently *usable* (no 3-D continuum element exists yet), but nothing here assumes it.

**Local limit.** As ``mu -> 0``, the Helmholtz relation degenerates to ``e* = eps(u)`` (an L2
projection of the local strain onto the ``e*`` shape-function space); for a linear (T3) element
whose strain field is itself constant, that projection is *exact*, so ``mu=0`` reproduces the
classical stiffness to floating-point precision, not just in a limiting sense - verified
numerically before this module was written.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.assembly.contributions import ContributionKind, OperatorRole
from nanofem.numerics.operators.base import Continuity, derived_continuity
from nanofem.numerics.tensors.conventions import VOIGT_LENGTH, VOIGT_ORDER
from nanofem.physics.base import ConstitutiveModel, Locality, Theory
from nanofem.physics.elasticity.plane import PlaneStrainConstitutive, PlaneStressConstitutive
from nanofem.state.layout import StateLayout
from nanofem.utils.exceptions import PhysicsError

_OPERATORS_USED: tuple[str, ...] = ("symmetric_gradient", "gradient", "voigt_map")

#: The field name conventions this theory declares (mirroring IsotropicElasticity's "u").
DISPLACEMENT_FIELD = "u"
NONLOCAL_STRAIN_FIELD = "e_star"

#: Spatial axis names, duplicated from core.fields' own convention rather than imported - R2's
#: layer contract forbids physics from importing core (the same reason
#: EulerBernoulliBendingTheory/TimoshenkoBeamTheory hard-code "y"/"z" as literals instead of
#: computing them).
_AXES: tuple[str, ...] = ("x", "y", "z")


def voigt_component_names(dim: int) -> tuple[str, ...]:
    """``("xx","yy","xy")`` for dim=2, ``("xx","yy","zz","yz","xz","xy")`` for dim=3.

    Built from ``VOIGT_ORDER`` and the same axis-name convention ``core.fields.component_names``
    uses for displacement fields (duplicated locally, not imported - see ``_AXES``) - never a
    hard-coded Voigt label, so a future dim=3 continuum element needs no change here.
    """
    return tuple(_AXES[i] + _AXES[j] for i, j in VOIGT_ORDER[dim])


class EringenDifferentialTheory(Theory):
    """Eringen differential nonlocal elasticity kinematics: displacement + nonlocal strain.

    Two fields: ``u`` (``dim`` components, the classical displacement) and ``e_star`` (the
    nonlocal strain, ``VOIGT_LENGTH[dim]`` components). Both use ``Continuity.C0`` - the
    Helmholtz relation's own weak form needs only first derivatives of ``e_star``, exactly
    like the classical `symmetric_gradient` needs only first derivatives of ``u``.
    """

    def __init__(self, dim: int = 2) -> None:
        if dim != 2:
            raise PhysicsError(
                f"EringenDifferentialTheory currently supports dim=2 only (no 3-D continuum "
                f"element exists yet to pair it with); dim={dim} is not yet usable"
            )
        self._dim = dim

    @property
    def dim(self) -> int:
        """The spatial dimension this instance was constructed with."""
        return self._dim

    def fields(self) -> tuple[tuple[str, int], ...]:
        """``u`` with ``dim`` components, ``e_star`` with ``VOIGT_LENGTH[dim]`` components."""
        return (
            (DISPLACEMENT_FIELD, self._dim),
            (NONLOCAL_STRAIN_FIELD, VOIGT_LENGTH[self._dim]),
        )

    def field_component_names(self) -> dict[str, tuple[str, ...]]:
        """``e_star``'s components are named by Voigt pair (``xx``, ``yy``, ``xy``, ...), not
        the positional ``x``/``y``/``z`` default, which is only correct for spatial fields."""
        return {NONLOCAL_STRAIN_FIELD: voigt_component_names(self._dim)}

    def continuity_requirements(self) -> dict[str, Continuity]:
        """Both fields need only ``Continuity.C0`` (SDS Section 8 rule)."""
        continuity = derived_continuity(self.operators_used())
        return {DISPLACEMENT_FIELD: continuity, NONLOCAL_STRAIN_FIELD: continuity}

    def required_properties(self) -> tuple[str, ...]:
        """``E``, ``nu`` (the local law) and ``e0a`` (``mu = (e0*a)^2``)."""
        return ("E", "nu", "e0a")

    def required_state(self) -> StateLayout:
        """Elastic law: no history state (SDS Section 7). ``e_star`` is a genuine extra
        field, not per-quadrature-point internal state."""
        return StateLayout(())

    def operators_used(self) -> tuple[str, ...]:
        """``symmetric_gradient`` for ``u``'s strain, ``gradient`` for ``e_star``'s Laplacian
        term (via its own gradient, integrated by parts in the weak form)."""
        return _OPERATORS_USED

    def operator_roles(self) -> tuple[OperatorRole, ...]:
        """Only STIFFNESS/FORCE - dynamics (MASS) is explicitly out of Stage 4's scope."""
        return (OperatorRole.STIFFNESS, OperatorRole.FORCE)

    def contribution_kinds(self) -> tuple[ContributionKind, ...]:
        """Local nonlocal-elasticity is cell-only."""
        return (ContributionKind.CELL,)

    def locality(self) -> Locality:
        """LOCAL, not PAIRWISE - despite the physics being "nonlocal," this is architecturally
        a local, multi-field theory (the Helmholtz PDE is solved by a standard local weak form,
        the same way EulerBernoulliBendingTheory's two coupled fields are local). PAIRWISE is
        reserved for the *integral* nonlocal model (kernel + neighbor search), explicitly out
        of scope this phase."""
        return Locality.LOCAL


class EringenDifferentialMaterial(ConstitutiveModel):
    """``sigma* = C : e_star`` - the classical elasticity law, applied to the nonlocal strain.

    Delegates entirely to a wrapped classical plane law (``PlaneStressConstitutive`` or
    ``PlaneStrainConstitutive``) - the defining property of Eringen's differential model is
    that ``C`` itself is *unchanged*; only which strain it acts on changes. No formula is
    duplicated. ``e0a`` is declared as a required property (so
    ``Model.validate()`` checks the material defines it) but is not consumed by
    ``respond_batch`` itself - it is a *kinematic coupling* parameter (used to build the
    Helmholtz weak form in `elements/continuum/nonlocal_continuum.py`), not part of the
    pointwise stress-strain map this interface represents.
    """

    def __init__(self, local_law: PlaneStressConstitutive | PlaneStrainConstitutive) -> None:
        self._local_law = local_law

    @property
    def local_law(self) -> PlaneStressConstitutive | PlaneStrainConstitutive:
        """The wrapped classical law - read by the element to build the D matrix."""
        return self._local_law

    def required_properties(self) -> tuple[str, ...]:
        """The wrapped law's own properties, plus ``e0a``."""
        return (*self._local_law.required_properties(), "e0a")

    def state_layout(self) -> StateLayout:
        """Elastic law: no state (delegated)."""
        return self._local_law.state_layout()

    def response_components(self) -> int:
        """Delegated: the same Voigt length the wrapped law already declares."""
        return self._local_law.response_components()

    def respond_batch(
        self,
        strains: NDArray[np.float64],
        properties: dict[str, NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``stress = D @ strains`` via the wrapped law - ``e0a`` is stripped
        before delegating, since the wrapped law knows nothing about it."""
        local_properties = {k: v for k, v in properties.items() if k != "e0a"}
        return self._local_law.respond_batch(strains, local_properties)
