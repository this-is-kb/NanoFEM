"""Euler-Bernoulli beam element: closed-form structural family (SDS Section 3, ADR-002).

The stiffness IS the discrete weak form (E-5):
``K = (EI/L^3) [[12,6L,-12,6L],[6L,4L^2,-6L,2L^2],[-12,-6L,12,-6L],
[6L,2L^2,-6L,4L^2]]`` in the member's own axis, per the sign convention
documented in ``physics/elasticity/euler_bernoulli.py`` - not derived through
the composed operators+constitutive path (that equivalence is instead proven
independently, in a verification test, per ADR-002's "declare equivalence...
verification tests enforce it").

This phase covers only a beam embedded in 1-D space (pure bending, no axial-
flexural coupling), so the local and global axes coincide and the E-10
transformation is the honestly trivial identity - not a stand-in for missing
rotation logic. ``Frame2D`` (``elements/structural/frame.py``) owns the real
direction-cosine transformation and axial+bending coupling for members
embedded in higher-dimensional space.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.elements.base import ElementDofSignature, StructuralElement
from nanofem.numerics.assembly.contributions import Contribution, ContributionKind, OperatorRole
from nanofem.utils.exceptions import InputValidationError
from nanofem.utils.validation import require_positive


@dataclass(frozen=True)
class BendingResponse:
    """A beam's recovered curvature/moment at both end nodes, shape ``(2,)`` each.

    Curvature is *linear* within a single cubic-Hermite element (curvature is
    the 2nd derivative of a cubic, hence linear) - so its value at the two end
    nodes captures the entire element's curvature field exactly, with no
    approximation and no quadrature needed, the same closed-form spirit as
    ``local_stiffness()`` itself.
    """

    curvature: NDArray[np.float64]
    moment: NDArray[np.float64]


class EulerBernoulliBeam(StructuralElement):
    """Two-node Euler-Bernoulli beam, embedded in 1-D space (SDS Section 3; ADR-002)."""

    def __init__(
        self,
        cell_id: int,
        node_ids: tuple[int, int],
        coordinates: NDArray[np.float64],
        global_dofs: tuple[int, int, int, int],
        young_modulus: float,
        second_moment: float,
    ) -> None:
        require_positive(young_modulus, "young_modulus")
        require_positive(second_moment, "second_moment")
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.shape != (2, 1):
            raise InputValidationError(
                f"EulerBernoulliBeam cell {cell_id}: coordinates must be (2, 1) "
                f"(1-D embedding only), got {coords.shape}"
            )
        length = float(coords[1, 0] - coords[0, 0])
        if length <= 0.0:
            raise InputValidationError(
                f"EulerBernoulliBeam cell {cell_id}: node order must give a positive length, "
                f"got {length}"
            )
        self.cell_id = cell_id
        self.node_ids = node_ids
        self.coordinates = coords
        self.global_dofs = global_dofs
        self.young_modulus = young_modulus
        self.second_moment = second_moment
        self.length = length

    def local_stiffness(self) -> NDArray[np.float64]:
        """The classical ``EI/L^3`` bending stiffness, DOF order ``(w1, theta1, w2, theta2)``."""
        length = self.length
        ei = self.young_modulus * self.second_moment
        return (ei / length**3) * np.array(
            [
                [12.0, 6.0 * length, -12.0, 6.0 * length],
                [6.0 * length, 4.0 * length**2, -6.0 * length, 2.0 * length**2],
                [-12.0, -6.0 * length, 12.0, -6.0 * length],
                [6.0 * length, 2.0 * length**2, -6.0 * length, 4.0 * length**2],
            ]
        )

    def dof_signature(self) -> ElementDofSignature:
        """Transverse displacement and rotation per node, named ``field.component`` (E-1)."""
        return ElementDofSignature((("u.y", "r.z"), ("u.y", "r.z")))

    def transformation_matrix(self) -> NDArray[np.float64]:
        """Identity: local and global axes coincide for a 1-D-in-1-D beam (E-10)."""
        t = np.eye(4)
        assert np.allclose(
            t.T @ t, np.eye(4)
        ), "EulerBernoulliBeam transformation must be orthonormal (E-10)"
        return t

    def contributions(self, role: OperatorRole) -> Iterator[Contribution]:
        """Emit the closed-form CELL stiffness block; nothing for any other role (E-6)."""
        if role is not OperatorRole.STIFFNESS:
            return
        dofs = np.array(self.global_dofs, dtype=np.int64)
        yield Contribution(ContributionKind.CELL, role, dofs, dofs, self.local_stiffness())

    def curvature_response(self, local_displacement: NDArray[np.float64]) -> BendingResponse:
        """Recover curvature/moment at both end nodes from ``(w1, theta1, w2, theta2)``.

        ``kappa(xi) = (1/L^2) * d2N/dxi^2 . d``, the classical cubic-Hermite
        curvature formula in terms of the *physical* rotation DOFs this
        element's own ``local_stiffness()`` is already expressed in (not the
        reference-parametrized ``dw/dxi`` the shape-function library's own
        "derivative" DOFs mean - see N-53's Jacobian-scaling correction,
        which does not apply here since this closed-form path never touches
        that pipeline). Verified against the classical cantilever result
        (``M(fixed end) = P*L`` for a tip load ``P``) and an independent
        finite-difference curvature check before being written.
        """
        d = np.asarray(local_displacement, dtype=np.float64)
        if d.shape != (4,):
            raise InputValidationError(
                f"EulerBernoulliBeam cell {self.cell_id}: local_displacement must have shape "
                f"(4,), got {d.shape}"
            )
        length = self.length
        xi = np.array([0.0, 1.0])
        d2n1 = -6.0 + 12.0 * xi
        d2n2 = length * (-4.0 + 6.0 * xi)
        d2n3 = 6.0 - 12.0 * xi
        d2n4 = length * (-2.0 + 6.0 * xi)
        curvature = (d2n1 * d[0] + d2n2 * d[1] + d2n3 * d[2] + d2n4 * d[3]) / length**2
        moment = self.young_modulus * self.second_moment * curvature
        return BendingResponse(curvature=curvature, moment=moment)
