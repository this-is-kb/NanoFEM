"""Bar (axial truss) element: closed-form structural family (SDS Section 3, ADR-002).

The stiffness IS the discrete weak form (E-5): ``K = (EA/L) [[1, -1], [-1, 1]]``
in the member's own axis - not derived through the composed operators+
constitutive path (that equivalence is instead proven independently, in a
verification test, per ADR-002's "declare equivalence... verification tests
enforce it").

This phase covers only a bar embedded in 1-D space, so the local and global
axes coincide and the E-10 transformation is the honestly trivial identity -
not a stand-in for missing rotation logic. ``Truss2D``/``Frame2D``
(``elements/structural/{truss,frame}.py``) own the real direction-cosine
transformation for members embedded in higher-dimensional space.
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
class AxialResponse:
    """A bar's recovered internal state - constant along its length (E-5's closed form)."""

    strain: float
    stress: float
    force: float


class Bar(StructuralElement):
    """Two-node axial bar, embedded in 1-D space (SDS Section 3; ADR-002)."""

    def __init__(
        self,
        cell_id: int,
        node_ids: tuple[int, int],
        coordinates: NDArray[np.float64],
        global_dofs: tuple[int, int],
        young_modulus: float,
        area: float,
    ) -> None:
        require_positive(young_modulus, "young_modulus")
        require_positive(area, "area")
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.shape != (2, 1):
            raise InputValidationError(
                f"Bar cell {cell_id}: coordinates must be (2, 1) (1-D embedding only), "
                f"got {coords.shape}"
            )
        length = float(coords[1, 0] - coords[0, 0])
        if length <= 0.0:
            raise InputValidationError(
                f"Bar cell {cell_id}: node order must give a positive length, got {length}"
            )
        self.cell_id = cell_id
        self.node_ids = node_ids
        self.coordinates = coords
        self.global_dofs = global_dofs
        self.young_modulus = young_modulus
        self.area = area
        self.length = length

    def local_stiffness(self) -> NDArray[np.float64]:
        """``K = (EA/L) [[1, -1], [-1, 1]]`` - the discrete weak form (ADR-002, E-5)."""
        k = self.young_modulus * self.area / self.length
        return k * np.array([[1.0, -1.0], [-1.0, 1.0]])

    def dof_signature(self) -> ElementDofSignature:
        """One axial DOF per node, named ``field.component`` (E-1)."""
        return ElementDofSignature((("u.x",), ("u.x",)))

    def transformation_matrix(self) -> NDArray[np.float64]:
        """Identity: local and global axes coincide for a 1-D-in-1-D bar (E-10)."""
        t = np.eye(2)
        assert np.allclose(t.T @ t, np.eye(2)), "Bar transformation must be orthonormal (E-10)"
        return t

    def contributions(self, role: OperatorRole) -> Iterator[Contribution]:
        """Emit the closed-form CELL stiffness block; nothing for any other role (E-6)."""
        if role is not OperatorRole.STIFFNESS:
            return
        dofs = np.array(self.global_dofs, dtype=np.int64)
        yield Contribution(ContributionKind.CELL, role, dofs, dofs, self.local_stiffness())

    def axial_response(self, local_displacement: NDArray[np.float64]) -> AxialResponse:
        """Recover strain/stress/force from the two local axial DOFs ``(u1, u2)``.

        Exact and constant along the member's length - a two-node bar's strain
        field is exactly ``(u2 - u1) / L`` by construction (E-5's closed form
        *is* the discrete weak form), not an approximation needing a recovery
        scheme the way a continuum element's quadrature-point field does.
        """
        displacement = np.asarray(local_displacement, dtype=np.float64)
        if displacement.shape != (2,):
            raise InputValidationError(
                f"Bar cell {self.cell_id}: local_displacement must have shape (2,), got "
                f"{displacement.shape}"
            )
        strain = float((displacement[1] - displacement[0]) / self.length)
        stress = self.young_modulus * strain
        force = stress * self.area
        return AxialResponse(strain=strain, stress=stress, force=force)
