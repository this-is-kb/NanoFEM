"""Timoshenko beam element: closed-form structural family (SDS Section 3, ADR-002).

The stiffness IS the discrete weak form (E-5) of the selective-reduced-
integration (SRI) formulation - full (2-point Gauss) integration of the
bending term, reduced (1-point Gauss) integration of the shear term, per SDS
clause E-3 - not derived through the composed operators+constitutive path at
run time (that equivalence is instead proven independently, in a
verification test, per ADR-002's "declare equivalence... verification tests
enforce it"; see ``physics/elasticity/timoshenko.py`` and
``docs/design/TIMOSHENKO_BEAM.md`` for the full derivation and the
mesh-convergence data confirming this closed form's correctness).

``K = [[G*As/L, G*As/2, -G*As/L, G*As/2],
[G*As/2, E*I/L+G*As*L/4, -G*As/2, -E*I/L+G*As*L/4],
[-G*As/L, -G*As/2, G*As/L, -G*As/2],
[G*As/2, -E*I/L+G*As*L/4, -G*As/2, E*I/L+G*As*L/4]]``
in DOF order ``(w1, theta1, w2, theta2)``.

This phase covers only a beam embedded in 1-D space (pure bending+shear, no
axial coupling), so the local and global axes coincide and the E-10
transformation is the honestly trivial identity - not a stand-in for missing
rotation logic. ``Frame2D`` (``elements/structural/frame.py``) owns the real
direction-cosine transformation and axial coupling for members embedded in
higher-dimensional space.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from nanofem.elements.base import ElementDofSignature, StructuralElement
from nanofem.numerics.assembly.contributions import Contribution, ContributionKind, OperatorRole
from nanofem.utils.exceptions import InputValidationError
from nanofem.utils.validation import require_positive


class TimoshenkoBeam(StructuralElement):
    """Two-node Timoshenko beam, embedded in 1-D space (SDS Section 3; ADR-002)."""

    def __init__(
        self,
        cell_id: int,
        node_ids: tuple[int, int],
        coordinates: NDArray[np.float64],
        global_dofs: tuple[int, int, int, int],
        young_modulus: float,
        shear_modulus: float,
        second_moment: float,
        shear_area: float,
    ) -> None:
        require_positive(young_modulus, "young_modulus")
        require_positive(shear_modulus, "shear_modulus")
        require_positive(second_moment, "second_moment")
        require_positive(shear_area, "shear_area")
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.shape != (2, 1):
            raise InputValidationError(
                f"TimoshenkoBeam cell {cell_id}: coordinates must be (2, 1) "
                f"(1-D embedding only), got {coords.shape}"
            )
        length = float(coords[1, 0] - coords[0, 0])
        if length <= 0.0:
            raise InputValidationError(
                f"TimoshenkoBeam cell {cell_id}: node order must give a positive length, "
                f"got {length}"
            )
        self.cell_id = cell_id
        self.node_ids = node_ids
        self.coordinates = coords
        self.global_dofs = global_dofs
        self.young_modulus = young_modulus
        self.shear_modulus = shear_modulus
        self.second_moment = second_moment
        self.shear_area = shear_area
        self.length = length

    def local_stiffness(self) -> NDArray[np.float64]:
        """The verified selective-reduced-integration stiffness, DOF order ``(w1,th1,w2,th2)``."""
        length = self.length
        ei = self.young_modulus * self.second_moment
        gas = self.shear_modulus * self.shear_area
        return np.array(
            [
                [gas / length, gas / 2.0, -gas / length, gas / 2.0],
                [
                    gas / 2.0,
                    ei / length + gas * length / 4.0,
                    -gas / 2.0,
                    -ei / length + gas * length / 4.0,
                ],
                [-gas / length, -gas / 2.0, gas / length, -gas / 2.0],
                [
                    gas / 2.0,
                    -ei / length + gas * length / 4.0,
                    -gas / 2.0,
                    ei / length + gas * length / 4.0,
                ],
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
        ), "TimoshenkoBeam transformation must be orthonormal (E-10)"
        return t

    def contributions(self, role: OperatorRole) -> Iterator[Contribution]:
        """Emit the closed-form CELL stiffness block; nothing for any other role (E-6)."""
        if role is not OperatorRole.STIFFNESS:
            return
        dofs = np.array(self.global_dofs, dtype=np.int64)
        yield Contribution(ContributionKind.CELL, role, dofs, dofs, self.local_stiffness())
