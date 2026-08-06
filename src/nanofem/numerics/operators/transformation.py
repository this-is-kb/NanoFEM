"""The transformation operator: frame rotations of vectors and Voigt-form tensors (SDS Section 8).

A thin dispatcher over ``nanofem.numerics.tensors.rotations`` - the Bond
transformation pair lives there because it is tensor algebra, not a
discretization concern; this module exists so structural elements and
orthotropic material frames have a single named entry point in the operator
vocabulary, matching every other SDS Section 8 row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.operators.base import OPERATOR_DERIVATIVE_ORDER, DiscreteOperator
from nanofem.numerics.operators.errors import OperatorError
from nanofem.numerics.tensors.errors import NotRotationError
from nanofem.numerics.tensors.rotations import (
    bond_matrix_strain,
    bond_matrix_stress,
    is_rotation,
    rotate_vector,
)
from nanofem.utils.exceptions import InputValidationError

TransformationKind = Literal["stress", "strain"]


def vector_transformation_matrix(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """``T = Q``: the frame-rotation matrix for a vector field is the rotation itself."""
    if not is_rotation(q):
        raise NotRotationError(f"q of shape {q.shape} is not a member of SO(d)")
    return np.asarray(q, dtype=np.float64)


def tensor_transformation_matrix_voigt(
    q: NDArray[np.float64], *, kind: TransformationKind
) -> NDArray[np.float64]:
    """The Bond matrix for rotating a Voigt-form stress or strain by ``Q``."""
    if kind == "stress":
        return bond_matrix_stress(q)
    if kind == "strain":
        return bond_matrix_strain(q)
    raise InputValidationError(f"kind must be 'stress' or 'strain', got {kind!r}")


@dataclass(frozen=True)
class TransformationOperator(DiscreteOperator):
    """``DiscreteOperator`` wrapper around vector and Voigt-form frame rotations."""

    def name(self) -> str:
        return "transformation"

    def required_derivative_order(self) -> int:
        return OPERATOR_DERIVATIVE_ORDER[self.name()]

    def verify(self) -> None:
        """A 90-degree rotation matches a hand-built vector rotation and round-trips a stiffness."""
        angle = np.pi / 2
        q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        t_vector = vector_transformation_matrix(q)
        if not np.allclose(rotate_vector(q, np.array([1.0, 0.0])), t_vector @ np.array([1.0, 0.0])):
            raise OperatorError("vector_transformation_matrix disagreed with rotate_vector")

        d_voigt = np.diag([10.0, 12.0, 3.0])
        m_sigma = tensor_transformation_matrix_voigt(q, kind="stress")
        m_epsilon = tensor_transformation_matrix_voigt(q, kind="strain")
        rotated = m_sigma @ d_voigt @ m_sigma.T
        rotated_back = np.linalg.inv(m_sigma) @ rotated @ np.linalg.inv(m_sigma).T
        if not np.allclose(rotated_back, d_voigt):
            raise OperatorError("stiffness rotation did not round-trip back to the original")
        if not np.allclose(m_epsilon, np.linalg.inv(m_sigma).T):
            raise OperatorError("tensor_transformation_matrix_voigt('strain') != M_sigma^-T")

    def is_valid(self) -> bool:
        try:
            self.verify()
        except (OperatorError, InputValidationError, NotRotationError):
            return False
        return True
