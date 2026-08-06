"""The second gradient (Hessian) operator ``u -> grad(grad(u))`` (SDS Section 8).

Deliberately thin: :meth:`GeometricMapping.physical_hessian` already computes
exactly ``grad(grad(N))``, the object SDS calls ``B_H``. The only value this
module adds is validating the input's shape/rank contract and declaring the
continuity consequence (consuming this operator forces ``Continuity.C1`` via
``derived_continuity`` in ``operators/base.py``). The specific contraction a
strain-gradient or couple-stress theory needs from this raw Hessian - the
curvature tensor, the higher-order strain measure - is theory-specific and
belongs with ``physics/strain_gradient``/``physics/couple_stress``, neither of
which exists yet; inventing that contraction now would be exactly the kind of
ahead-of-a-consumer work this project's phase discipline forbids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.operators.base import OPERATOR_DERIVATIVE_ORDER, DiscreteOperator
from nanofem.numerics.operators.errors import OperatorError, OperatorShapeError
from nanofem.utils.exceptions import InputValidationError


def second_gradient_tensor(physical_hessians: NDArray[np.float64]) -> NDArray[np.float64]:
    """Validated pass-through of the physical Hessian batch, shape ``(n_qp, n_fun, dim, dim)``."""
    if physical_hessians.ndim != 4:
        raise OperatorShapeError(
            "physical_hessians must have shape (n_qp, n_fun, dim, dim), got "
            f"{physical_hessians.shape}"
        )
    if physical_hessians.shape[-1] != physical_hessians.shape[-2]:
        raise OperatorShapeError(
            "the trailing two axes of physical_hessians must be equal, got "
            f"{physical_hessians.shape}"
        )
    return np.asarray(physical_hessians, dtype=np.float64)


@dataclass(frozen=True)
class SecondGradientOperator(DiscreteOperator):
    """``DiscreteOperator`` wrapper around :func:`second_gradient_tensor`."""

    def name(self) -> str:
        return "second_gradient"

    def required_derivative_order(self) -> int:
        return OPERATOR_DERIVATIVE_ORDER[self.name()]

    def evaluate(self, physical_hessians: NDArray[np.float64]) -> NDArray[np.float64]:
        return second_gradient_tensor(physical_hessians)

    def verify(self) -> None:
        """The pass-through is exact, and a malformed (non-square trailing) batch is refused."""
        hessians = np.zeros((2, 3, 2, 2))
        hessians[0, 0, 0, 1] = 5.0
        result = self.evaluate(hessians)
        if not np.array_equal(result, hessians):
            raise OperatorError("second_gradient_tensor altered its input")
        try:
            self.evaluate(np.zeros((2, 3, 2, 3)))
        except OperatorShapeError:
            pass
        else:
            raise OperatorError("second_gradient_tensor accepted a non-square Hessian block")

    def is_valid(self) -> bool:
        try:
            self.verify()
        except (OperatorError, InputValidationError):
            return False
        return True
