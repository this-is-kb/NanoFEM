"""Module-level verification suite for the operator library (SDS Section 8).

Loops the registry, running each operator's own ``verify()``, plus two
cross-cutting checks that no single operator class can make on its own: that
the catalog's derivative-order table and ``derived_continuity`` still agree
with what each operator class declares, and that independently-written
recipes (divergence, symmetric gradient) agree on the same field.
"""

from __future__ import annotations

import numpy as np

from nanofem.numerics.operators.base import (
    OPERATOR_DERIVATIVE_ORDER,
    Continuity,
    derived_continuity,
)
from nanofem.numerics.operators.divergence import divergence_matrix
from nanofem.numerics.operators.errors import OperatorError
from nanofem.numerics.operators.registry import OPERATOR_REGISTRY
from nanofem.numerics.operators.symmetric_gradient import symmetric_gradient_matrix
from nanofem.numerics.tensors.voigt import voigt_to_strain


def verify_registry_self_consistency() -> None:
    """Every registered operator's declared name and derivative order match the catalog."""
    for key, operator_class in OPERATOR_REGISTRY.items():
        instance = operator_class()
        if instance.name() != key:
            raise OperatorError(
                f"registry key {key!r} does not match {operator_class.__name__}.name() "
                f"= {instance.name()!r}"
            )
        expected = OPERATOR_DERIVATIVE_ORDER[key]
        if instance.required_derivative_order() != expected:
            raise OperatorError(
                f"{operator_class.__name__}.required_derivative_order() = "
                f"{instance.required_derivative_order()}, expected {expected}"
            )


def verify_continuity_derivation() -> None:
    """``second_gradient`` still forces ``Continuity.C1`` through the shared derivation rule."""
    if derived_continuity(("second_gradient",)) is not Continuity.C1:
        raise OperatorError("derived_continuity(('second_gradient',)) is no longer C1")
    if derived_continuity(("gradient", "symmetric_gradient")) is not Continuity.C0:
        raise OperatorError("derived_continuity of two first-derivative operators is no longer C0")


def verify_cross_operator_consistency() -> None:
    """Divergence and the trace of the symmetric gradient agree on the same field."""
    rng = np.random.default_rng(2026)
    gradients = rng.normal(size=(4, 3, 2))  # n_qp=4, n_fun=3, dim=2
    dofs = rng.normal(size=(3, 2))

    divergence = np.einsum("qoai,ai->qo", divergence_matrix(gradients), dofs)[:, 0]

    strain_voigt = np.einsum("qvai,ai->qv", symmetric_gradient_matrix(gradients), dofs)
    strain_full = voigt_to_strain(strain_voigt)
    trace_of_strain = np.trace(strain_full, axis1=-2, axis2=-1)

    if not np.allclose(divergence, trace_of_strain, atol=1e-10):
        raise OperatorError("divergence disagreed with trace(symmetric_gradient) on the same field")


def verify_operator_library() -> None:
    """Run every registered operator's own ``verify()`` plus the cross-cutting checks."""
    verify_registry_self_consistency()
    verify_continuity_derivation()
    verify_cross_operator_consistency()
    for operator_class in OPERATOR_REGISTRY.values():
        operator_class().verify()


def is_operator_library_valid() -> bool:
    """Return ``True`` if :func:`verify_operator_library` passes, ``False`` otherwise."""
    try:
        verify_operator_library()
    except OperatorError:
        return False
    return True
