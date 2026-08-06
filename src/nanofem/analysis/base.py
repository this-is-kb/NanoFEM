"""Analysis: orchestration metadata only, physics-free (ADR-008, SDS 2.18).

Each analysis declares the operator roles its recipe needs; validate()
cross-checks the model's theories against them (pure metadata, zero math).
run() refuses until phase 2 - no solving happens in phase 1, by contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from nanofem.core.model import Model
from nanofem.numerics.assembly.contributions import OperatorRole
from nanofem.utils.exceptions import ModelError


class AnalysisBase(ABC):
    """Shared skeleton: model reference, role requirements, describable plan."""

    def __init__(self, model: Model) -> None:
        self._model = model

    @property
    def model(self) -> Model:
        """The (borrowed, never mutated) model."""
        return self._model

    @abstractmethod
    def required_roles(self) -> tuple[OperatorRole, ...]:
        """Operator roles this analysis recipe assembles (SDS Section 11)."""

    def validate(self) -> None:
        """Model completeness + role availability (fail fast, with names)."""
        self._model.validate()
        available: set[OperatorRole] = set()
        for theory in self._model.theories.values():
            available.update(theory.operator_roles())
        missing = [r.name for r in self.required_roles() if r not in available]
        if missing:
            raise ModelError(
                f"{type(self).__name__} requires roles {missing} which no registered "
                f"theory declares (available: {sorted(r.name for r in available)})"
            )

    def describe(self) -> dict[str, object]:
        """Orchestration metadata: the plan, not the solve."""
        return {
            "analysis": type(self).__name__,
            "required_roles": [r.name for r in self.required_roles()],
            "fields": [s.to_dict() for s in self._model.field_specs()],
            "model_fingerprint": self._model.fingerprint(),
        }

    def run(self) -> object:
        """Refuses in phase 1: solving begins with phase-2 numerics."""
        raise NotImplementedError(
            "phase 1 builds the object model only; assembly and solving arrive in phase 2"
        )
