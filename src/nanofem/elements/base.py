"""Element contract: composition rule, not physics owner (SDS Section 3, clauses E-1..E-13).

Elements compose numerics (interpolation, quadrature, mapping) with an
injected Theory, Material, section geometry, and state views into
contribution providers. Batch evaluability (E-13) is contract: tabulated
data shared per (cell, interpolation, quadrature) triple, pure functions,
no per-element Python state in the hot path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.assembly.contributions import Contribution, OperatorRole


@dataclass(frozen=True)
class ElementDofSignature:
    """Per-node (field, component) attachments in SDS C-2 order (clause E-1)."""

    dof_names_per_node: tuple[tuple[str, ...], ...]


class Element(ABC):
    """Contract every element honors; full clause set lands with each phase."""

    @abstractmethod
    def dof_signature(self) -> ElementDofSignature:
        """Return the DOF signature the DofHandler consumes (E-1)."""

    @abstractmethod
    def contributions(self, role: OperatorRole) -> Iterator[Contribution]:
        """Yield blocks for one role per the emission discipline (E-6, SDS Section 10)."""


class StructuralElement(Element):
    """Closed-form structural family (ADR-002): local frame + transformation (E-10)."""

    @abstractmethod
    def transformation_matrix(self) -> NDArray[np.float64]:
        """Return orthonormal T with K_global = T^T K_local T (asserted in debug)."""
