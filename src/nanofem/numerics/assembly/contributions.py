"""The assembly currency: kinds, roles, the Contribution tuple, the provider protocol.

This module is the load-bearing seam of the whole framework (ADR-001/008,
SDS Sections 10-11). Kinds are defined by the topological dimension of the
integration domain, a definition complete in any d; in d = 2, EDGE coincides
with VERTEX and is normatively mapped to it. Roles are the vocabulary
analyses use to request global operators; COUPLING and SENSITIVITY acquire
their (field-pair / parameter) tags with their first consumers.

The assembler is the ONLY writer of global operators: nodal loads and point
attachments are VERTEX providers like everything else.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, unique
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@unique
class ContributionKind(Enum):
    """Integration-domain kinds by codimension (SDS Section 10)."""

    CELL = "cell"
    FACET = "facet"
    PAIR = "pair"
    EDGE = "edge"
    VERTEX = "vertex"


@unique
class OperatorRole(Enum):
    """Global operator vocabulary (SDS Section 11)."""

    STIFFNESS = "stiffness"
    MASS = "mass"
    DAMPING = "damping"
    GEOMETRIC_STIFFNESS = "geometric_stiffness"
    COUPLING = "coupling"
    FORCE = "force"
    SENSITIVITY = "sensitivity"


@dataclass(frozen=True, eq=False)
class Contribution:
    """One emitted block: (kind, role, rows, cols, dense block).

    Emission discipline (SDS Section 10): global int64 indices; float64 block
    shaped (len(rows), len(cols)); vector contributions set cols to None and
    ship a 1-D block; every touched (i, j) lies inside the precomputed
    sparsity pattern.
    """

    kind: ContributionKind
    role: OperatorRole
    row_dofs: NDArray[np.int64]
    col_dofs: NDArray[np.int64] | None
    block: NDArray[np.float64]


@runtime_checkable
class ContributionProvider(Protocol):
    """Anything that can yield contributions for a requested role (SDS 2.12)."""

    def contributions(self, role: OperatorRole) -> Iterator[Contribution]:
        """Yield this provider's blocks for one operator role."""
        ...
