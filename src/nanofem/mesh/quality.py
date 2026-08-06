"""Mesh quality checks: inverted/degenerate cells, aspect-ratio outliers (SDS 2.1)."""

from __future__ import annotations

from nanofem.mesh.mesh import Mesh


class MeshQualityChecker:
    """Emit warnings with element ids; strict mode escalates (SDS Section 13)."""

    def check(self, mesh: Mesh) -> None:
        """Scan the mesh and warn on quality outliers, naming offenders."""
        raise NotImplementedError("TODO(phase-2): quality scan")
