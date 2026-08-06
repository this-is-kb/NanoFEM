"""Parametric geometry -> meshed Mesh through the gmsh Python API.

gmsh is an optional extra ([meshing]); it is imported lazily inside methods
only (rule R3) so the core install never requires it.
"""

from __future__ import annotations

from nanofem.mesh.mesh import Mesh


class GmshGeometryBuilder:
    """Stepwise builder for parametric domain geometry (Builder pattern)."""

    def build(self) -> Mesh:
        """Mesh the accumulated geometry; physical groups become regions."""
        raise NotImplementedError("TODO(phase-2): gmsh-backed geometry builder")
