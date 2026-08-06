"""Mesh import: thin wrapper over the io/ meshio adapter (rule R3)."""

from __future__ import annotations

from nanofem.mesh.mesh import Mesh


class MeshImporter:
    """Load a Mesh from disk via nanofem.io.meshio_adapter."""

    def load(self, path: str) -> Mesh:
        """Read a mesh file, preserving physical groups as regions."""
        raise NotImplementedError("TODO(phase-0.5): meshio-backed import")
