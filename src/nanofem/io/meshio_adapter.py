"""Bidirectional meshio <-> nanofem.Mesh conversion preserving physical groups.

Never imported by the physics core (rule R3): mesh/ and postprocess/ are the
only internal consumers. ``io`` sits *below* ``mesh`` in the layer contract
(the reverse of what "mesh conversion" might suggest), so this module works
in plain geometry primitives (points, per-block cell arrays) rather than a
``nanofem.mesh.Mesh`` object - a caller that already holds a ``Mesh``
(``postprocess.export.VTKExporter``, which sits above both ``mesh`` and
``io`` and may import either) extracts those primitives itself.

Only the write direction (``build_meshio_mesh``) is implemented this phase,
for ``VTKExporter``'s VTU output - the "minimal path" this module's own TODO
has named since v0.0.1. Reading an external mesh file (physical-group
reconstruction, region recovery, and reassembly into a ``Mesh`` - which
would need to live in ``mesh/``, the one layer above this that is allowed to
import both ``io`` and construct a ``Mesh``) is a separate, larger
undertaking than "write VTK output" needs and remains deferred, not
half-built.
"""

from __future__ import annotations

from collections.abc import Sequence

import meshio
import numpy as np
from numpy.typing import NDArray

from nanofem.utils.exceptions import InputValidationError


class MeshIOAdapter:
    """Import/export through the meshio exchange hub."""

    def read(self, path: str) -> object:
        """Return mesh data read from a meshio-supported file."""
        raise NotImplementedError("TODO(future): meshio -> Mesh import path")

    def build_meshio_mesh(
        self,
        points: NDArray[np.float64],
        cells: Sequence[tuple[str, NDArray[np.int64]]],
    ) -> meshio.Mesh:
        """Build a ``meshio.Mesh`` from plain geometry primitives (write path only).

        ``points`` is padded to 3-D if given in 1-D/2-D (VTK's own convention
        - a coordinate array with fewer than 3 columns is not itself a valid
        VTU point array). ``cells`` is ``(meshio_cell_type, connectivity)``
        pairs, one per homogeneous block, in meshio's own cell-type naming.
        """
        coords = np.asarray(points, dtype=np.float64)
        if coords.ndim != 2 or not 1 <= coords.shape[1] <= 3:
            raise InputValidationError(
                f"points must be (n_points, d) with d in 1..3, got {coords.shape}"
            )
        if coords.shape[1] < 3:
            coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
        return meshio.Mesh(points=coords, cells=list(cells))
