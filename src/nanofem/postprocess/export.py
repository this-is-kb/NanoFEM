"""File export: VTU via meshio, XDMF time series (SDS 2.19; rule R3 thin edge).

``VTKExporter`` writes a solved model's mesh plus recovered fields to a
``.vtu`` file for visualization in ParaView/VisIt - the last link in the
Stage-3 success criteria's own list ("recover stresses... export results").

``io.meshio_adapter.MeshIOAdapter`` sits *below* ``mesh`` in the layer
contract, so it cannot import ``nanofem.mesh.Mesh`` itself; this module sits
*above* both (R3: "io is a thin edge - only mesh and postprocess may import
it") and is the one that extracts plain geometry primitives (points, one
``(meshio_cell_type, connectivity)`` pair per block) from a ``Mesh`` before
handing them to the adapter. Point/cell field data is supplied as plain
dictionaries by the caller - this module does not know about
``postprocess.recovery``'s own dataclasses, keeping the two independently
usable and testable.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from nanofem.io.meshio_adapter import MeshIOAdapter
from nanofem.mesh.mesh import Mesh
from nanofem.utils.exceptions import InputValidationError

#: Mesh cell-type names this codebase produces (the Stage-3 minimal element library:
#: Bar/beams on "line2", T3 on "tri3", Q4 on "quad4") -> meshio's own linear-cell names.
_CELL_TYPE_TO_MESHIO: dict[str, str] = {"line2": "line", "tri3": "triangle", "quad4": "quad"}


def _split_by_block(mesh: Mesh, values: NDArray[np.float64]) -> list[NDArray[np.float64]]:
    """A flat per-cell array (global cell order) -> one array per cell block (meshio's model)."""
    if values.shape[0] != mesh.num_cells:
        raise InputValidationError(
            f"cell data must have one row per cell ({mesh.num_cells}), got {values.shape[0]}"
        )
    blocks = []
    offset = 0
    for block in mesh.blocks:
        blocks.append(values[offset : offset + block.num_cells])
        offset += block.num_cells
    return blocks


class VTKExporter:
    """Write a ``Mesh`` plus nodal/cell fields to a VTU file, via meshio."""

    def write(
        self,
        path: str,
        mesh: Mesh,
        point_data: Mapping[str, NDArray[np.float64]] | None = None,
        cell_data: Mapping[str, NDArray[np.float64]] | None = None,
    ) -> None:
        """Write ``mesh`` to ``path`` (``.vtu``), attaching any supplied point/cell data.

        ``point_data`` values are one row per node (``mesh.num_nodes``);
        ``cell_data`` values are one row per cell in global cell order
        (``mesh.num_cells``) and are split across meshio's per-block
        structure internally.
        """
        cells: list[tuple[str, NDArray[np.int64]]] = []
        for block in mesh.blocks:
            try:
                meshio_type = _CELL_TYPE_TO_MESHIO[block.cell_type]
            except KeyError:
                raise InputValidationError(
                    f"no meshio cell-type mapping for '{block.cell_type}'; supported: "
                    f"{sorted(_CELL_TYPE_TO_MESHIO)}"
                ) from None
            cells.append((meshio_type, block.connectivity))
        meshio_mesh = MeshIOAdapter().build_meshio_mesh(mesh.coordinates, cells)

        if point_data:
            for name, values in point_data.items():
                if values.shape[0] != mesh.num_nodes:
                    raise InputValidationError(
                        f"point data '{name}' must have one row per node "
                        f"({mesh.num_nodes}), got {values.shape[0]}"
                    )
            meshio_mesh.point_data = dict(point_data)
        if cell_data:
            meshio_mesh.cell_data = {
                name: _split_by_block(mesh, values) for name, values in cell_data.items()
            }
        meshio_mesh.write(path)


class TimeSeriesWriter:
    """Stream transient states to XDMF."""
