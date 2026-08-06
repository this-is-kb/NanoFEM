"""``MeshIOAdapter.to_meshio``/``VTKExporter``: VTU export via meshio (SDS 2.19).

Verified by a real write-then-read round trip through meshio itself (not
just constructing a ``meshio.Mesh`` object and trusting it) - geometry,
point data, and cell data must all survive a genuine ``.vtu`` file.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.io.meshio_adapter import MeshIOAdapter
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.postprocess.export import VTKExporter
from nanofem.utils.exceptions import InputValidationError

meshio = pytest.importorskip("meshio")


def _t3_mesh() -> Mesh:
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    block = CellBlock("tri3", np.array([[0, 1, 2], [0, 2, 3]]), region="plate")
    return Mesh(coords, (block,))


def test_build_meshio_mesh_preserves_geometry_and_pads_to_3d() -> None:
    mesh = _t3_mesh()
    converted = MeshIOAdapter().build_meshio_mesh(
        mesh.coordinates, [("triangle", mesh.blocks[0].connectivity)]
    )
    assert converted.points.shape == (4, 3)
    np.testing.assert_allclose(converted.points[:, :2], mesh.coordinates)
    np.testing.assert_allclose(converted.points[:, 2], 0.0)
    assert len(converted.cells) == 1
    assert converted.cells[0].type == "triangle"
    np.testing.assert_array_equal(converted.cells[0].data, mesh.blocks[0].connectivity)


def test_build_meshio_mesh_rejects_bad_point_shape() -> None:
    with pytest.raises(InputValidationError, match="points must be"):
        MeshIOAdapter().build_meshio_mesh(np.zeros((3, 4)), [])


def test_vtk_exporter_rejects_an_unmapped_cell_type() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    block = CellBlock("tet4", np.array([[0, 1]]), region="r")
    mesh = Mesh(coords, (block,))
    with pytest.raises(InputValidationError, match="no meshio cell-type mapping"):
        VTKExporter().write("unused.vtu", mesh)


def test_vtk_exporter_round_trips_point_and_cell_data(tmp_path) -> None:  # type: ignore[no-untyped-def]
    mesh = _t3_mesh()
    displacement = np.array([0.0, 0.1, 0.2, 0.05])
    element_stress = np.array([1.0e6, 2.0e6])
    path = str(tmp_path / "plate.vtu")

    VTKExporter().write(
        path,
        mesh,
        point_data={"displacement": displacement},
        cell_data={"stress_xx": element_stress},
    )

    reread = meshio.read(path)
    np.testing.assert_allclose(reread.point_data["displacement"], displacement)
    np.testing.assert_allclose(reread.cell_data["stress_xx"][0], element_stress)


def test_vtk_exporter_rejects_wrong_shaped_point_data(tmp_path) -> None:  # type: ignore[no-untyped-def]
    mesh = _t3_mesh()
    with pytest.raises(InputValidationError, match="point data"):
        VTKExporter().write(str(tmp_path / "bad.vtu"), mesh, point_data={"u": np.array([1.0, 2.0])})


def test_vtk_exporter_rejects_wrong_shaped_cell_data(tmp_path) -> None:  # type: ignore[no-untyped-def]
    mesh = _t3_mesh()
    with pytest.raises(InputValidationError, match="cell data"):
        VTKExporter().write(
            str(tmp_path / "bad.vtu"), mesh, cell_data={"s": np.array([1.0, 2.0, 3.0])}
        )
