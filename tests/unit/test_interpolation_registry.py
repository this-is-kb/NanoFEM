"""Registry, serialization, value semantics, and frozen-convention regressions.

The regression tests pin node counts, numbering, and DOF ordering so a future
refactor cannot silently renumber the elements every formulation will be
built on.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from nanofem.numerics.interpolation import (
    AVAILABLE_INTERPOLATIONS,
    HermiteInterpolation,
    HierarchicalInterpolation,
    Interpolation,
    InterpolationFamily,
    LagrangeInterpolation,
    SerendipityInterpolation,
    SpectralInterpolation,
    available_interpolations,
    interpolation,
    interpolation_from_dict,
)
from nanofem.numerics.interpolation.future import _FutureInterpolation
from nanofem.numerics.reference.cell import REFERENCE_CELLS
from nanofem.numerics.reference.enums import CellType
from nanofem.utils.exceptions import InputValidationError

ALL: list[Interpolation] = [interpolation(f, c, k) for f, c, k in available_interpolations()]


# ---- registry ---------------------------------------------------------------


def test_factory_accepts_enums_and_strings() -> None:
    """The factory resolves both enum and string arguments."""
    by_enum = interpolation(InterpolationFamily.LAGRANGE, CellType.TRIANGLE, 2)
    by_string = interpolation("lagrange", "triangle", 2)
    assert by_enum == by_string
    assert isinstance(by_string, LagrangeInterpolation)
    assert isinstance(interpolation("hermite", "line", 3), HermiteInterpolation)


def test_registry_lists_exactly_what_is_implemented() -> None:
    """Eleven combinations: Lagrange on three cells at three orders, Hermite on two."""
    assert len(AVAILABLE_INTERPOLATIONS) == 11
    assert available_interpolations() == AVAILABLE_INTERPOLATIONS
    families = {f for f, _, _ in AVAILABLE_INTERPOLATIONS}
    assert families == {InterpolationFamily.LAGRANGE, InterpolationFamily.HERMITE}
    for family, cell, order in AVAILABLE_INTERPOLATIONS:
        interpolation(family, cell, order).validate()


def test_factory_rejects_unknown_family_cell_and_order() -> None:
    """Unknown names and unsupported combinations raise with context."""
    with pytest.raises(InputValidationError, match="family"):
        interpolation("bezier", "line", 1)
    with pytest.raises(InputValidationError, match="cell type"):
        interpolation("lagrange", "banana", 1)
    with pytest.raises(InputValidationError, match="order 7"):
        interpolation("lagrange", "line", 7)
    with pytest.raises(InputValidationError, match="not implemented on"):
        interpolation("lagrange", "hexahedron", 1)
    with pytest.raises(InputValidationError, match="order 2"):
        interpolation("hermite", "line", 2)


def test_hermite_triangle_refuses_with_the_reason() -> None:
    """The cubic Hermite triangle is C0, so it is refused rather than mislabelled C1."""
    with pytest.raises(NotImplementedError) as excinfo:
        HermiteInterpolation(CellType.TRIANGLE, 3)
    message = str(excinfo.value)
    assert "C0" in message and "Argyris" in message


@pytest.mark.parametrize(
    "placeholder", [SerendipityInterpolation, HierarchicalInterpolation, SpectralInterpolation]
)
def test_placeholders_refuse_and_document_what_they_need(
    placeholder: type[_FutureInterpolation],
) -> None:
    """Placeholders raise on construction, name their blocker, and carry metadata."""
    with pytest.raises(NotImplementedError, match="declared placeholder"):
        placeholder("quadrilateral", 2)
    assert placeholder.BLOCKED_BY
    assert placeholder.PROVISIONAL_METADATA["family"]


def test_spectral_is_blocked_by_quadrature_not_by_scheduling() -> None:
    """The spectral family cannot precede quadrature: GLL nodes are quadrature points."""
    assert "quadrature" in SpectralInterpolation.BLOCKED_BY
    assert SpectralInterpolation.PROVISIONAL_METADATA["node_set"] == "gauss_lobatto_legendre"


def test_hierarchical_is_the_family_without_interpolation_nodes() -> None:
    """Hierarchical DOFs are moments, which is why evaluation_points is a separate query."""
    assert not InterpolationFamily.HIERARCHICAL.is_nodal_family
    assert InterpolationFamily.LAGRANGE.is_nodal_family
    assert HierarchicalInterpolation.PROVISIONAL_METADATA["has_interpolation_nodes"] is False


def test_family_implemented_flags() -> None:
    """The enum reports which families have concrete implementations."""
    assert InterpolationFamily.LAGRANGE.is_implemented
    assert InterpolationFamily.HERMITE.is_implemented
    assert not InterpolationFamily.SPECTRAL.is_implemented


# ---- the mesh cell-name bridge ---------------------------------------------


def test_mesh_cell_names_that_the_phase_zero_registry_already_knows() -> None:
    """The classical low-order elements map onto registered mesh cell types."""
    for family, cell, order, expected in [
        (InterpolationFamily.LAGRANGE, CellType.LINE, 1, "line2"),
        (InterpolationFamily.LAGRANGE, CellType.LINE, 2, "line3"),
        (InterpolationFamily.LAGRANGE, CellType.TRIANGLE, 1, "tri3"),
        (InterpolationFamily.LAGRANGE, CellType.TRIANGLE, 2, "tri6"),
        (InterpolationFamily.LAGRANGE, CellType.QUADRILATERAL, 1, "quad4"),
        (InterpolationFamily.HERMITE, CellType.LINE, 3, "line2"),
        (InterpolationFamily.HERMITE, CellType.QUADRILATERAL, 3, "quad4"),
    ]:
        element = interpolation(family, cell, order)
        assert element.mesh_cell_name == expected
        assert expected in REFERENCE_CELLS


def test_every_mesh_cell_name_now_resolves() -> None:
    """The higher-order Lagrange cells are registered (dev note N-21, closed in phase 4).

    Phase 3 found that the registry held ``quad8`` - the *serendipity* cell -
    but not ``quad9``, which Lagrange order 2 on a quadrilateral needs, nor the
    cubic cells. The rule was that entries land with the family that consumes
    them; the shape function library is that family, so they were added there.
    ``quad8`` still has no family and still waits for serendipity.
    """
    for family, cell, order, name in [
        (InterpolationFamily.LAGRANGE, CellType.LINE, 3, "line4"),
        (InterpolationFamily.LAGRANGE, CellType.TRIANGLE, 3, "tri10"),
        (InterpolationFamily.LAGRANGE, CellType.QUADRILATERAL, 2, "quad9"),
        (InterpolationFamily.LAGRANGE, CellType.QUADRILATERAL, 3, "quad16"),
    ]:
        element = interpolation(family, cell, order)
        assert element.mesh_cell_name == name
        assert name in REFERENCE_CELLS
        assert REFERENCE_CELLS[name].num_nodes == element.num_nodes
    assert "quad8" in REFERENCE_CELLS  # the serendipity cell, still awaiting its family


# ---- serialization ----------------------------------------------------------


@pytest.mark.parametrize("element", ALL, ids=lambda e: f"{e.family.value}-{e.cell_type.value}")
def test_to_dict_is_complete_and_json_safe(element: Interpolation) -> None:
    """The payload carries the full metadata record and survives json.dumps."""
    data = element.to_dict()
    assert data["schema"] == "nanofem-interpolation/1"
    assert data["family"] == element.family.value
    assert data["num_dofs"] == element.num_dofs
    assert len(data["nodes"]) == element.num_nodes
    assert len(data["dofs"]) == element.num_dofs
    assert len(data["shape_function_ids"]) == element.num_dofs
    assert len(data["basis_ids"]) == element.space_dimension
    assert data["polynomial_space"]["name"] == element.polynomial_space.name
    json.dumps(data)


@pytest.mark.parametrize("element", ALL, ids=lambda e: f"{e.family.value}-{e.cell_type.value}")
def test_dict_round_trip_reconstructs_an_equal_element(element: Interpolation) -> None:
    """from_dict(to_dict(x)) == x for every implemented element."""
    restored = interpolation_from_dict(element.to_dict())
    assert restored == element
    assert restored.to_dict() == element.to_dict()


@pytest.mark.parametrize("element", ALL, ids=lambda e: f"{e.family.value}-{e.cell_type.value}")
def test_to_json_is_deterministic(element: Interpolation) -> None:
    """JSON output is stable across calls (keys sorted)."""
    assert element.to_json() == element.to_json()
    assert json.loads(element.to_json())["order"] == element.order


def test_from_dict_rejects_missing_keys_and_tampered_counts() -> None:
    """Incomplete payloads and inconsistent DOF counts raise."""
    with pytest.raises(InputValidationError, match="missing"):
        interpolation_from_dict({"family": "lagrange", "cell_type": "line"})
    tampered = LagrangeInterpolation(CellType.TRIANGLE, 2).to_dict()
    tampered["num_dofs"] = 99
    with pytest.raises(InputValidationError, match="declares 99 dofs"):
        interpolation_from_dict(tampered)


# ---- immutability and utilities --------------------------------------------


@pytest.mark.parametrize("element", ALL, ids=lambda e: f"{e.family.value}-{e.cell_type.value}")
def test_node_locations_are_read_only(element: Interpolation) -> None:
    """The node coordinate array cannot be mutated through the property."""
    with pytest.raises(ValueError):
        element.node_locations[0, 0] = 99.0
    with pytest.raises(ValueError):
        element.evaluation_points()[0, 0] = 99.0


def test_elements_are_frozen() -> None:
    """Interpolations are frozen dataclasses: no attribute assignment."""
    element = LagrangeInterpolation(CellType.LINE, 2)
    with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError is an AttributeError
        element.order_ = 3  # type: ignore[misc]


@pytest.mark.parametrize("element", ALL, ids=lambda e: f"{e.family.value}-{e.cell_type.value}")
def test_pretty_and_debug_summaries(element: Interpolation) -> None:
    """Summaries report the key facts; the debug dump adds nodes, DOFs, and monomials."""
    pretty = element.pretty()
    assert element.family.value in pretty and element.polynomial_space.name in pretty
    debug = element.debug_summary()
    assert "nodes:" in debug and "dofs (functionals):" in debug
    assert "monomial basis" in debug and "unisolvence cond" in debug
    assert debug.count("\n") > pretty.count("\n")


@pytest.mark.parametrize("element", ALL, ids=lambda e: f"{e.family.value}-{e.cell_type.value}")
def test_repr_is_informative(element: Interpolation) -> None:
    """repr names the class, cell, order, DOF count, and space."""
    text = repr(element)
    assert type(element).__name__ in text
    assert element.cell_type.value in text
    assert f"dofs={element.num_dofs}" in text


# ---- equality and hashing ---------------------------------------------------


def test_equality_is_by_family_cell_and_order() -> None:
    """Same triple compares equal; any difference does not."""
    assert LagrangeInterpolation(CellType.TRIANGLE, 2) == LagrangeInterpolation(
        CellType.TRIANGLE, 2
    )
    assert LagrangeInterpolation(CellType.TRIANGLE, 2) != LagrangeInterpolation(
        CellType.TRIANGLE, 1
    )
    assert LagrangeInterpolation(CellType.TRIANGLE, 2) != LagrangeInterpolation(
        CellType.QUADRILATERAL, 2
    )
    assert LagrangeInterpolation(CellType.LINE, 3) != HermiteInterpolation(CellType.LINE, 3)
    assert LagrangeInterpolation(CellType.LINE, 1) != object()


def test_hashing_enables_set_and_dict_use() -> None:
    """Equal elements share a hash: usable as tabulation cache keys."""
    a = LagrangeInterpolation(CellType.TRIANGLE, 2)
    b = LagrangeInterpolation(CellType.TRIANGLE, 2)
    assert hash(a) == hash(b)
    assert len({a, b, LagrangeInterpolation(CellType.LINE, 1)}) == 2
    cache = {element: element.num_dofs for element in ALL}
    assert cache[LagrangeInterpolation(CellType.TRIANGLE, 2)] == 6
    assert cache[HermiteInterpolation(CellType.LINE, 3)] == 4


# ---- regression: the frozen conventions -------------------------------------


def test_regression_classical_node_counts() -> None:
    """Node counts pin the classical elements: line2/3/4, tri3/6/10, quad4/9/16."""
    counts = {
        (cell.value, order): LagrangeInterpolation(cell, order).num_nodes
        for cell in (CellType.LINE, CellType.TRIANGLE, CellType.QUADRILATERAL)
        for order in (1, 2, 3)
    }
    assert counts == {
        ("line", 1): 2,
        ("line", 2): 3,
        ("line", 3): 4,
        ("triangle", 1): 3,
        ("triangle", 2): 6,
        ("triangle", 3): 10,
        ("quadrilateral", 1): 4,
        ("quadrilateral", 2): 9,
        ("quadrilateral", 3): 16,
    }
    assert HermiteInterpolation(CellType.LINE, 3).num_dofs == 4
    assert HermiteInterpolation(CellType.QUADRILATERAL, 3).num_dofs == 16


def test_regression_tri6_node_coordinates() -> None:
    """tri6 lattice: vertices then edge midpoints in reference edge order."""
    assert np.allclose(
        LagrangeInterpolation(CellType.TRIANGLE, 2).node_locations,
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]],
    )


def test_regression_quad9_node_coordinates() -> None:
    """quad9 lattice: vertices, bottom/right/top/left midpoints, centre."""
    assert np.allclose(
        LagrangeInterpolation(CellType.QUADRILATERAL, 2).node_locations,
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, 0.0],
        ],
    )


def test_regression_line3_node_coordinates() -> None:
    """line3: the two vertices, then the midpoint."""
    assert np.allclose(
        LagrangeInterpolation(CellType.LINE, 2).node_locations, [[-1.0], [1.0], [0.0]]
    )


def test_regression_hermite_line_payload() -> None:
    """A payload snapshot pins the cubic Hermite line against silent drift."""
    data = HermiteInterpolation(CellType.LINE, 3).to_dict()
    assert data["family"] == "hermite"
    assert data["continuity"] == "C1"
    assert data["is_nodal"] is False
    assert data["num_nodes"] == 2 and data["num_dofs"] == 4
    assert data["dofs_per_node"] == [2, 2]
    assert data["shape_function_ids"] == ["N_v0", "N_v0_d1", "N_v1", "N_v1_d1"]
    assert data["basis_ids"] == ["1", "xi", "xi^2", "xi^3"]
    assert data["mesh_cell_name"] == "line2"


def test_regression_completeness_degrees() -> None:
    """Completeness degree equals the order for every element (Q_k contains P_k)."""
    for element in ALL:
        assert element.completeness_degree == element.order
    quad3 = LagrangeInterpolation(CellType.QUADRILATERAL, 3)
    assert quad3.max_total_degree == 6  # xi^3*eta^3, well above the completeness degree


# ---- independence -----------------------------------------------------------


def test_module_needs_only_the_reference_layer() -> None:
    """The interpolation framework builds on phase 2 and nothing above it."""
    import subprocess
    import sys

    script = (
        "from nanofem.numerics.interpolation import LagrangeInterpolation;"
        "from nanofem.numerics.reference.enums import CellType;"
        "e = LagrangeInterpolation(CellType.TRIANGLE, 2);"
        "assert e.num_nodes == 6 and e.completeness_degree == 2;"
        "e.validate();"
        "assert e.to_json();"
        "print('standalone OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "standalone OK" in result.stdout
