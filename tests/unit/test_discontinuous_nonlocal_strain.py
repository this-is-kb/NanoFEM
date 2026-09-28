"""Discontinuous (cell-owned) nonlocal strain with interior-penalty facet coupling.

With a continuous equal-order e*, the discrete inf-sup constant of the (u, e*) pair decays as
O(h) and the mu = 0 solution only converges to the classical one. A cell-owned e* contains
eps(u_h) on every cell, so at mu = 0 the reduced operator must equal the classical stiffness
exactly, on any mesh; for mu > 0 the Helmholtz regularity is imposed by SIPG facet terms. These
tests pin down the DOF-ownership contract, the facet skeleton, the SIPG operator's
definiteness, the exact local limit, and that the continuous default is untouched.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.handler import ConstraintHandler
from nanofem.core.dof_handler import DofHandler
from nanofem.core.model import DomainDefinition, Model
from nanofem.elements.continuum.interior_penalty import NonlocalInteriorPenaltyFacet
from nanofem.elements.continuum.nonlocal_continuum import NonlocalContinuumElement
from nanofem.elements.factory import build_elements
from nanofem.geometry.plane import PlaneGeometry
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.numerics.assembly.assembler import Assembler
from nanofem.numerics.assembly.contributions import ContributionKind, OperatorRole
from nanofem.numerics.assembly.sparsity import SparsityPattern
from nanofem.numerics.operators.base import Continuity
from nanofem.physics.elasticity.eringen_differential import (
    EringenDifferentialMaterial,
    EringenDifferentialTheory,
)
from nanofem.physics.elasticity.isotropic import IsotropicElasticity
from nanofem.physics.elasticity.plane import PlaneStressConstitutive
from nanofem.utils.exceptions import DofMappingError, PhysicsError

YOUNG_MODULUS = 200e9
POISSON = 0.3
THICKNESS = 0.01
N_X, N_Y = 4, 2


def _mesh(cell: str, n_y: int = N_Y) -> Mesh:
    """A 2 x 1 plate on a 2n_y x n_y grid of quads, or of triangles split along one diagonal."""
    n_x = 2 * n_y
    xs, ys = np.linspace(0.0, 2.0, n_x + 1), np.linspace(0.0, 1.0, n_y + 1)
    coords = np.array([[x, y] for y in ys for x in xs])

    def nid(i: int, j: int) -> int:
        return j * (n_x + 1) + i

    cells: list[list[int]] = []
    for j in range(n_y):
        for i in range(n_x):
            a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            cells += [[a, b, c, d]] if cell == "quad4" else [[a, b, c], [a, c, d]]
    regions = (
        Region("left_edge", 0, tuple(nid(0, j) for j in range(n_y + 1))),
        Region("right_edge", 0, tuple(nid(n_x, j) for j in range(n_y + 1))),
        Region("origin", 0, (nid(0, 0),)),
        Region("load_point", 0, (nid(n_x // 2, n_y),)),
    )
    return Mesh(coords, (CellBlock(cell, np.array(cells), region="plate"),), regions)


def _model(
    cell: str,
    e0a: float | None,
    continuity: Continuity = Continuity.DISCONTINUOUS,
    n_y: int = N_Y,
) -> Model:
    """Mixed Eringen model (``e0a`` given) or classical plane stress (``e0a`` None)."""
    model = Model(_mesh(cell, n_y))
    model.add_section("plane", PlaneGeometry(thickness=THICKNESS))
    if e0a is None:
        model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON))
        model.add_theory("theory", IsotropicElasticity(dim=2))
        model.add_constitutive("law", PlaneStressConstitutive())
    else:
        model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON, e0a=e0a))
        model.add_theory(
            "theory", EringenDifferentialTheory(dim=2, nonlocal_strain_continuity=continuity)
        )
        model.add_constitutive("law", EringenDifferentialMaterial(PlaneStressConstitutive()))
    model.add_domain(
        DomainDefinition("dom", "plate", "theory", "steel", geometry="plane", constitutive="law")
    )
    model.add_dirichlet(DirichletBC("left_edge", "u", ("x",), 0.0))
    model.add_dirichlet(DirichletBC("origin", "u", ("y",), 0.0))
    return model


def _assembled(model: Model) -> tuple[sp.csr_matrix, np.ndarray, DofHandler, tuple[object, ...]]:
    dof_handler = model.build_dof_handler()
    elements = build_elements(model, dof_handler)
    pattern = SparsityPattern.from_providers(elements, OperatorRole.STIFFNESS, dof_handler.num_dofs)
    stiffness = cast(sp.csr_matrix, Assembler(pattern).assemble(elements, OperatorRole.STIFFNESS))
    bcs = cast(tuple[DirichletBC, ...], model.dirichlet_bcs)
    free = np.asarray(ConstraintHandler(model.mesh, dof_handler, bcs).partition().free_dofs)
    return sp.csr_matrix(stiffness), free, dof_handler, tuple(elements)


def _blocks(model: Model) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, str]]]:
    """``(K_ue, K_ee, K_classical, u keys)`` on free DOFs, K_classical ordered like K_ue's rows."""
    stiffness, free, dof_handler, _ = _assembled(model)
    u_pos, e_pos, keys = [], [], []
    for pos, g in enumerate(free):
        record = dof_handler.dof(int(g))
        if record.field == "u":
            u_pos.append(pos)
            keys.append((record.node_id, record.component))
        else:
            e_pos.append(pos)
    reduced = stiffness[free][:, free].toarray()
    classical = _model(_cell_of(model), None)
    k_cl, free_cl, dh_cl, _ = _assembled(classical)
    index = {int(g): k for k, g in enumerate(free_cl)}
    order = [index[dh_cl.global_dof(n, "u", c)] for n, c in keys]
    k_cl_free = k_cl[free_cl][:, free_cl].toarray()[np.ix_(order, order)]
    return reduced[np.ix_(u_pos, e_pos)], -reduced[np.ix_(e_pos, e_pos)], k_cl_free, keys


def _cell_of(model: Model) -> str:
    return model.mesh.blocks[0].cell_type


# ---------------------------------------------------------------------------- DOF ownership


def test_continuous_default_keeps_nodal_numbering_and_emits_no_facets() -> None:
    model = _model("tri3", 0.1, Continuity.C0)
    _, _, dof_handler, elements = _assembled(model)
    n_nodes = model.mesh.num_nodes
    assert dof_handler.num_dofs == 5 * n_nodes
    assert all(dof_handler.dof(g).cell_id is None for g in range(dof_handler.num_dofs))
    # node-major, then field (u, e_star), then component: unchanged by the new capability
    assert dof_handler.global_dof(3, "u", "x") == 5 * 3
    assert dof_handler.global_dof(3, "e_star", "xy") == 5 * 3 + 4
    assert not any(isinstance(e, NonlocalInteriorPenaltyFacet) for e in elements)
    assert EringenDifferentialTheory(dim=2).contribution_kinds() == (ContributionKind.CELL,)


@pytest.mark.parametrize("cell", ["tri3", "quad4"])
def test_discontinuous_field_gets_cell_owned_dofs_after_nodal_ones(cell: str) -> None:
    model = _model(cell, 0.1)
    dof_handler = model.build_dof_handler()
    n_nodes, mesh = model.mesh.num_nodes, model.mesh
    n_cell_dofs = sum(3 * len(mesh.cell(c).connectivity) for c in range(mesh.num_cells))
    assert dof_handler.num_dofs == 2 * n_nodes + n_cell_dofs
    # u keeps its nodal numbering; every e* DOF comes after it, owned by a cell
    assert dof_handler.global_dof(0, "u", "y") == 1
    cell0 = mesh.cell(0).connectivity
    assert dof_handler.cell_dof(0, cell0[0], "e_star", "xx") == 2 * n_nodes
    assert len(dof_handler.dofs_of_cell(0)) == 3 * len(cell0)
    shared = cell0[0]
    owners = [c for c in range(mesh.num_cells) if shared in mesh.cell(c).connectivity]
    copies = {dof_handler.cell_dof(c, shared, "e_star", "xx") for c in owners}
    assert len(copies) == len(owners)  # one independent copy per adjacent cell
    with pytest.raises(DofMappingError, match="discontinuous"):
        dof_handler.global_dof(shared, "e_star", "xx")


def test_cell_owned_numbering_round_trips_through_export() -> None:
    dof_handler = _model("quad4", 0.1).build_dof_handler()
    restored = DofHandler.import_numbering(dof_handler.export_numbering())
    assert restored.fingerprint() == dof_handler.fingerprint()
    last = restored.dof(restored.num_dofs - 1)
    assert last.cell_id is not None
    assert restored.cell_dof(last.cell_id, last.node_id, last.field, last.component) == (
        restored.num_dofs - 1
    )


def test_theory_rejects_a_c1_nonlocal_strain() -> None:
    with pytest.raises(PhysicsError, match="nonlocal_strain_continuity"):
        EringenDifferentialTheory(dim=2, nonlocal_strain_continuity=Continuity.C1)


# ---------------------------------------------------------------------------- facet skeleton


@pytest.mark.parametrize(
    ("cell", "interior_edges"),
    [
        # quads: vertical interior edges + horizontal interior edges
        ("quad4", (N_X - 1) * N_Y + N_X * (N_Y - 1)),
        # triangles: the quad count plus one diagonal per grid cell
        ("tri3", (N_X - 1) * N_Y + N_X * (N_Y - 1) + N_X * N_Y),
    ],
)
def test_one_facet_element_per_interior_edge(cell: str, interior_edges: int) -> None:
    _, _, _, elements = _assembled(_model(cell, 0.1))
    facets = [e for e in elements if isinstance(e, NonlocalInteriorPenaltyFacet)]
    assert len(facets) == interior_edges
    assert sum(isinstance(e, NonlocalContinuumElement) for e in elements) == len(
        _model(cell, 0.1).mesh.blocks[0].connectivity
    )
    block = facets[0].local_stiffness()
    np.testing.assert_allclose(block, block.T, atol=1e-12 * np.abs(block).max())


@pytest.mark.parametrize("cell", ["tri3", "quad4"])
def test_sipg_operator_is_semidefinite_with_only_the_constant_null_space(cell: str) -> None:
    """K_ee(mu=1) - K_ee(mu=0) is the assembled SIPG operator (times thickness)."""
    _, k_ee_1, _, _ = _blocks(_model(cell, 1.0))
    _, k_ee_0, _, _ = _blocks(_model(cell, 0.0))
    a = k_ee_1 - k_ee_0
    eigenvalues = np.linalg.eigvalsh(0.5 * (a + a.T)) / np.abs(a).max()
    assert eigenvalues[0] > -1e-10
    assert int(np.sum(np.abs(eigenvalues) < 1e-10)) == 3  # one constant per Voigt component


# ---------------------------------------------------------------------------- local limit


@pytest.mark.parametrize("cell", ["tri3", "quad4"])
def test_local_limit_is_mesh_exact(cell: str) -> None:
    """At mu = 0 the reduced operator K_ue K_ee^-1 K_eu equals the classical stiffness."""
    k_ue, k_ee, k_classical, _ = _blocks(_model(cell, 0.0))
    reduced = k_ue @ np.linalg.solve(k_ee, k_ue.T)
    np.testing.assert_allclose(reduced, k_classical, rtol=0, atol=1e-10 * np.abs(k_classical).max())


@pytest.mark.parametrize("cell", ["tri3", "quad4"])
def test_continuous_e_star_is_not_mesh_exact(cell: str) -> None:
    """The contrast that motivates the capability: the C0 reduced operator is strictly softer."""
    k_ue, k_ee, k_classical, _ = _blocks(_model(cell, 0.0, Continuity.C0))
    reduced = k_ue @ np.linalg.solve(k_ee, k_ue.T)
    gap = np.abs(reduced - k_classical).max() / np.abs(k_classical).max()
    assert gap > 1e-3


@pytest.mark.parametrize("cell", ["tri3", "quad4"])
def test_point_load_solution_equals_classical_at_mu_zero(cell: str) -> None:
    """A concentrated force excites exactly the modes the continuous e* mishandles."""
    k_ue, k_ee, k_classical, keys = _blocks(_model(cell, 0.0))
    load_node = _mesh(cell).nodes_in_region("load_point")[0]
    f = np.array([-1.0e4 if key == (load_node, "y") else 0.0 for key in keys])
    n_u, n_e = k_ue.shape
    block = np.block([[np.zeros((n_u, n_u)), k_ue], [k_ue.T, -k_ee]])
    u_mixed = np.linalg.solve(block, np.concatenate([f, np.zeros(n_e)]))[:n_u]
    u_classical = np.linalg.solve(k_classical, f)
    np.testing.assert_allclose(u_mixed, u_classical, rtol=0, atol=1e-9 * np.abs(u_classical).max())


def _bending_energy(continuity: Continuity, n_y: int) -> float:
    """Strain energy f.u/2 under pure bending (linear end traction) at e0a = 0.1."""
    model = _model("quad4", 0.1, continuity, n_y)
    stiffness, free, dof_handler, _ = _assembled(model)
    right = model.mesh.nodes_in_region("right_edge")  # bottom to top
    dy = 1.0 / n_y
    traction = [1.0e4 * (2.0 * j * dy - 1.0) for j in range(n_y + 1)]
    f = np.zeros(dof_handler.num_dofs)
    for j in range(n_y):  # consistent nodal forces of a linear traction, segment by segment
        t0, t1 = traction[j], traction[j + 1]
        f[dof_handler.global_dof(right[j], "u", "x")] += dy / 6 * (2 * t0 + t1)
        f[dof_handler.global_dof(right[j + 1], "u", "x")] += dy / 6 * (t0 + 2 * t1)
    k = sp.csc_matrix(stiffness[free][:, free])
    return float(0.5 * f[free] @ spla.spsolve(k, f[free]))


def test_both_nonlocal_strain_spaces_converge_to_the_same_solution_for_a_smooth_load() -> None:
    """For mu > 0 both e* spaces are consistent discretizations of the same problem, so under
    pure bending their strain energies must close on each other as the mesh is refined. They
    are *not* close on a mesh that does not resolve e0a (h = 0.5 vs e0a = 0.1 here: gap ~7%),
    where the penalty over-stiffens the unresolved discontinuous e*."""
    gaps = []
    for n_y in (2, 8):
        c0, dg = (_bending_energy(c, n_y) for c in (Continuity.C0, Continuity.DISCONTINUOUS))
        gaps.append(abs(dg - c0) / c0)
    assert gaps[1] < 0.05 * gaps[0]
    assert gaps[1] < 2e-3
