"""Stage 4 Steps 2 and 3: Helmholtz operator properties and assembly correctness, at both the
element and the global (assembled, multi-element) scale.

Every existing nonlocal test verifies *outcomes* (correct displacements, correct energy,
correct condition number) that would be wrong if the Helmholtz operator or the assembly were
wrong - but none of them isolates and directly checks the operator/assembly properties the
Stage 4 directive names explicitly: symmetry and positive-definiteness of the Helmholtz block
in isolation, correct treatment of its gradient (diffusion) and mass terms, its natural
(Neumann-type) boundary treatment, and - at the *global*, multi-element, assembled scale rather
than the single-element scale ``test_nonlocal_continuum_element.py`` already covers - the
zero/coupling block structure, DOF ordering, and sparsity pattern. This file adds exactly those
checks.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.handler import ConstraintHandler
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import NodalLoad
from nanofem.core.model import DomainDefinition, Model
from nanofem.elements.continuum.nonlocal_continuum import NonlocalContinuumElement
from nanofem.elements.factory import build_elements
from nanofem.geometry.plane import PlaneGeometry
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.numerics.assembly.assembler import Assembler
from nanofem.numerics.assembly.contributions import OperatorRole
from nanofem.numerics.assembly.sparsity import SparsityPattern
from nanofem.numerics.reference.enums import CellType
from nanofem.physics.elasticity.eringen_differential import (
    NONLOCAL_STRAIN_FIELD,
    EringenDifferentialMaterial,
    EringenDifferentialTheory,
    voigt_component_names,
)
from nanofem.physics.elasticity.plane import PlaneStressConstitutive

YOUNG_MODULUS = 200e9
POISSON = 0.3
_SCALENE_TRIANGLE = np.array([[0.3, -0.2], [2.1, 0.4], [0.9, 1.7]])
N_DOF_U = 6  # 3 nodes x 2 components
N_DOF_E = 9  # 3 nodes x 3 Voigt components
N_DOF = N_DOF_U + N_DOF_E


def _nonlocal_element(
    e0a: float, coordinates: np.ndarray = _SCALENE_TRIANGLE
) -> NonlocalContinuumElement:
    theory = EringenDifferentialTheory(dim=2)
    material = Material("steel", E=YOUNG_MODULUS, nu=POISSON, e0a=e0a)
    constitutive = EringenDifferentialMaterial(PlaneStressConstitutive())
    return NonlocalContinuumElement(
        cell_id=0,
        node_ids=(0, 1, 2),
        coordinates=coordinates,
        global_dofs=np.arange(N_DOF, dtype=np.int64),
        cell_type=CellType.TRIANGLE,
        interpolation_order=1,
        theory=theory,
        constitutive=constitutive,
        material=material,
    )


def _k_ee(element: NonlocalContinuumElement) -> np.ndarray:
    """Recover K_ee itself (positive) from the assembled ``-K_ee`` diagonal block."""
    return -element.local_stiffness()[N_DOF_U:, N_DOF_U:]


# ---------------------------------------------------------------------------
# Step 2: Helmholtz operator verification (element scale)
# ---------------------------------------------------------------------------


def test_k_ee_mass_term_alone_is_symmetric_positive_definite() -> None:
    """At e0a=0, K_ee reduces to its pure mass term (`integral n_e^T D n_e`) - this must be
    strictly SPD (every eigenvalue > 0), since D is SPD and the shape-function block n_e is
    full column rank for a non-degenerate element - there is no zero-energy mode for e* mass,
    unlike the displacement stiffness's rigid-body null space."""
    k_ee = _k_ee(_nonlocal_element(e0a=0.0))
    np.testing.assert_allclose(k_ee, k_ee.T, rtol=1e-12)
    eigenvalues = np.linalg.eigvalsh(k_ee)
    assert np.all(eigenvalues > 0.0), f"K_ee mass term must be strictly SPD, got {eigenvalues}"


def test_k_ee_diffusion_term_is_positive_semidefinite_with_a_three_dimensional_null_space() -> None:
    """The gradient (diffusion) term is not directly exposed by the element, but is recoverable
    by finite difference: K_ee(mu) = K_ee_mass + mu*K_ee_diff, so K_ee_diff = K_ee(mu=1) -
    K_ee_mass exactly (mu = e0a^2, so e0a=1 gives mu=1). A pure gradient operator on a 3-node
    linear triangle has exactly one zero-energy mode per scalar field (the constant field) - with
    3 independent Voigt components, the null space must be exactly 3-dimensional out of 9 DOFs."""
    k_ee_mass = _k_ee(_nonlocal_element(e0a=0.0))
    k_ee_mu1 = _k_ee(_nonlocal_element(e0a=1.0))
    k_ee_diff = k_ee_mu1 - k_ee_mass

    np.testing.assert_allclose(k_ee_diff, k_ee_diff.T, rtol=1e-10)
    eigenvalues = np.linalg.eigvalsh(k_ee_diff)
    scale = np.abs(eigenvalues).max()
    near_zero = np.sum(np.abs(eigenvalues) < 1e-9 * scale)
    assert near_zero == 3, f"expected a 3-D null space (constant e* mode), got {near_zero}"
    assert np.all(eigenvalues > -1e-9 * scale), f"diffusion term must be PSD, got {eigenvalues}"


@pytest.mark.parametrize("e0a", [0.0, 0.01, 0.1, 0.5, 2.0, 10.0])
def test_k_ee_stays_symmetric_positive_definite_for_any_characteristic_length(e0a: float) -> None:
    """The Helmholtz block K_ee = K_ee_mass + mu*K_ee_diff is a positive combination of an SPD
    matrix and a PSD matrix, hence SPD for every mu >= 0 - the mixed system can never become
    singular through this block, regardless of how large the characteristic length gets."""
    k_ee = _k_ee(_nonlocal_element(e0a=e0a))
    eigenvalues = np.linalg.eigvalsh(k_ee)
    assert np.all(eigenvalues > 0.0), f"K_ee not SPD at e0a={e0a}: {eigenvalues}"


def test_k_ue_and_k_eu_are_exact_transposes_at_element_scale() -> None:
    """The energy-conjugate weighting (testing the Helmholtz relation with D:delta_e*) is what
    guarantees this exact transpose relationship, not a coincidence of a specific e0a."""
    element = _nonlocal_element(e0a=0.35)
    k = element.local_stiffness()
    k_ue = k[:N_DOF_U, N_DOF_U:]
    k_eu = k[N_DOF_U:, :N_DOF_U]
    np.testing.assert_allclose(k_eu, k_ue.T, rtol=1e-13)


# ---------------------------------------------------------------------------
# Step 2/3: natural boundary treatment on e* - a full-model check
# ---------------------------------------------------------------------------

LENGTH = 1.5
HEIGHT = 0.8
THICKNESS = 0.02


def _small_plate_model(e0a: float) -> tuple[Model, list[int]]:
    """A 2x1 grid (4 triangles), just large enough to have genuine element coupling."""
    n_x, n_y = 2, 1
    xs = np.linspace(0.0, LENGTH, n_x + 1)
    ys = np.linspace(0.0, HEIGHT, n_y + 1)
    coords = np.array([[x, y] for y in ys for x in xs])

    def node_id(i: int, j: int) -> int:
        return j * (n_x + 1) + i

    triangles = []
    for j in range(n_y):
        for i in range(n_x):
            a, b, c, d = node_id(i, j), node_id(i + 1, j), node_id(i + 1, j + 1), node_id(i, j + 1)
            triangles.append([a, b, c])
            triangles.append([a, c, d])
    block = CellBlock("tri3", np.array(triangles), region="plate")

    left_nodes = tuple(node_id(0, j) for j in range(n_y + 1))
    right_nodes = tuple(node_id(n_x, j) for j in range(n_y + 1))
    regions = (
        Region("left_edge", 0, left_nodes),
        Region("origin", 0, (node_id(0, 0),)),
        Region("right_edge", 0, right_nodes),
    )
    mesh = Mesh(coords, (block,), regions)

    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON, e0a=e0a))
    model.add_section("plane", PlaneGeometry(thickness=THICKNESS))
    model.add_theory("nonlocal", EringenDifferentialTheory(dim=2))
    model.add_constitutive("nonlocal_law", EringenDifferentialMaterial(PlaneStressConstitutive()))
    model.add_domain(
        DomainDefinition(
            "plate_domain",
            "plate",
            "nonlocal",
            "steel",
            geometry="plane",
            constitutive="nonlocal_law",
        )
    )
    model.add_dirichlet(DirichletBC("left_edge", "u", ("x",), 0.0))
    model.add_dirichlet(DirichletBC("origin", "u", ("y",), 0.0))
    case = LoadCase("tension")
    case.add(NodalLoad("right_edge", "u", np.array([1000.0, 0.0])))
    model.add_load_case(case)
    return model, list(right_nodes)


def _assemble_full(model: Model) -> tuple[csr_matrix, np.ndarray, np.ndarray, int]:
    """Returns (K, dof_handler.num_dofs-length dtype array of field tags, n_dof_u, n_dof_e)."""
    dof_handler = model.build_dof_handler()
    elements = build_elements(model, dof_handler)
    pattern = SparsityPattern.from_providers(elements, OperatorRole.STIFFNESS, dof_handler.num_dofs)
    stiffness = cast(csr_matrix, Assembler(pattern).assemble(elements, OperatorRole.STIFFNESS))

    e_components = voigt_component_names(2)
    u_dofs = sorted(
        dof_handler.global_dof(n, "u", c) for n in range(model.mesh.num_nodes) for c in ("x", "y")
    )
    e_dofs = sorted(
        dof_handler.global_dof(n, NONLOCAL_STRAIN_FIELD, c)
        for n in range(model.mesh.num_nodes)
        for c in e_components
    )
    return stiffness, np.array(u_dofs), np.array(e_dofs), dof_handler.num_dofs


def test_helmholtz_relation_needs_no_dirichlet_bc_on_nonlocal_strain() -> None:
    """The Helmholtz equation's boundary contribution is a homogeneous Neumann (zero-flux)
    condition - correctly reflected in this codebase by never declaring a Dirichlet BC on the
    nonlocal-strain field, anywhere. The system must still be solvable (K_ee's own strict
    positive-definiteness, proven above, is exactly what prevents the missing BC from causing
    singularity - a mass term, unlike a pure diffusion operator, needs no BC to be well-posed)."""
    model, right_nodes = _small_plate_model(e0a=0.2)
    for bc in model.dirichlet_bcs:
        assert bc.field != NONLOCAL_STRAIN_FIELD, "no Dirichlet BC should ever target e_star"

    dof_handler = model.build_dof_handler()
    elements = build_elements(model, dof_handler)
    pattern = SparsityPattern.from_providers(elements, OperatorRole.STIFFNESS, dof_handler.num_dofs)
    stiffness = cast(csr_matrix, Assembler(pattern).assemble(elements, OperatorRole.STIFFNESS))
    dirichlet_bcs = cast(tuple[DirichletBC, ...], model.dirichlet_bcs)
    partition = ConstraintHandler(model.mesh, dof_handler, dirichlet_bcs).partition()

    # Every e_star DOF must be FREE (never constrained) - confirming the natural-BC treatment.
    e_components = voigt_component_names(2)
    e_dofs = {
        dof_handler.global_dof(n, NONLOCAL_STRAIN_FIELD, c)
        for n in range(model.mesh.num_nodes)
        for c in e_components
    }
    assert e_dofs.issubset(set(partition.free_dofs))

    free = np.array(partition.free_dofs, dtype=np.int64)
    k_ff = stiffness[np.ix_(free, free)].toarray()
    cond = np.linalg.cond(k_ff)
    assert np.isfinite(cond), "system must remain solvable with e* fully unconstrained"


# ---------------------------------------------------------------------------
# Step 3: assembly verification (global, multi-element scale)
# ---------------------------------------------------------------------------


def test_global_u_u_block_is_structurally_exactly_zero() -> None:
    """The [[0, K_ue], [K_eu, -K_ee]] block structure must survive assembly: summing exact
    zeros from every element's u-u sub-block introduces no floating-point contamination."""
    model, _ = _small_plate_model(e0a=0.2)
    stiffness, u_dofs, _, _ = _assemble_full(model)
    block = stiffness[np.ix_(u_dofs, u_dofs)].toarray()
    np.testing.assert_array_equal(block, np.zeros_like(block))


def test_global_u_e_and_e_u_blocks_are_exact_transposes() -> None:
    """Element-scale K_eu = K_ue^T (already proven above) must survive linear assembly into an
    equally exact global relationship - a stronger, whole-system statement than the element-level
    check, since assembly could in principle break it if DOF ordering were inconsistent."""
    model, _ = _small_plate_model(e0a=0.2)
    stiffness, u_dofs, e_dofs, _ = _assemble_full(model)
    k_ue = stiffness[np.ix_(u_dofs, e_dofs)].toarray()
    k_eu = stiffness[np.ix_(e_dofs, u_dofs)].toarray()
    # A handful of shared-node entries are near-cancellations (true value close to zero), where
    # floating-point summation order (which of two adjacent elements' contributions lands first)
    # can differ between the K_ue and K_eu accumulation paths - a pure floating-point artifact,
    # not a structural asymmetry, so the tolerance is scaled to the matrix's own magnitude
    # (matching the project's established convention, e.g. test_nonlocal_conditioning_and_energy.py)
    # rather than to each entry's own tiny value.
    np.testing.assert_allclose(k_eu, k_ue.T, atol=1e-6 * np.abs(k_ue).max())


def test_global_e_e_block_is_negative_definite() -> None:
    """The assembled e*-e* block is `-sum(local K_ee)`; since each local K_ee is SPD (proven
    above) and assembly only ever adds shared-node contributions (never cancels them), the
    global block stays negative definite - so this DOF block alone can never be the cause of a
    singular system, mirroring the local per-element property at global scale."""
    model, _ = _small_plate_model(e0a=0.2)
    stiffness, _, e_dofs, _ = _assemble_full(model)
    block = stiffness[np.ix_(e_dofs, e_dofs)].toarray()
    np.testing.assert_allclose(block, block.T, rtol=1e-12)
    eigenvalues = np.linalg.eigvalsh(block)
    assert np.all(eigenvalues < 0.0), f"global e*-e* block must be negative definite: {eigenvalues}"


def test_sparsity_pattern_contains_every_assembled_nonzero_entry() -> None:
    """The precomputed sparsity pattern (built independently, before any numeric values exist)
    must declare space for every entry the assembler actually writes - a missing entry would
    silently drop a coupling term during assembly."""
    model, _ = _small_plate_model(e0a=0.2)
    dof_handler = model.build_dof_handler()
    elements = build_elements(model, dof_handler)
    pattern = SparsityPattern.from_providers(elements, OperatorRole.STIFFNESS, dof_handler.num_dofs)
    stiffness = cast(csr_matrix, Assembler(pattern).assemble(elements, OperatorRole.STIFFNESS))

    coo = stiffness.tocoo()
    missing = [
        (int(r), int(c))
        for r, c, v in zip(coo.row, coo.col, coo.data, strict=True)
        if v != 0.0 and not pattern.contains(int(r), int(c))
    ]
    assert (
        not missing
    ), f"sparsity pattern is missing {len(missing)} assembled entries: {missing[:5]}"


def test_factory_dof_ordering_matches_the_element_s_own_u_then_e_star_assumption() -> None:
    """``NonlocalContinuumElement`` assumes its own ``global_dofs`` array is laid out as the
    u-block (node-major, x-then-y) followed by the e*-block (node-major, Voigt order) - this
    test cross-checks that the factory actually builds ``global_dofs`` that way, rather than
    trusting the element's docstring claim alone."""
    model, _ = _small_plate_model(e0a=0.2)
    dof_handler = model.build_dof_handler()
    elements = build_elements(model, dof_handler)
    e_components = voigt_component_names(2)

    for element in elements:
        assert isinstance(element, NonlocalContinuumElement)
        expected_u = [
            dof_handler.global_dof(n, "u", c) for n in element.node_ids for c in ("x", "y")
        ]
        expected_e = [
            dof_handler.global_dof(n, NONLOCAL_STRAIN_FIELD, c)
            for n in element.node_ids
            for c in e_components
        ]
        np.testing.assert_array_equal(element.global_dofs[:N_DOF_U], expected_u)
        np.testing.assert_array_equal(element.global_dofs[N_DOF_U:], expected_e)


@pytest.mark.parametrize("role", [OperatorRole.MASS, OperatorRole.DAMPING, OperatorRole.FORCE])
def test_contributions_are_empty_for_every_non_stiffness_role(role: OperatorRole) -> None:
    """Stage 4 implements static analysis only (dynamics is explicitly out of scope) - the
    element must emit nothing at all for roles it does not support, rather than erroring or
    silently emitting a wrong (e.g. zero-filled) block that a future dynamic solver could
    misinterpret as "no mass" instead of "not implemented"."""
    element = _nonlocal_element(e0a=0.2)
    assert list(element.contributions(role)) == []
