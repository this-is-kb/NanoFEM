"""Facet regions + ``TractionLoadProvider``: a real surface-traction load, end to end.

Before this phase, ``TractionLoad``/``NeumannBC`` were pure dataclasses with
no integrator - a traction BC could only be approximated by hand-splitting
a total force across edge nodes (as ``test_static_t3_plate_analytical.py``
and ``test_postprocess_recovery.py`` both still do, for the simple
straight-edge case where the split is exact and obvious by hand). This file
verifies the real consistent-load integral against that same hand split
(the closed form for a *uniform* traction on a straight 2-node edge is
exactly ``traction * length / 2`` at each end node - a direct instance of
the same ``integral N_a f`` pattern ``ContinuumBodyForceProvider`` already
proved for body forces, v0.9.0), then re-solves the same uniaxial-tension T3
plate as ``test_static_t3_plate_analytical.py`` using a real ``TractionLoad``
instead of a hand-split ``NodalLoad``, and checks the two give identical
results.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.analysis.static import LinearStaticAnalysis
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import TractionLoad
from nanofem.constraints.traction import TractionLoadProvider
from nanofem.core.dof_handler import DofHandler
from nanofem.core.fields import FieldSpec
from nanofem.core.model import DomainDefinition, Model
from nanofem.geometry.plane import PlaneGeometry
from nanofem.materials.material import Material
from nanofem.mesh.facet_region import FacetRegion
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.numerics.assembly.contributions import ContributionKind, OperatorRole
from nanofem.physics.elasticity.isotropic import IsotropicElasticity
from nanofem.physics.elasticity.plane import PlaneStressConstitutive
from nanofem.utils.exceptions import InputValidationError, MeshError

YOUNG_MODULUS = 200e9
POISSON = 0.3
LENGTH = 2.0
HEIGHT = 1.0
THICKNESS = 0.01
TOTAL_FORCE = 10_000.0


def _plate_mesh() -> Mesh:
    coords = np.array([[0.0, 0.0], [LENGTH, 0.0], [LENGTH, HEIGHT], [0.0, HEIGHT]])
    block = CellBlock("tri3", np.array([[0, 1, 2], [0, 2, 3]]), region="plate")
    regions = (
        Region("left_edge", 0, (0, 3)),
        Region("origin", 0, (0,)),
        Region("right_edge_nodes", 0, (1, 2)),
    )
    # Triangle 0's facet 0 is local vertices (1, 2) -> global nodes (1, 2): the right edge.
    facet_regions = (FacetRegion("right_edge", ((0, 0),)),)
    return Mesh(coords, (block,), regions, facet_regions)


def test_facet_region_resolves_the_right_edge_nodes() -> None:
    mesh = _plate_mesh()
    assert mesh.facets_in_region("right_edge") == ((0, 0),)
    assert mesh.facet_node_ids(0, 0) == (1, 2)


def test_facet_region_rejects_out_of_range_cell() -> None:
    coords = np.array([[0.0, 0.0], [LENGTH, 0.0], [LENGTH, HEIGHT], [0.0, HEIGHT]])
    block = CellBlock("tri3", np.array([[0, 1, 2], [0, 2, 3]]), region="plate")
    with pytest.raises(MeshError, match="cell id"):
        Mesh(coords, (block,), facet_regions=(FacetRegion("bad", ((99, 0),)),))


def test_facet_region_rejects_duplicate_facets() -> None:
    with pytest.raises(InputValidationError, match="duplicate"):
        FacetRegion("r", ((0, 0), (0, 0)))


def test_traction_load_provider_matches_the_hand_split() -> None:
    """A uniform traction on a straight edge: consistent load = traction * length / 2 per node."""
    mesh = _plate_mesh()
    field_specs = (FieldSpec("u", ("x", "y")),)
    dof_handler = DofHandler.generate(mesh, field_specs)
    traction = np.array([500_000.0, 0.0])  # force per unit length along the edge
    load = TractionLoad("right_edge", "u", traction)
    provider = TractionLoadProvider(load, mesh, dof_handler, ("x", "y"))

    (contribution,) = list(provider.contributions(OperatorRole.FORCE))
    assert contribution.kind is ContributionKind.FACET
    expected_block = np.array([traction[0] * HEIGHT / 2.0, 0.0, traction[0] * HEIGHT / 2.0, 0.0])
    np.testing.assert_allclose(contribution.block, expected_block, rtol=1e-12)

    empty = list(provider.contributions(OperatorRole.STIFFNESS))
    assert empty == []


def _plate_model(*, use_traction: bool) -> Model:
    mesh = _plate_mesh()
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON))
    model.add_section("plane", PlaneGeometry(thickness=THICKNESS))
    model.add_theory("plane_stress_kinematics", IsotropicElasticity(dim=2))
    model.add_constitutive("plane_stress_law", PlaneStressConstitutive())
    model.add_domain(
        DomainDefinition(
            "plate_domain",
            "plate",
            "plane_stress_kinematics",
            "steel",
            geometry="plane",
            constitutive="plane_stress_law",
        )
    )
    model.add_dirichlet(DirichletBC("left_edge", "u", ("x",), 0.0))
    model.add_dirichlet(DirichletBC("origin", "u", ("y",), 0.0))
    case = LoadCase("tension")
    if use_traction:
        traction = np.array([TOTAL_FORCE / HEIGHT, 0.0])
        case.add(TractionLoad("right_edge", "u", traction))
    else:
        from nanofem.constraints.loads import NodalLoad

        case.add(NodalLoad("right_edge_nodes", "u", np.array([TOTAL_FORCE / 2.0, 0.0])))
    model.add_load_case(case)
    return model


def test_traction_load_reproduces_the_hand_split_nodal_load_result() -> None:
    """A real TractionLoad and the hand-split NodalLoad must solve to the same state."""
    traction_result = LinearStaticAnalysis(_plate_model(use_traction=True)).run()["tension"]
    nodal_result = LinearStaticAnalysis(_plate_model(use_traction=False)).run()["tension"]
    np.testing.assert_allclose(
        traction_result.displacements, nodal_result.displacements, rtol=1e-12, atol=1e-18
    )
    np.testing.assert_allclose(
        traction_result.reactions,
        nodal_result.reactions,
        rtol=1e-12,
        atol=1e-9 * np.abs(nodal_result.reactions).max(),
    )

    expected = TOTAL_FORCE * LENGTH / (YOUNG_MODULUS * HEIGHT * THICKNESS)
    dof_handler = traction_result.dof_handler
    for node in (1, 2):
        u_x = traction_result.displacements[dof_handler.global_dof(node, "u", "x")]
        assert u_x == pytest.approx(expected, rel=1e-9)
