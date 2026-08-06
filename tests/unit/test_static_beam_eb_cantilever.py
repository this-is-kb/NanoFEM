"""EulerBernoulliBeam through the full Mesh -> Model -> LinearStaticAnalysis pipeline.

Before this file, ``EulerBernoulliBeam`` had never been solved through ``Model`` at all -
``elements/factory.py``'s dispatch (v0.14.0) was only checked against a directly-constructed
element's *stiffness matrix* (``test_elements_factory.py``), never a real boundary-condition +
load solve, and every beam benchmark (``test_beam_eb_cantilever_benchmark.py``) built the
element by hand. This file closes that gap: a cantilever beam (fixed at one end, transverse tip
load at the other) solved end to end, checked against the classical closed-form tip
deflection/rotation *and* the recovered bending moment at the fixed end.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.analysis.static import LinearStaticAnalysis
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import NodalLoad
from nanofem.core.model import DomainDefinition, Model
from nanofem.elements.factory import build_elements
from nanofem.elements.structural.beam_eb import EulerBernoulliBeam
from nanofem.geometry.standard import CircularSection
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.physics.elasticity.euler_bernoulli import EulerBernoulliBendingTheory

YOUNG_MODULUS = 200e9
RADIUS = 0.01
LENGTH = 1.5
TIP_FORCE = 1_000.0


def _cantilever_model() -> tuple[Model, float]:
    """One line2 beam, fixed at node 0 (w=theta=0), transverse tip load at node 1."""
    coords = np.array([[0.0], [LENGTH]])
    block = CellBlock("line2", np.array([[0, 1]]), region="beam")
    mesh = Mesh(coords, (block,), (Region("fixed", 0, (0,)), Region("tip", 0, (1,))))
    model = Model(mesh)
    section = CircularSection(radius=RADIUS)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=0.3))
    model.add_section("circ", section)
    model.add_theory("bending", EulerBernoulliBendingTheory())
    model.add_domain(DomainDefinition("beam_domain", "beam", "bending", "steel", "circ"))
    model.add_dirichlet(DirichletBC("fixed", "u", ("y",), 0.0))
    model.add_dirichlet(DirichletBC("fixed", "r", ("z",), 0.0))
    case = LoadCase("service")
    case.add(NodalLoad("tip", "u", np.array([TIP_FORCE])))
    model.add_load_case(case)
    return model, section.second_moment_z()


def test_tip_deflection_and_rotation_match_the_classical_cantilever_formulas() -> None:
    model, i = _cantilever_model()
    result = LinearStaticAnalysis(model).run()["service"]
    dof_handler = result.dof_handler
    ei = YOUNG_MODULUS * i

    w_tip = result.displacements[dof_handler.global_dof(1, "u", "y")]
    theta_tip = result.displacements[dof_handler.global_dof(1, "r", "z")]
    assert w_tip == pytest.approx(TIP_FORCE * LENGTH**3 / (3.0 * ei), rel=1e-9)
    assert theta_tip == pytest.approx(TIP_FORCE * LENGTH**2 / (2.0 * ei), rel=1e-9)

    w_fixed = result.displacements[dof_handler.global_dof(0, "u", "y")]
    theta_fixed = result.displacements[dof_handler.global_dof(0, "r", "z")]
    assert w_fixed == pytest.approx(0.0, abs=1e-15)
    assert theta_fixed == pytest.approx(0.0, abs=1e-15)


def test_reaction_matches_the_applied_tip_force() -> None:
    model, _ = _cantilever_model()
    result = LinearStaticAnalysis(model).run()["service"]
    # Constrained DOFs, sorted (SDS C-5): (node0, u.y) then (node0, r.z). The transverse
    # reaction balances the tip force; the reaction moment balances the tip force's moment
    # about the fixed end (M = -P*L, since equilibrium is R = ... - f_c, SDS 2.14).
    assert result.reactions[0] == pytest.approx(-TIP_FORCE, rel=1e-9)
    assert result.reactions[1] == pytest.approx(-TIP_FORCE * LENGTH, rel=1e-9)


def test_bending_moment_recovers_correctly_through_the_full_pipeline() -> None:
    """Displacements -> stress recovery, closing the pipeline for EulerBernoulliBeam.

    Uses the real ``build_elements`` factory to get the solved beam back (not a hand-built
    duplicate), then its own ``curvature_response`` on the recovered local displacements -
    the classical result ``M(fixed end) = P*L``, ``M(tip) = 0``.
    """
    model, _ = _cantilever_model()
    result = LinearStaticAnalysis(model).run()["service"]
    dof_handler = result.dof_handler
    (element,) = build_elements(model, dof_handler)
    assert isinstance(element, EulerBernoulliBeam)
    local_displacement = result.displacements[np.array(element.global_dofs)]
    response = element.curvature_response(local_displacement)
    assert response.moment[0] == pytest.approx(TIP_FORCE * LENGTH, rel=1e-9)
    assert response.moment[1] == pytest.approx(0.0, abs=1e-6)
