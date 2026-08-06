"""LinearStaticAnalysis.run(): single bar, one end fixed, tip force P (SDS 2.18).

The full Mesh -> Model -> LinearStaticAnalysis chain, checked against the
classical closed form ``u = PL/(EA)`` and an independently (in-test)
recomputed reaction, not against ``ReducedSystem.reactions`` itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.analysis.static import LinearStaticAnalysis
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import NodalLoad
from nanofem.constraints.time_functions import ConstantTF
from nanofem.core.model import DomainDefinition, Model
from nanofem.elements.factory import build_elements
from nanofem.elements.structural.bar import Bar
from nanofem.geometry.standard import CircularSection
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.physics.elasticity.isotropic import IsotropicElasticity
from nanofem.utils.exceptions import ModelError

YOUNG_MODULUS = 200.0e9
RADIUS = 0.01
LENGTH = 1.5
TIP_FORCE = 1_000.0


def _bar_model() -> tuple[Model, float]:
    """One line2 bar, fixed at node 0, tip load at node 1; returns (model, area)."""
    coords = np.array([[0.0], [LENGTH]])
    block = CellBlock("line2", np.array([[0, 1]]), region="bar")
    mesh = Mesh(coords, (block,), (Region("fixed", 0, (0,)), Region("tip", 0, (1,))))
    model = Model(mesh)
    section = CircularSection(radius=RADIUS)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=0.3, rho=7850.0))
    model.add_section("circ", section)
    model.add_theory("axial", IsotropicElasticity())
    model.add_domain(DomainDefinition("bar_domain", "bar", "axial", "steel", "circ"))
    model.add_dirichlet(DirichletBC("fixed", "u", ("x",), 0.0))
    case = LoadCase("service")
    case.add(NodalLoad("tip", "u", np.array([TIP_FORCE])))
    model.add_load_case(case)
    return model, section.area()


def test_tip_displacement_matches_closed_form_pl_over_ea() -> None:
    model, area = _bar_model()
    result = LinearStaticAnalysis(model).run()["service"]
    dof_handler = result.dof_handler
    u_tip = result.displacements[dof_handler.global_dof(1, "u", "x")]
    assert u_tip == pytest.approx(TIP_FORCE * LENGTH / (YOUNG_MODULUS * area), rel=1e-9)
    u_fixed = result.displacements[dof_handler.global_dof(0, "u", "x")]
    assert u_fixed == pytest.approx(0.0)


def test_reaction_matches_independently_recomputed_value() -> None:
    """R = K_cf u_f + K_cc u_c - f_c, recomputed here from the raw K/f, not via ReducedSystem."""
    model, area = _bar_model()
    result = LinearStaticAnalysis(model).run()["service"]
    dof_handler = result.dof_handler
    assert result.reactions == pytest.approx([-TIP_FORCE], rel=1e-9)

    k_axial = YOUNG_MODULUS * area / LENGTH
    k_full = k_axial * np.array([[1.0, -1.0], [-1.0, 1.0]])
    f_full = np.array([0.0, TIP_FORCE])
    fixed_dof = dof_handler.global_dof(0, "u", "x")
    free_dof = dof_handler.global_dof(1, "u", "x")
    u_full = result.displacements
    reaction = k_full[fixed_dof, fixed_dof] * u_full[fixed_dof]
    reaction += k_full[fixed_dof, free_dof] * u_full[free_dof]
    reaction -= f_full[fixed_dof]
    assert reaction == pytest.approx(result.reactions[0], rel=1e-9)


def test_axial_stress_recovers_correctly_through_the_full_pipeline() -> None:
    """Displacements -> stress recovery, closing the pipeline for Bar (SDS 2.19).

    Uses the real ``build_elements`` factory to get the solved ``Bar`` back (not a
    hand-built duplicate), then its own ``axial_response`` on the recovered local
    displacements - the same round trip a stress-recovery-consuming caller would do.
    """
    model, area = _bar_model()
    result = LinearStaticAnalysis(model).run()["service"]
    dof_handler = result.dof_handler
    (element,) = build_elements(model, dof_handler)
    assert isinstance(element, Bar)
    local_displacement = result.displacements[np.array(element.global_dofs)]
    response = element.axial_response(local_displacement)
    expected_stress = TIP_FORCE / area
    assert response.stress == pytest.approx(expected_stress, rel=1e-9)
    assert response.force == pytest.approx(TIP_FORCE, rel=1e-9)


def test_default_selects_all_registered_load_cases() -> None:
    model, _ = _bar_model()
    results = LinearStaticAnalysis(model).run()
    assert set(results) == {"service"}


def test_time_function_on_a_load_entry_raises() -> None:
    model, _ = _bar_model()
    case = LoadCase("dynamic")
    case.add(NodalLoad("tip", "u", np.array([1.0])), time_function=ConstantTF(1.0))
    model.add_load_case(case)
    with pytest.raises(ModelError, match="time function"):
        LinearStaticAnalysis(model).run()
