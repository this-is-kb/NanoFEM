"""Stress/strain recovery, principal values, von Mises, strain energy (SDS 2.19).

Verified against the same uniaxial-tension T3 plate as
``test_static_t3_plate_analytical.py``: a prescribed total force ``P`` on the
right edge produces the exact, closed-form uniaxial stress state
``sigma_xx = P/(H t)``, ``sigma_yy = tau_xy = 0`` everywhere - a clean,
hand-checkable target for every derived quantity this module computes
(principal stresses, von Mises, and the out-of-plane strain term), and
strain energy is cross-checked against Clapeyron's theorem
(``U = 0.5 * F . u`` for a linear-elastic system loaded from zero), a
formula independent of ``0.5 u^T K u`` itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.analysis.static import LinearStaticAnalysis
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import NodalLoad
from nanofem.core.model import DomainDefinition, Model
from nanofem.elements.continuum.continuum import ContinuumElement
from nanofem.elements.factory import build_elements
from nanofem.geometry.plane import PlaneGeometry
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.physics.elasticity.isotropic import IsotropicElasticity
from nanofem.physics.elasticity.plane import PlaneStrainConstitutive, PlaneStressConstitutive
from nanofem.postprocess.recovery import (
    RecoveryInput,
    recover_element_fields,
    recover_nodal_fields,
    strain_energy,
)
from nanofem.utils.exceptions import InputValidationError, PhysicsError

YOUNG_MODULUS = 200e9
POISSON = 0.3
LENGTH = 2.0
HEIGHT = 1.0
THICKNESS = 0.01
TOTAL_FORCE = 10_000.0


def _plate_model() -> Model:
    coords = np.array([[0.0, 0.0], [LENGTH, 0.0], [LENGTH, HEIGHT], [0.0, HEIGHT]])
    block = CellBlock("tri3", np.array([[0, 1, 2], [0, 2, 3]]), region="plate")
    regions = (
        Region("left_edge", 0, (0, 3)),
        Region("origin", 0, (0,)),
        Region("right_edge", 0, (1, 2)),
    )
    mesh = Mesh(coords, (block,), regions)
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
    case.add(NodalLoad("right_edge", "u", np.array([TOTAL_FORCE / 2.0, 0.0])))
    model.add_load_case(case)
    return model


def _recovery_inputs(model: Model) -> tuple[list[RecoveryInput], np.ndarray]:
    result = LinearStaticAnalysis(model).run()["tension"]
    dof_handler = result.dof_handler
    elements = build_elements(model, dof_handler)
    inputs = []
    for element in elements:
        assert isinstance(element, ContinuumElement)
        inputs.append(RecoveryInput(element, PlaneStressConstitutive(), POISSON))
    return inputs, result.displacements


def test_element_stress_matches_uniaxial_closed_form() -> None:
    model = _plate_model()
    inputs, displacements = _recovery_inputs(model)
    element_results = recover_element_fields(inputs, displacements)
    sigma_xx = TOTAL_FORCE / (HEIGHT * THICKNESS)

    for result in element_results:
        expected_stress = np.array([[sigma_xx, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(result.field.stress, expected_stress, atol=1.0e-6 * sigma_xx)
        np.testing.assert_allclose(
            result.field.principal_stresses, [0.0, 0.0, sigma_xx], atol=1.0e-6 * sigma_xx
        )
        assert result.field.von_mises_stress == pytest.approx(sigma_xx, rel=1e-9)

        eps_xx = sigma_xx / YOUNG_MODULUS
        eps_yy = -POISSON * eps_xx
        eps_zz = -POISSON / (1.0 - POISSON) * (eps_xx + eps_yy)
        expected_strain = np.array([[eps_xx, 0.0, 0.0], [0.0, eps_yy, 0.0], [0.0, 0.0, eps_zz]])
        np.testing.assert_allclose(result.field.strain, expected_strain, atol=1.0e-6 * eps_xx)
        # Plane stress's own defining condition, cross-checked independently of the reduction:
        assert eps_zz == pytest.approx(eps_yy, rel=1e-9)


def test_nodal_recovery_matches_the_uniform_element_field() -> None:
    """A uniform stress state must recover identically at every node (trivial averaging)."""
    model = _plate_model()
    inputs, displacements = _recovery_inputs(model)
    element_results = recover_element_fields(inputs, displacements)
    nodal = recover_nodal_fields(inputs, element_results)
    assert len(nodal) == 4
    sigma_xx = TOTAL_FORCE / (HEIGHT * THICKNESS)
    for field in nodal.values():
        assert field.von_mises_stress == pytest.approx(sigma_xx, rel=1e-9)
        np.testing.assert_allclose(
            field.stress, element_results[0].field.stress, atol=1e-9 * sigma_xx
        )


def test_strain_energy_matches_clapeyron_theorem() -> None:
    """``U = 0.5 F.u`` for a linear-elastic system loaded from zero - independent of 0.5 u^T K u."""
    model = _plate_model()
    inputs, displacements = _recovery_inputs(model)
    element_results = recover_element_fields(inputs, displacements)
    u_tip = displacements[np.argmax(np.abs(displacements))]
    expected = 0.5 * TOTAL_FORCE * u_tip
    assert strain_energy(element_results) == pytest.approx(expected, rel=1e-6)


def test_plane_strain_out_of_plane_stress_matches_hand_formula() -> None:
    """Plane strain: eps_zz = 0 exactly; sigma_zz = nu (sigma_xx + sigma_yy)."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    theory = IsotropicElasticity(dim=2)
    material = Material("steel", E=YOUNG_MODULUS, nu=POISSON)
    element = ContinuumElement(
        cell_id=0,
        node_ids=(0, 1, 2),
        coordinates=coords,
        global_dofs=np.arange(6, dtype=np.int64),
        cell_type="triangle",
        interpolation_order=1,
        theory=theory,
        constitutive=PlaneStrainConstitutive(),
        material=material,
        section_measure=1.0,
    )
    a, b = 0.001, -0.0004  # eps_xx, eps_yy; zero shear
    displacement = np.zeros(6)
    for i, (x, y) in enumerate(coords):
        displacement[2 * i] = a * x
        displacement[2 * i + 1] = b * y

    inputs = [RecoveryInput(element, PlaneStrainConstitutive(), POISSON)]
    (result,) = recover_element_fields(inputs, displacement)
    assert result.field.strain[2, 2] == pytest.approx(0.0, abs=1e-15)

    c = YOUNG_MODULUS / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))
    sigma_xx = c * ((1.0 - POISSON) * a + POISSON * b)
    sigma_yy = c * (POISSON * a + (1.0 - POISSON) * b)
    expected_sigma_zz = POISSON * (sigma_xx + sigma_yy)
    assert result.field.stress[2, 2] == pytest.approx(expected_sigma_zz, rel=1e-9)


def test_unsupported_constitutive_raises() -> None:
    from nanofem.physics.elasticity.isotropic import IsotropicElasticConstitutive

    with pytest.raises(PhysicsError, match="PlaneStressConstitutive|PlaneStrainConstitutive"):
        RecoveryInput(
            element=None,  # type: ignore[arg-type]
            constitutive=IsotropicElasticConstitutive(),
            poisson_ratio=POISSON,
        )


def test_quadrature_point_response_rejects_wrong_shape() -> None:
    model = _plate_model()
    dof_handler = model.build_dof_handler()
    elements = build_elements(model, dof_handler)
    element = elements[0]
    assert isinstance(element, ContinuumElement)
    with pytest.raises(InputValidationError):
        element.quadrature_point_response(np.zeros(3))
