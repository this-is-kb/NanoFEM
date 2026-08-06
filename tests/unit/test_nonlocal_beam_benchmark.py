"""Eringen differential nonlocal elasticity: the simply-supported nonlocal beam benchmark.

Mirrors ``test_nonlocal_bar_benchmark.py``'s structure and discipline exactly, one derivative
order up. See ``docs/design/ERINGEN_DIFFERENTIAL_BEAM.md`` for the full derivation (verified
two independent symbolic ways, plus a numerical weak-form check against the real Hermite
interpolation stack - which caught a real bug, a missing shape-function-value rescaling, before
any production code was written).

Problem: a simply-supported Euler-Bernoulli beam (w(0)=w(L)=0, moment-free ends) under a
sinusoidal distributed transverse load q(x) = q0*sin(pi*x/L), obeying Eringen's differential
nonlocal bending relation (M* - mu*M*'' = E*I*kappa, mu = (e0*a)^2) combined with beam
equilibrium. Closed form:

    w(x) = (q0*L^4 / (pi^4*E*I)) * (1 + mu*(pi/L)^2) * sin(pi*x/L)

No stiffness change from the classical beam is needed - the nonlocal effect is entirely a load
correction (``NonlocalTransverseLoad``/``NonlocalTransverseLoadProvider``), so this benchmark
exercises ``EulerBernoulliBeam``/``EulerBernoulliBendingTheory`` completely unmodified, through
the real ``Mesh -> Model -> LinearStaticAnalysis`` pipeline.

As with the nonlocal bar, this load shape is deliberately not uniform or a pure point load: both
leave the differential model's deflection identical to the classical solution (the Peddieson
paradox), which would make the benchmark unable to discriminate a correct implementation from an
incorrect one.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.analysis.static import LinearStaticAnalysis
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import NonlocalTransverseLoad
from nanofem.core.model import DomainDefinition, Model
from nanofem.geometry.standard import CircularSection
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.physics.elasticity.euler_bernoulli import EulerBernoulliBendingTheory

YOUNG_MODULUS = 200e9
SECOND_MOMENT = 8.0e-6
RADIUS = (4.0 * SECOND_MOMENT / np.pi) ** 0.25  # circular section reproducing SECOND_MOMENT
LENGTH = 3.0
Q0 = 5.0e4
MU = 0.05


def w_exact(x: np.ndarray, mu: float) -> np.ndarray:
    ei = YOUNG_MODULUS * SECOND_MOMENT
    return (
        (Q0 * LENGTH**4 / (np.pi**4 * ei))
        * (1.0 + mu * (np.pi / LENGTH) ** 2)
        * np.sin(np.pi * x / LENGTH)
    )


def _simply_supported_beam_model(n_elements: int, mu: float) -> Model:
    node_x = np.linspace(0.0, LENGTH, n_elements + 1)
    coords = node_x.reshape(-1, 1)
    connectivity = np.array([[i, i + 1] for i in range(n_elements)])
    block = CellBlock("line2", connectivity, region="beam")
    regions = (Region("left", 0, (0,)), Region("right", 0, (n_elements,)))
    mesh = Mesh(coords, (block,), regions)

    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=0.3))
    model.add_section("circ", CircularSection(radius=RADIUS))
    model.add_theory("bending", EulerBernoulliBendingTheory())
    model.add_domain(DomainDefinition("beam_domain", "beam", "bending", "steel", "circ"))
    model.add_dirichlet(DirichletBC("left", "u", ("y",), 0.0))
    model.add_dirichlet(DirichletBC("right", "u", ("y",), 0.0))

    q_nodal = Q0 * np.sin(np.pi * node_x / LENGTH)
    case = LoadCase("sinusoid")
    case.add(NonlocalTransverseLoad("beam", "u", q_nodal, mu))
    model.add_load_case(case)
    return model


def _max_error(n_elements: int, mu: float) -> tuple[float, float]:
    model = _simply_supported_beam_model(n_elements, mu)
    result = LinearStaticAnalysis(model).run()["sinusoid"]
    dof_handler = result.dof_handler
    node_x = np.linspace(0.0, LENGTH, n_elements + 1)
    w_h = np.array(
        [result.displacements[dof_handler.global_dof(i, "u", "y")] for i in range(n_elements + 1)]
    )
    exact = w_exact(node_x, mu)
    return float(np.max(np.abs(w_h - exact))), float(np.max(np.abs(exact)))


@pytest.mark.parametrize("n_elements", [8, 16, 32])
def test_nodal_deflections_converge_to_the_closed_form_solution(n_elements: int) -> None:
    error, scale = _max_error(n_elements, MU)
    assert error / scale < 0.05


def test_mesh_convergence_is_monotonic_and_roughly_second_order() -> None:
    errors = [_max_error(n, MU)[0] for n in (4, 8, 16, 32)]
    for coarse, fine in zip(errors, errors[1:], strict=False):
        assert fine < coarse
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    for ratio in ratios:
        assert 1.5 < ratio < 5.0


def test_zero_nonlocal_parameter_reduces_to_the_classical_sin_load_beam() -> None:
    error, scale = _max_error(32, 0.0)
    assert error / scale < 0.05


def test_nonlocal_correction_is_resolved_well_above_discretization_error() -> None:
    n_elements = 16
    error_nonlocal, _ = _max_error(n_elements, MU)
    node_x = np.linspace(0.0, LENGTH, n_elements + 1)
    nonlocal_effect = np.max(np.abs(w_exact(node_x, MU) - w_exact(node_x, 0.0)))
    assert nonlocal_effect > 10.0 * error_nonlocal


def _uniform_load_model(n_elements: int, mu: float) -> Model:
    node_x = np.linspace(0.0, LENGTH, n_elements + 1)
    coords = node_x.reshape(-1, 1)
    connectivity = np.array([[i, i + 1] for i in range(n_elements)])
    block = CellBlock("line2", connectivity, region="beam")
    regions = (Region("left", 0, (0,)), Region("right", 0, (n_elements,)))
    mesh = Mesh(coords, (block,), regions)
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=0.3))
    model.add_section("circ", CircularSection(radius=RADIUS))
    model.add_theory("bending", EulerBernoulliBendingTheory())
    model.add_domain(DomainDefinition("beam_domain", "beam", "bending", "steel", "circ"))
    model.add_dirichlet(DirichletBC("left", "u", ("y",), 0.0))
    model.add_dirichlet(DirichletBC("right", "u", ("y",), 0.0))
    case = LoadCase("uniform")
    case.add(NonlocalTransverseLoad("beam", "u", np.full(n_elements + 1, Q0), mu))
    model.add_load_case(case)
    return model


def test_uniform_load_shows_no_nonlocal_effect() -> None:
    """The beam analogue of the Peddieson paradox: a uniform load has q''=0, so mu drops out
    of the strong form entirely - local and nonlocal solutions must coincide exactly."""
    n_elements = 8
    local = LinearStaticAnalysis(_uniform_load_model(n_elements, 0.0)).run()["uniform"]
    nonlocal_ = LinearStaticAnalysis(_uniform_load_model(n_elements, MU)).run()["uniform"]
    np.testing.assert_allclose(local.displacements, nonlocal_.displacements, rtol=1e-9)
