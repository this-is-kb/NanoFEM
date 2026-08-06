"""Eringen differential nonlocal elasticity: the fixed-fixed nonlocal bar benchmark.

Stage 4's first target - see ``docs/design/ERINGEN_DIFFERENTIAL_BAR.md`` for the full
derivation (verified two independent symbolic ways, plus a numerical mesh-convergence check
against the real NanoFEM quadrature/interpolation/mapping stack, before any production code
was written).

Problem: a bar of length L, fixed at both ends (u(0)=u(L)=0), under a sinusoidal distributed
axial load q(x) = q0*sin(pi*x/L), obeying Eringen's differential nonlocal constitutive law
(sigma - mu*sigma'' = E*u', mu = (e0*a)^2) combined with axial equilibrium. Closed form:

    u(x) = (q0 / (pi**2 * A * E)) * (L**2 + pi**2 * mu) * sin(pi*x/L)

No stiffness change from the classical bar is needed - the nonlocal effect is entirely a load
correction (``NonlocalAxialLoad``/``NonlocalAxialLoadProvider``), so this benchmark exercises
``Bar``/``IsotropicElasticity(dim=1)`` completely unmodified, through the real
``Mesh -> Model -> LinearStaticAnalysis`` pipeline.

This load shape is deliberately *not* uniform or a pure end load: both leave Eringen's
differential-model displacement field identical to the classical local solution (the
well-documented "Peddieson paradox" - see the design doc), which would make this benchmark
unable to discriminate a correct nonlocal implementation from a classical one at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.analysis.static import LinearStaticAnalysis
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import NonlocalAxialLoad
from nanofem.core.model import DomainDefinition, Model
from nanofem.geometry.standard import CircularSection
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.physics.elasticity.isotropic import IsotropicElasticity

YOUNG_MODULUS = 200e9
RADIUS = 0.008
LENGTH = 1.5
Q0 = 5.0e4
MU = 0.02  # (e0*a)^2, m^2 - deliberately large enough to be well above discretization error


def u_exact(x: np.ndarray, area: float, mu: float) -> np.ndarray:
    return (
        (Q0 / (np.pi**2 * area * YOUNG_MODULUS))
        * (LENGTH**2 + np.pi**2 * mu)
        * np.sin(np.pi * x / LENGTH)
    )


def _nonlocal_bar_model(n_elements: int, mu: float) -> Model:
    node_x = np.linspace(0.0, LENGTH, n_elements + 1)
    coords = node_x.reshape(-1, 1)
    connectivity = np.array([[i, i + 1] for i in range(n_elements)])
    block = CellBlock("line2", connectivity, region="bar")
    regions = (
        Region("left", 0, (0,)),
        Region("right", 0, (n_elements,)),
    )
    mesh = Mesh(coords, (block,), regions)

    model = Model(mesh)
    section = CircularSection(radius=RADIUS)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=0.3))
    model.add_section("circ", section)
    model.add_theory("axial", IsotropicElasticity(dim=1))
    model.add_domain(DomainDefinition("bar_domain", "bar", "axial", "steel", geometry="circ"))
    model.add_dirichlet(DirichletBC("left", "u", ("x",), 0.0))
    model.add_dirichlet(DirichletBC("right", "u", ("x",), 0.0))

    q_nodal = Q0 * np.sin(np.pi * node_x / LENGTH)
    case = LoadCase("nonlocal_axial")
    case.add(NonlocalAxialLoad("bar", "u", q_nodal, mu))
    model.add_load_case(case)
    return model


def _max_error(n_elements: int, mu: float) -> tuple[float, float]:
    model = _nonlocal_bar_model(n_elements, mu)
    result = LinearStaticAnalysis(model).run()["nonlocal_axial"]
    dof_handler = result.dof_handler
    node_x = np.linspace(0.0, LENGTH, n_elements + 1)
    area = CircularSection(radius=RADIUS).area()
    u_h = np.array(
        [result.displacements[dof_handler.global_dof(i, "u", "x")] for i in range(n_elements + 1)]
    )
    exact = u_exact(node_x, area, mu)
    return float(np.max(np.abs(u_h - exact))), float(np.max(np.abs(exact)))


@pytest.mark.parametrize("n_elements", [4, 8, 16, 32])
def test_nodal_displacements_converge_to_the_closed_form_solution(n_elements: int) -> None:
    error, scale = _max_error(n_elements, MU)
    # Isoparametric load interpolation gives O(h^2) convergence (verified numerically before
    # this file was written) - already well within 5% by 4 elements (2 elements alone is
    # coarser, ~16% relative error, checked separately by the monotonic sweep below).
    assert error / scale < 0.05


def test_mesh_convergence_is_monotonic_and_roughly_second_order() -> None:
    errors = [_max_error(n, MU)[0] for n in (2, 4, 8, 16, 32)]
    for coarse, fine in zip(errors, errors[1:], strict=False):
        assert fine < coarse
    # Halving the element size should cut the error by ~4x (O(h^2)); allow generous slack.
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    for ratio in ratios:
        assert 3.0 < ratio < 5.0


def test_zero_nonlocal_parameter_reduces_to_the_classical_sin_load_bar() -> None:
    """The full Model/LinearStaticAnalysis pipeline at mu=0 must match the classical
    sin-loaded fixed-fixed bar solution (mu absent from the formula entirely)."""
    error, scale = _max_error(32, 0.0)
    assert error / scale < 0.05


def test_nonlocal_correction_is_resolved_well_above_discretization_error() -> None:
    """The mu-dependent part of the closed form must not be swamped by mesh error at a
    moderate mesh size - otherwise this benchmark could not actually discriminate a correct
    nonlocal implementation from an incorrect (or purely classical) one."""
    n_elements = 16
    error_nonlocal, _ = _max_error(n_elements, MU)
    node_x = np.linspace(0.0, LENGTH, n_elements + 1)
    area = CircularSection(radius=RADIUS).area()
    nonlocal_effect = np.max(np.abs(u_exact(node_x, area, MU) - u_exact(node_x, area, 0.0)))
    assert nonlocal_effect > 10.0 * error_nonlocal


def _uniform_load_model(n_elements: int, mu: float) -> Model:
    node_x = np.linspace(0.0, LENGTH, n_elements + 1)
    coords = node_x.reshape(-1, 1)
    connectivity = np.array([[i, i + 1] for i in range(n_elements)])
    block = CellBlock("line2", connectivity, region="bar")
    regions = (Region("left", 0, (0,)), Region("right", 0, (n_elements,)))
    mesh = Mesh(coords, (block,), regions)
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=0.3))
    model.add_section("circ", CircularSection(radius=RADIUS))
    model.add_theory("axial", IsotropicElasticity(dim=1))
    model.add_domain(DomainDefinition("bar_domain", "bar", "axial", "steel", geometry="circ"))
    model.add_dirichlet(DirichletBC("left", "u", ("x",), 0.0))
    model.add_dirichlet(DirichletBC("right", "u", ("x",), 0.0))
    case = LoadCase("uniform")
    case.add(NonlocalAxialLoad("bar", "u", np.full(n_elements + 1, Q0), mu))
    model.add_load_case(case)
    return model


def test_uniform_load_shows_no_nonlocal_effect() -> None:
    """The 'Peddieson paradox': a uniform distributed load has q''=0, so mu drops out of the
    strong form entirely - the nonlocal and classical solutions must coincide exactly."""
    n_elements = 8
    local = LinearStaticAnalysis(_uniform_load_model(n_elements, 0.0)).run()["uniform"]
    nonlocal_ = LinearStaticAnalysis(_uniform_load_model(n_elements, MU)).run()["uniform"]
    np.testing.assert_allclose(local.displacements, nonlocal_.displacements, rtol=1e-10)
