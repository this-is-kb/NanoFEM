"""A simply supported steel beam under a midspan point load.

A 4 m solid steel rod, pinned (transverse displacement fixed, rotation free) at both ends, with
a 5 kN point load at midspan. Follows the same seven-step workflow as
``ex08_bar_under_end_load.py`` (docs/source/tutorials/getting_started.md), one level up:
Euler-Bernoulli bending instead of axial extension, and a two-element mesh so a node lands
exactly at midspan where the load is applied.

Because the load is a concentrated force at a node (not a distributed load along the span),
the classical Euler-Bernoulli beam element's cubic Hermite shape functions represent the exact
deflected shape between load points - so the FE answer matches the textbook closed form to
machine precision, not just approximately.
"""

from __future__ import annotations

import numpy as np

from nanofem.analysis.static import LinearStaticAnalysis
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import NodalLoad
from nanofem.core.model import DomainDefinition, Model
from nanofem.geometry.standard import CircularSection
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.physics.elasticity.euler_bernoulli import EulerBernoulliBendingTheory

YOUNG_MODULUS = 200.0e9  # Pa, steel
RADIUS = 0.05  # m
LENGTH = 4.0  # m
MID_LOAD = 5_000.0  # N


def build_model() -> tuple[Model, float]:
    """Mesh -> Material -> Section -> Theory -> Model, ready to solve. Returns (model, I)."""
    node_x = np.array([0.0, LENGTH / 2.0, LENGTH])
    coordinates = node_x.reshape(-1, 1)
    connectivity = np.array([[0, 1], [1, 2]])  # two beam elements: left-half, right-half
    block = CellBlock("line2", connectivity, region="beam")
    regions = (Region("left", 0, (0,)), Region("mid", 0, (1,)), Region("right", 0, (2,)))
    mesh = Mesh(coordinates, (block,), regions)

    model = Model(mesh)
    section = CircularSection(radius=RADIUS)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=0.3))
    model.add_section("circ", section)
    model.add_theory("bending", EulerBernoulliBendingTheory())
    model.add_domain(DomainDefinition("beam_domain", "beam", "bending", "steel", "circ"))

    # Simply supported: transverse displacement fixed at both ends, rotation left FREE
    # (a cantilever, by contrast, also fixes rotation at the support - see ex08's beam
    # counterpart, test_static_beam_eb_cantilever.py, for that comparison).
    model.add_dirichlet(DirichletBC("left", "u", ("y",), 0.0))
    model.add_dirichlet(DirichletBC("right", "u", ("y",), 0.0))

    service = LoadCase("service")
    service.add(NodalLoad("mid", "u", np.array([MID_LOAD])))
    model.add_load_case(service)
    return model, section.second_moment_z()


def main() -> None:
    """Build, solve, and check the simply supported beam against the closed form."""
    print("=== build the model ===")
    model, i = build_model()
    ei = YOUNG_MODULUS * i
    print(f"  E = {YOUNG_MODULUS:.3e} Pa, I = {i:.6e} m^4, EI = {ei:.6e} N*m^2, L = {LENGTH} m")

    print("\n=== solve ===")
    result = LinearStaticAnalysis(model).run()["service"]
    dof_handler = result.dof_handler
    w_left = result.displacements[dof_handler.global_dof(0, "u", "y")]
    w_mid = result.displacements[dof_handler.global_dof(1, "u", "y")]
    w_right = result.displacements[dof_handler.global_dof(2, "u", "y")]
    print(f"  w_left  = {w_left:.6e} m")
    print(f"  w_mid   = {w_mid:.6e} m")
    print(f"  w_right = {w_right:.6e} m")
    print(f"  reactions = {result.reactions} N  (each should balance half the applied load)")

    print("\n=== payoff: check against the classical closed form ===")
    w_expected = MID_LOAD * LENGTH**3 / (48.0 * ei)
    print(f"  w_mid vs P*L^3/(48*E*I):   {w_mid:.9e}  vs  {w_expected:.9e}")
    assert np.isclose(w_mid, w_expected, rtol=1e-9)

    reaction_expected = -MID_LOAD / 2.0
    print(f"  each reaction vs -P/2:     {result.reactions[0]:.6f}  vs  {reaction_expected:.6f}")
    assert np.isclose(result.reactions[0], reaction_expected, rtol=1e-9)
    assert np.isclose(result.reactions[1], reaction_expected, rtol=1e-9)

    print("\nboth checks passed - the simply supported beam matches the textbook solution")


if __name__ == "__main__":
    main()
