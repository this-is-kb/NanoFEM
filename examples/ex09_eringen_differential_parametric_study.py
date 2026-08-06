"""Eringen Differential Nonlocal Elasticity: parametric study and benchmark figures (Stage 4).

Generates three publication-quality figures, each reusing already-verified NanoFEM machinery
(no new physics here - this script is pure post-hoc visualization of results whose correctness
is independently proven in the test suite):

1. Nonlocal bar displacement profile for several characteristic lengths (v0.20.0's closed form,
   ``docs/design/ERINGEN_DIFFERENTIAL_BAR.md``) - shows the amplification
   ``[1 + mu*(pi/L)^2]`` directly.
2. Nonlocal beam mesh-convergence study (v0.23.0, ``docs/design/ERINGEN_DIFFERENTIAL_BEAM.md``)
   - log-log error vs. element count, confirming the O(h^2) rate the test suite already checks.
3. Characteristic-length parametric study on the 2-D cantilever (v0.24.0,
   ``docs/design/ERINGEN_DIFFERENTIAL_CONTINUUM.md`` Section 7) - tip deflection vs. e0a,
   showing nonlocal softening (increasing compliance with the characteristic length).

Figures are saved as PNG files next to this script's output directory (created on demand).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: this script must run without a display
import matplotlib.pyplot as plt
import numpy as np

from nanofem.analysis.static import LinearStaticAnalysis
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import NodalLoad, NonlocalTransverseLoad
from nanofem.core.model import DomainDefinition, Model
from nanofem.geometry.plane import PlaneGeometry
from nanofem.geometry.standard import CircularSection
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.physics.elasticity.eringen_differential import (
    EringenDifferentialMaterial,
    EringenDifferentialTheory,
)
from nanofem.physics.elasticity.euler_bernoulli import EulerBernoulliBendingTheory
from nanofem.physics.elasticity.plane import PlaneStressConstitutive

OUTPUT_DIR = Path(__file__).parent / "output"

_PLOT_STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


# ---------------------------------------------------------------------------
# Figure 1: nonlocal bar displacement profile for several characteristic lengths
# ---------------------------------------------------------------------------

YOUNG_MODULUS_BAR = 200e9
BAR_RADIUS = 0.01
BAR_LENGTH = 1.5
BAR_Q0 = 5.0e4


def _bar_u_exact(x: np.ndarray, mu: float, area: float) -> np.ndarray:
    return (
        (BAR_Q0 / (np.pi**2 * area * YOUNG_MODULUS_BAR))
        * (BAR_LENGTH**2 + np.pi**2 * mu)
        * np.sin(np.pi * x / BAR_LENGTH)
    )


def figure_1_nonlocal_bar_profile() -> None:
    area = CircularSection(radius=BAR_RADIUS).area()
    x = np.linspace(0.0, BAR_LENGTH, 400)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for e0a in (0.0, 0.1, 0.2, 0.3):
        mu = e0a**2
        ax.plot(x, 1e3 * _bar_u_exact(x, mu, area), label=f"$e_0a = {e0a:.2f}$")
    ax.set_xlabel("Axial position $x$ (m)")
    ax.set_ylabel("Displacement $u(x)$ (mm)")
    ax.set_title("Eringen differential nonlocal bar: $q(x) = q_0 \\sin(\\pi x / L)$")
    ax.legend(title="Nonlocal parameter")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_nonlocal_bar_profile.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: nonlocal beam mesh-convergence study
# ---------------------------------------------------------------------------

YOUNG_MODULUS_BEAM = 200e9
SECOND_MOMENT = 8.0e-6
BEAM_RADIUS = (4.0 * SECOND_MOMENT / np.pi) ** 0.25
BEAM_LENGTH = 3.0
BEAM_Q0 = 5.0e4
BEAM_MU = 0.05


def _beam_w_exact(x: np.ndarray, mu: float) -> np.ndarray:
    ei = YOUNG_MODULUS_BEAM * SECOND_MOMENT
    return (
        (BEAM_Q0 * BEAM_LENGTH**4 / (np.pi**4 * ei))
        * (1.0 + mu * (np.pi / BEAM_LENGTH) ** 2)
        * np.sin(np.pi * x / BEAM_LENGTH)
    )


def _beam_mesh_error(n_elements: int) -> float:
    node_x = np.linspace(0.0, BEAM_LENGTH, n_elements + 1)
    coords = node_x.reshape(-1, 1)
    connectivity = np.array([[i, i + 1] for i in range(n_elements)])
    block = CellBlock("line2", connectivity, region="beam")
    regions = (Region("left", 0, (0,)), Region("right", 0, (n_elements,)))
    mesh = Mesh(coords, (block,), regions)

    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS_BEAM, nu=0.3))
    model.add_section("circ", CircularSection(radius=BEAM_RADIUS))
    model.add_theory("bending", EulerBernoulliBendingTheory())
    model.add_domain(DomainDefinition("beam_domain", "beam", "bending", "steel", "circ"))
    model.add_dirichlet(DirichletBC("left", "u", ("y",), 0.0))
    model.add_dirichlet(DirichletBC("right", "u", ("y",), 0.0))

    q_nodal = BEAM_Q0 * np.sin(np.pi * node_x / BEAM_LENGTH)
    case = LoadCase("sinusoid")
    case.add(NonlocalTransverseLoad("beam", "u", q_nodal, BEAM_MU))
    model.add_load_case(case)

    result = LinearStaticAnalysis(model).run()["sinusoid"]
    dof_handler = result.dof_handler
    w_h = np.array(
        [result.displacements[dof_handler.global_dof(i, "u", "y")] for i in range(n_elements + 1)]
    )
    exact = _beam_w_exact(node_x, BEAM_MU)
    return float(np.max(np.abs(w_h - exact)))


def figure_2_beam_mesh_convergence() -> None:
    element_counts = np.array([2, 4, 8, 16, 32])
    errors = np.array([_beam_mesh_error(int(n)) for n in element_counts])

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.loglog(element_counts, errors, "o-", label="Nonlocal beam, $e_0a$ fixed")
    reference = errors[0] * (element_counts[0] / element_counts) ** 2
    ax.loglog(element_counts, reference, "k--", label=r"$O(h^2)$ reference")
    ax.set_xlabel("Number of elements")
    ax.set_ylabel("Max nodal displacement error (m)")
    ax.set_title("Nonlocal Euler-Bernoulli beam: mesh convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_beam_mesh_convergence.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: characteristic-length parametric study, 2-D cantilever
# ---------------------------------------------------------------------------

YOUNG_MODULUS_2D = 200e9
POISSON_2D = 0.3
THICKNESS_2D = 0.05
CANTILEVER_LENGTH = 2.0
CANTILEVER_HEIGHT = 0.4
TIP_FORCE = 1_000.0


def _cantilever_tip_deflection(n_x: int, n_y: int, e0a: float) -> float:
    xs = np.linspace(0.0, CANTILEVER_LENGTH, n_x + 1)
    ys = np.linspace(-CANTILEVER_HEIGHT / 2.0, CANTILEVER_HEIGHT / 2.0, n_y + 1)
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
    mesh = Mesh(coords, (block,), (Region("left", 0, left_nodes), Region("right", 0, right_nodes)))

    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS_2D, nu=POISSON_2D, e0a=e0a))
    model.add_section("plane", PlaneGeometry(thickness=THICKNESS_2D))
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
    model.add_dirichlet(DirichletBC("left", "u", ("x", "y"), 0.0))
    per_node_force = -TIP_FORCE / len(right_nodes)
    case = LoadCase("tip")
    case.add(NodalLoad("right", "u", np.array([0.0, per_node_force])))
    model.add_load_case(case)

    result = LinearStaticAnalysis(model).run()["tip"]
    dof_handler = result.dof_handler
    return float(
        np.mean([result.displacements[dof_handler.global_dof(n, "u", "y")] for n in right_nodes])
    )


def figure_3_characteristic_length_study() -> None:
    e0a_values = np.linspace(0.0, 0.25, 11)
    deflections = np.array([_cantilever_tip_deflection(10, 5, e0a) for e0a in e0a_values])

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(e0a_values, 1e3 * np.abs(deflections), "o-", color="darkred")
    ax.set_xlabel("Characteristic length $e_0a$ (m)")
    ax.set_ylabel("Tip deflection magnitude (mm)")
    ax.set_title("2-D cantilever: nonlocal softening vs. characteristic length")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_characteristic_length_study.png")
    plt.close(fig)


def main() -> None:
    # matplotlib's rcParams stub types each key as its own Literal, so a plain dict-of-floats
    # update() call cannot be checked precisely; this is a stub-precision limitation, not a
    # real type-safety gap (every key/value pair here is a documented, valid rcParam).
    plt.rcParams.update(_PLOT_STYLE)  # type: ignore[arg-type]
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Generating Figure 1: nonlocal bar displacement profile...")
    figure_1_nonlocal_bar_profile()
    print("Generating Figure 2: nonlocal beam mesh convergence...")
    figure_2_beam_mesh_convergence()
    print("Generating Figure 3: characteristic-length parametric study...")
    figure_3_characteristic_length_study()
    print(f"\nFigures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
