# Getting started with NanoFEM

This is a first-time-user walkthrough: install NanoFEM, understand the handful of objects every
model is built from, and run one complete worked example end to end. It assumes no prior
familiarity with the codebase.

## What NanoFEM is

NanoFEM is a research-grade finite element solver, in pure Python (numpy/scipy). It ships a
complete classical FEM backbone (bars, beams, plane-stress/strain triangles and quadrilaterals)
and, on top of that same backbone, Eringen nonlocal elasticity (differential form) for
nanoscale mechanics — switching a model from classical to nonlocal elasticity never touches the
mesh, the element library, or the solver, only the physics theory attached to it.

## Prerequisites

- Python 3.11 or newer
- `pip`

## Installation

Open a terminal in the `nanofem/` directory (the one containing `pyproject.toml`) and run:

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

**macOS/Linux (bash):**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs NanoFEM in "editable" mode along with its core dependencies (`numpy`, `scipy`,
`matplotlib`, `meshio`). You only need the extra `pip install -e ".[dev]"` form if you plan to
run the test suite or the linting/formatting tools — not for using the library.

## Verify the install

```powershell
python -c "import nanofem; print(nanofem.__version__)"
```

This should print `0.25.0` (or later). If it raises `ModuleNotFoundError: No module named
'nanofem'`, see [Troubleshooting](#troubleshooting) below.

## Using a code editor (VS Code)

PowerShell and a code editor are not alternatives to each other — they do different jobs, and
you use both together:

- **PowerShell (or any terminal)** is where you *run* things: install packages, run a script.
- **VS Code (or any editor)** is where you *write and read* the actual `.py` files.

The normal workflow is to open the project folder in VS Code and use its **built-in terminal**
to run commands, so you never need a separate terminal window:

1. Open VS Code, then `File → Open Folder...` and select the `nanofem/` directory (the one
   containing `pyproject.toml`).
2. Open the built-in terminal: `Terminal → New Terminal` (or `` Ctrl+` ``). This is the same
   PowerShell you used for installation, just docked inside the editor window.
3. Activate the virtual environment in that terminal if it isn't already active:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
4. Use the file explorer panel on the left to open any `.py` file (e.g. under `examples/`) and
   read or edit it.
5. To run a script, either type `python examples/ex08_bar_under_end_load.py` in the terminal, or
   (with the Python extension installed) click the ▶ "Run" arrow at the top-right of the editor.

In short: **edit code in the editor pane, run it in the terminal pane** — both inside the same
VS Code window.

## Core concepts, briefly

Every NanoFEM model is assembled from the same small set of objects, regardless of what physics
you're solving:

| Object | What it is |
|---|---|
| **Mesh** | Node coordinates, element connectivity (`CellBlock`), and named groups of nodes/cells (`Region`) that boundary conditions and loads attach to |
| **Material** | Physical properties — Young's modulus `E`, Poisson's ratio `nu`, density `rho`, etc. |
| **Section / Geometry** | Cross-sectional shape for 1-D members (e.g. `CircularSection`) or thickness for 2-D plane problems (`PlaneGeometry`) |
| **Theory** | The physics being solved — e.g. `IsotropicElasticity` (classical elasticity) |
| **Model** | Ties mesh, material, section, and theory together into named `DomainDefinition`s |
| **Boundary conditions & loads** | `DirichletBC` (prescribed displacements) and a `LoadCase` of applied loads (e.g. `NodalLoad`) |
| **Analysis & Results** | `LinearStaticAnalysis` solves the model; the returned `Results` object holds nodal displacements and reactions |

## The seven-step workflow

Every NanoFEM script — from the simplest bar to a full nonlocal-elasticity benchmark — follows
the same shape:

1. Build the **mesh** (node coordinates + element connectivity + named regions).
2. Define the **material**.
3. Define the **section/geometry**.
4. Choose the **theory** (the physics).
5. Assemble the **model** (bind mesh + material + section + theory into a domain).
6. Apply **boundary conditions and loads**.
7. **Run** the analysis and **read the results**.

The worked example below follows this checklist step by step.

## Worked example: a steel bar fixed at one end, loaded at the tip

This is `examples/ex08_bar_under_end_load.py`, already included in the repository. It builds a
1.5 m steel bar, clamped at one end, with a 1000 N tensile load at the tip — and checks its own
answer against the closed-form solution `u = P*L/(E*A)`.

### Step 1-3: mesh, material, section

```python
import numpy as np

from nanofem.core.model import DomainDefinition, Model
from nanofem.geometry.standard import CircularSection
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region

YOUNG_MODULUS = 200.0e9  # Pa, steel
RADIUS = 0.01  # m
LENGTH = 1.5  # m
TIP_FORCE = 1_000.0  # N

# Two nodes, one line element between them, with named end regions.
coordinates = np.array([[0.0], [LENGTH]])
block = CellBlock("line2", np.array([[0, 1]]), region="bar")
mesh = Mesh(coordinates, (block,), (Region("fixed", 0, (0,)), Region("tip", 0, (1,))))
```

`coordinates` holds one row per node; `block` connects node `0` to node `1` as a two-node line
element in the region named `"bar"`; `Region("fixed", 0, (0,))` and `Region("tip", 0, (1,))`
name node `0` and node `1` so later steps can refer to "the fixed end" and "the tip" instead of
raw node indices.

### Step 4-5: theory and model

```python
from nanofem.physics.elasticity.isotropic import IsotropicElasticity

model = Model(mesh)
model.add_material(Material("steel", E=YOUNG_MODULUS, nu=0.3, rho=7850.0))
model.add_section("circular", CircularSection(radius=RADIUS))
model.add_theory("axial", IsotropicElasticity())
model.add_domain(DomainDefinition("bar_domain", "bar", "axial", "steel", "circular"))
```

`IsotropicElasticity()` is the classical elasticity theory. `add_domain` binds everything
together: the `"bar"` region of the mesh is solved with the `"axial"` theory, the `"steel"`
material, and the `"circular"` cross-section.

### Step 6: boundary conditions and loads

```python
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import NodalLoad

model.add_dirichlet(DirichletBC("fixed", "u", ("x",), 0.0))

service = LoadCase("service")
service.add(NodalLoad("tip", "u", np.array([TIP_FORCE])))
model.add_load_case(service)
```

`DirichletBC` clamps the `x`-displacement at the `"fixed"` region to zero. `NodalLoad` applies
`TIP_FORCE` at the `"tip"` region, collected into a `LoadCase` named `"service"` (a model can
hold several named load cases and solve them independently).

### Step 7: run and read results

```python
from nanofem.analysis.static import LinearStaticAnalysis

result = LinearStaticAnalysis(model).run()["service"]
dof_handler = result.dof_handler

u_tip = result.displacements[dof_handler.global_dof(1, "u", "x")]
reaction = result.reactions[0]

print(f"u_tip    = {u_tip:.6e} m")
print(f"reaction = {reaction:.6f} N")
```

`LinearStaticAnalysis(model).run()` returns a dictionary keyed by load-case name. `dof_handler`
translates a `(node, field, component)` triple — "node 1's `u` displacement, `x` component" —
into the row index of the `displacements` array.

### Run it yourself

The full script (`examples/ex08_bar_under_end_load.py`) includes everything above plus an
additional cross-check that re-derives the same stiffness from lower-level building blocks
(shape functions, quadrature, and the strain-displacement operator) — a good next read once
you're comfortable with the steps above, but not necessary for using NanoFEM day to day.

Run it from the `nanofem/` directory (with your virtual environment activated):

```powershell
python examples/ex08_bar_under_end_load.py
```

Expected output:

```
=== build the model ===
  E = 2.000e+11 Pa, A = 3.141593e-04 m^2, L = 1.5 m

=== solve ===
  u_fixed = 0.000000e+00 m
  u_tip   = 2.387324e-05 m
  reaction at fixed end = -1000.000000 N

=== payoff: check against independent closed forms ===
  u_tip vs P*L/(E*A):        2.387324146e-05  vs  2.387324146e-05
  reaction vs -P:            -1000.000000  vs  -1000.000000
  closed-form K   =
[[ 41887902.04786391 -41887902.04786391]
 [-41887902.04786391  41887902.04786391]]
  composed-path K =
[[ 41887902.04786391 -41887902.04786391]
 [-41887902.04786391  41887902.04786391]]

all three checks passed - the walking skeleton is complete (v0.8.0)
```

If your own run prints these same numbers, your installation is working correctly.

## Second worked example: a simply supported beam

This is `examples/ex11_simply_supported_beam.py` — a 4 m solid steel rod, pinned (simply
supported) at both ends, with a 5 kN point load at midspan, checked against the classical
closed form `w_mid = P*L^3/(48*E*I)`. It follows the same seven-step workflow, one level up
from the bar: bending instead of axial extension, and a two-element mesh so a node lands
exactly at midspan where the load is applied.

### Step 1: mesh

A simply supported beam under a midspan point load needs a node exactly at midspan, so this
uses **two** elements (three nodes) rather than the one-element bar above:

```python
node_x = np.array([0.0, LENGTH / 2.0, LENGTH])
coordinates = node_x.reshape(-1, 1)
connectivity = np.array([[0, 1], [1, 2]])  # two beam elements: left-half, right-half
block = CellBlock("line2", connectivity, region="beam")
regions = (Region("left", 0, (0,)), Region("mid", 0, (1,)), Region("right", 0, (2,)))
mesh = Mesh(coordinates, (block,), regions)
```

### Steps 2-3: material and section

```python
model.add_material(Material("steel", E=200.0e9, nu=0.3))
model.add_section("circ", CircularSection(radius=0.05))
```

### Step 4: theory

```python
from nanofem.physics.elasticity.euler_bernoulli import EulerBernoulliBendingTheory

model.add_theory("bending", EulerBernoulliBendingTheory())
```

This is the only line that changes physics compared to the axial bar example above — the
mesh/solve pipeline is otherwise identical.

### Step 5: model/domain

```python
model.add_domain(DomainDefinition("beam_domain", "beam", "bending", "steel", "circ"))
```

### Step 6: boundary conditions and load — the "simply supported" detail

```python
model.add_dirichlet(DirichletBC("left", "u", ("y",), 0.0))
model.add_dirichlet(DirichletBC("right", "u", ("y",), 0.0))

service = LoadCase("service")
service.add(NodalLoad("mid", "u", np.array([5_000.0])))
model.add_load_case(service)
```

Only the **transverse displacement** (`u.y`) is fixed at each end — rotation (`r.z`) is left
free. That is what "simply supported" (pinned) means, as opposed to a cantilever, which also
fixes rotation at its support (see `examples/ex08_bar_under_end_load.py`'s beam counterpart,
`tests/unit/test_static_beam_eb_cantilever.py`, for that comparison).

### Step 7: run and check

```python
result = LinearStaticAnalysis(model).run()["service"]
w_mid = result.displacements[result.dof_handler.global_dof(1, "u", "y")]
```

### Run it yourself

```powershell
python examples/ex11_simply_supported_beam.py
```

Expected output:

```
=== build the model ===
  E = 2.000e+11 Pa, I = 4.908739e-06 m^4, EI = 9.817477e+05 N*m^2, L = 4.0 m

=== solve ===
  w_left  = 0.000000e+00 m
  w_mid   = 6.790611e-03 m
  w_right = 0.000000e+00 m
  reactions = [-2500. -2500.] N  (each should balance half the applied load)

=== payoff: check against the classical closed form ===
  w_mid vs P*L^3/(48*E*I):   6.790610905e-03  vs  6.790610905e-03
  each reaction vs -P/2:     -2500.000000  vs  -2500.000000

both checks passed - the simply supported beam matches the textbook solution
```

The FE answer matches the textbook formula to 9 significant figures, with no discretization
error at all — a concentrated load placed exactly at a node is exactly representable by the
Euler-Bernoulli element's cubic shape functions. Try changing `LENGTH`, `RADIUS`, or `MID_LOAD`
in the script and re-running to see the deflection scale accordingly.

## What's next

- `examples/ex09_eringen_differential_parametric_study.py` — generates publication-quality
  figures (nonlocal bar displacement profile, mesh-convergence study, characteristic-length
  softening curve).
- `examples/ex10_classical_to_eringen_theory_swap.py` — the same model solved twice, once with
  classical elasticity and once with Eringen nonlocal elasticity, changing only the theory —
  the clearest demonstration of how nonlocal elasticity plugs into the same backbone.
- `docs/design/` — the frozen architecture, the Software Design Specification, and one design
  document per physics/element family, for the mathematics behind each theory.
- `CONTRIBUTING.md` and `docs/source/developer_guide.md` — if you want to modify NanoFEM itself
  rather than just use it.

## Troubleshooting

**Commands from a copy-pasted block run together, or arguments go missing** — if you paste
several lines at once into PowerShell, the newlines between them can be lost, merging two
commands into one (e.g. `python -m venv .venv.venv\Scripts\Activate.ps1` instead of two
separate commands), or a trailing character like the `.` in `pip install -e .` can get dropped,
producing `pip install -e` with pip's usage help instead of an install. If any command's output
looks unexpected, type that one line by itself and press Enter, rather than pasting the whole
block. If a bad venv folder was already created this way (e.g. a stray `.venv.venv` directory),
delete it and the real `.venv` and start over:

```powershell
Remove-Item -Recurse -Force .venv, .venv.venv -ErrorAction SilentlyContinue
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

**`ModuleNotFoundError: No module named 'nanofem'`** — the editable install didn't run, or
you're in a different terminal session than the one where you ran `pip install -e .`. Re-run
the installation steps above, and confirm your virtual environment is activated (your prompt
should be prefixed with `(.venv)`).

**PowerShell refuses to run `Activate.ps1`** (`... cannot be loaded because running scripts is
disabled on this system`) — allow scripts for the current process only, then retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

**Forgot to activate the virtual environment** — every command above assumes `.venv` is active
for your current terminal session; if you open a new terminal, reactivate it first
(`.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` on macOS/Linux) before
running `pip` or `python` commands.
