# NanoFEM

**Research-grade finite element framework for nanoscale mechanics** — Eringen
nonlocal elasticity (differential and integral two-phase forms),
strain-gradient and couple-stress theories, surface elasticity, and auxetic
metamaterials — in pure Python (numpy/scipy core).

> **Status: phase 0 (architectural skeleton).** This repository contains the
> complete, tool-enforced architecture and **zero finite element
> mathematics** — by design. Phase 0 exists to verify that the frozen
> architecture and Software Design Specification (SDS) are internally
> consistent and implementable before any numerics land.

## Getting started

New to NanoFEM? [`docs/source/tutorials/getting_started.md`](docs/source/tutorials/getting_started.md)
walks through installation and one complete worked example (a bar under an end load), aimed at
first-time users rather than contributors.

## Design documents

The architecture is frozen and lives in this repository:

| Document | Contents |
|---|---|
| `docs/design/ARCHITECTURE.md` | v1 blueprint: principles, layering, testing, workflow |
| `docs/design/ARCHITECTURE_v2.md` | revision 2: material/constitutive split, physics package, numerics extraction |
| `docs/design/NanoFEM_SDS.md` | the Software Design Specification: every mathematical and software contract |

Every source package's `__init__.py` states its purpose, responsibilities,
future modules, and TODO list. Reading order for newcomers: **numerics** (how
we approximate) → **physics** (what we solve) → **elements** (where they
meet) → **analysis** (how a run is orchestrated).

## Installation (development)

```
pip install -e ".[dev]"
```

Optional extras: `[symbolic]` (sympy oracles), `[meshing]` (gmsh),
`[viz]` (pyvista).

## Phase-0 verification

The phase-0 success criteria are executable:

```
black --check src tests      # formatting
isort --check-only src tests # import order
ruff check src tests         # linting
mypy                         # strict type checking
lint-imports                 # architecture contracts (rules R1-R3, layering)
pytest                       # import, integrity, and layout tests
```

`lint-imports` is the phase-0 centerpiece: the layered dependency graph of
ARCHITECTURE v2 and the SDS rules R2/R3 are encoded as import-linter
contracts, so an architecture violation is a CI failure, not a review
comment.

## Repository map

```
src/nanofem/     the package (see per-package __init__ docstrings)
tests/           unit | symbolic | element | verification | convergence | regression (CI gate)
research/        validation/ (evidence tiers T1-T4) | benchmarks/ | papers/ (reproducibility)
docs/            Diataxis skeleton + design documents + developer notes
examples/        tutorial-grade scripts (sphinx-gallery targets)
templates/       plugin package template (entry-point registration)
```

## Roadmap

Phases 0-7 are defined in `docs/source/roadmap.md`; the invariant audited at
every phase review is **"existing interfaces touched: none."**

## License

MIT — see `LICENSE`. Cite via `CITATION.cff`.
