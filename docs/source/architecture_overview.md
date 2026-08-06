# Architecture overview

NanoFEM is a strictly layered architecture with strategy-pattern injection
for all physics and numerics. Full detail: `docs/design/ARCHITECTURE_v2.md`
and the SDS. The one-paragraph version:

**Contribution-based assembly** (ADR-001/008) is the load-bearing idea: the
assembler consumes `(kind, role, rows, cols, block)` tuples from providers
and is the only writer of global operators. Kinds `{CELL, FACET, PAIR, EDGE,
VERTEX}` cover local, boundary, nonlocal-pair, edge, and point physics; roles
`{STIFFNESS, MASS, ...}` are how analyses request operators without knowing
formulas.

**Three enforced rules** (import-linter, CI-blocking):
R1 — `numerics/` never imports mechanics; R2 — `physics/` never imports
discretization (it receives evaluated kinematic batches); R3 — `io/` and
`symbolics/` are thin edges never imported by the core.

**Reading order:** numerics → physics → elements → analysis.

Two layer-order consequences discovered while building phase 0 (recorded in
`docs/dev/notes.md`): physics declares field requirements as plain
`(name, n_components)` data because `core.FieldSpec` sits above it, and the
`Continuity` enum lives in `numerics/operators` because both operators and
interpolation share it while R2 forbids physics from importing interpolation.
