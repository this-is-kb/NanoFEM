# Package guide

Authoritative responsibilities table: ARCHITECTURE v2, Section 2 (owns /
must-not-do per package), reproduced in `docs/design/ARCHITECTURE_v2.md`.
Quick orientation:

| Package | One line |
|---|---|
| `utils` | exceptions, logging, validation, scaling; imports nothing internal |
| `numerics` | reference cells, interpolation, quadrature, mapping, operators, tensors, assembly, linalg, time integration, search — mechanics-free (R1) |
| `mesh` | domain topology and regions; algorithms live in numerics.search |
| `geometry` | cross-sections and thickness (NOT the meshed domain) |
| `materials` | validated property records; zero constitutive math |
| `kernels` | attenuation functions of distance for nonlocal theories |
| `state` | trial/committed quadrature-point variable banks |
| `physics` | theories: kinematics + constitution + weak forms (R2) |
| `core` | Model facade, DofHandler, FieldSpec, Registry |
| `elements` | composition of numerics + physics + geometry into providers |
| `constraints` | BC/load/MPC descriptions and DOF partitioning |
| `analysis` | orchestration only; requests operators by role |
| `postprocess` | recovery, sampling, diagrams, export, plots |
| `io` / `symbolics` | thin edges (R3): format conversion / SymPy oracles |
