# research/ — the science lives here (ADR-012, SDS Section 14)

`tests/` answers *is the code right* in seconds and gates merges;
`research/` answers *is the science reproducible* and runs nightly.

- `validation/` — the SDS Section 14 framework: evidence tiers T1
  (analytical), T2 (published literature), T3 (commercial FEM outputs +
  archived input decks), T4 (experimental, with the anti-circularity clause
  for fitted parameters). Each case ships metadata with DOIs, a
  deterministic model builder, QoI extractors, acceptance criteria, and a
  scorecard feeding the validation matrix.
- `benchmarks/` — performance tracking over time; never gates merges.
- `papers/<year>_<venue>_<slug>/` — environment lockfile, pinned nanofem
  tag, inputs, one `run_all` entry point, expected figures. `registry.md`
  maps every published figure to (script, tag).
