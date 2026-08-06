# Contributing to NanoFEM

## Ground rules

1. **The architecture is frozen.** ARCHITECTURE v2 + the SDS (in
   `docs/design/`) govern every change; contradictions are resolved by ADR,
   never by drive-by edits. `lint-imports` enforces the layering in CI.
2. **Verification is a feature.** Any physics change ships its closed-form
   verification test (tests/verification) and, where relevant, convergence
   evidence in the PR description.
3. **Formatting is a machine's job.** Run `pre-commit install` once; black,
   isort, ruff, mypy, and import-linter run on every commit. Style is not a
   review topic.

## Workflow

- GitHub Flow: protected `main`, short-lived branches named
  `type/short-topic` (`feat/timoshenko-beam`, `fix/dof-partition-order`).
- Conventional Commits with package scopes:
  `feat(elements): add Timoshenko beam with selective integration`.
- Every change lands by PR against the checklist: CI green (format, lint,
  types, import contracts, full test pyramid); verification evidence for
  physics changes; docs + CHANGELOG updated; ADR added/amended if an
  architectural decision was made; research manifest updated if `research/`
  was touched.

## Where things go

Consult the package table in `docs/source/package_guide.md` and each
package's `__init__` docstring ("must NOT do" lists included). When in
doubt: numerics is mechanics-free (R1), physics is discretization-free (R2),
io and symbolics are thin edges (R3).

## Plugins

Third-party elements, theories, constitutive laws, kernels, and solvers
register through entry points without forking — see `templates/plugin/` and
SDS Section 12. The conformance kit (arrives with phase 0.5) certifies
plugin contract compliance.
