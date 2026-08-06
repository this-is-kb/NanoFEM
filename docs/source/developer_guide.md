# Developer guide

## Setup

```
pip install -e ".[dev]"
pre-commit install
```

## The verification loop

Run exactly what CI runs: `black --check src tests`, `isort --check-only src
tests`, `ruff check src tests`, `mypy`, `lint-imports`, `pytest`.

## Placeholder conventions (phase 0)

- Value objects are frozen dataclasses with declared fields (declarations are
  architecture, not implementation).
- Interfaces are ABCs/Protocols with docstring-only abstract methods.
- Concrete stubs raise `NotImplementedError("TODO(phase-N): ...")` — not bare
  `pass` — because mypy --strict requires annotated functions to satisfy
  their return type, and a raise does. See `docs/dev/notes.md`, N-1.
- Enums encode the SDS vocabularies (kinds, roles, locality, continuity):
  vocabulary is contract, not implementation.

## Where new code goes

Each package `__init__` states purpose, responsibilities, future modules,
and TODOs; the "must NOT do" column lives in `package_guide.md`. If your
import would violate a layer, `lint-imports` will tell you before review
does.
