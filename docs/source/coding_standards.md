# Coding standards

Normative source: ARCHITECTURE v1 Sections 7-8 (standards, naming) plus SDS
Section 0.3 conventions C-1..C-8. Mechanically enforced here by black
(format, line 100), isort (imports, black profile), ruff (lint), mypy
--strict (types; arrays as `NDArray[np.float64]` with shapes documented in
docstrings), and import-linter (layering). NumPy-style docstrings are
mandatory on public objects; kernel modules open with a symbol table; no
`print`, loggers only; no module-level mutable state; float64/int64 per C-4;
determinism per C-5.
