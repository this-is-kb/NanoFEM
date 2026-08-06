"""Independent symbolic reference cells (ADR-013).

Deliberate duplication of numerics/reference: a verification path that
reuses the implementation under test verifies nothing. sympy is imported
lazily inside functions only; a runtime import of this package is a CI
failure enforced by import-linter.
"""

from __future__ import annotations


class SymbolicReferenceCell:
    """Symbolic cell definitions sharing no code with numerics."""
