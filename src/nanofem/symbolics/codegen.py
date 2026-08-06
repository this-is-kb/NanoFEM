"""Emit derived tables as plain numpy source with provenance headers (ADR-013).

The runtime never depends on sympy: generated code is committed with a header
naming the generating script and its inputs.
"""

from __future__ import annotations


class CodeGenerator:
    """Derivation -> committed source, provenance-stamped."""
