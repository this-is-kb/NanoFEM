"""Phase-0 integrity tests: docstrings, vocabularies, and the exception tree.

These check that the SDS *contracts encoded as declarations* are actually
present: every module documents itself, the contribution vocabulary matches
SDS Sections 10-11, and every error derives from NanoFEMError.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import nanofem
from nanofem.numerics.assembly.contributions import (
    ContributionKind,
    ContributionProvider,
    OperatorRole,
)
from nanofem.utils import exceptions


def _all_modules() -> list[str]:
    names = ["nanofem"]
    names += [i.name for i in pkgutil.walk_packages(nanofem.__path__, prefix="nanofem.")]
    return names


def test_every_module_has_docstring() -> None:
    """Every package and module carries a docstring (self-documenting skeleton)."""
    undocumented = [n for n in _all_modules() if not importlib.import_module(n).__doc__]
    assert not undocumented, f"missing docstrings: {undocumented}"


def test_every_package_states_responsibilities_and_todo() -> None:
    """Package __init__ docstrings state Responsibilities and TODO sections."""
    bad: list[str] = []
    for name in _all_modules():
        module = importlib.import_module(name)
        if module.__name__ != getattr(module, "__package__", None):
            continue  # only packages
        doc = module.__doc__ or ""
        if "Responsibilities" not in doc or "TODO" not in doc:
            bad.append(name)
    assert not bad, f"package docstrings missing required sections: {bad}"


def test_contribution_kinds_match_sds_section_10() -> None:
    """Kinds are exactly {CELL, FACET, PAIR, EDGE, VERTEX}."""
    assert {k.name for k in ContributionKind} == {"CELL", "FACET", "PAIR", "EDGE", "VERTEX"}


def test_operator_roles_cover_sds_section_11() -> None:
    """The role vocabulary covers the SDS operator set."""
    expected = {
        "STIFFNESS",
        "MASS",
        "DAMPING",
        "GEOMETRIC_STIFFNESS",
        "COUPLING",
        "FORCE",
        "SENSITIVITY",
    }
    assert {r.name for r in OperatorRole} == expected


def test_provider_protocol_is_runtime_checkable() -> None:
    """The assembly currency is a runtime-checkable protocol with contributions()."""

    class Fake:
        def contributions(self, role: OperatorRole) -> object:
            return iter(())

    assert isinstance(Fake(), ContributionProvider)


def test_exception_tree_roots_at_nanofemerror() -> None:
    """Every public exception derives from NanoFEMError (single catch clause)."""
    strays: list[str] = []
    for name, obj in inspect.getmembers(exceptions, inspect.isclass):
        if obj.__module__ != exceptions.__name__:
            continue
        if obj is exceptions.NanoFEMError:
            continue
        if not issubclass(obj, exceptions.NanoFEMError):
            strays.append(name)
    assert not strays, f"exceptions outside the tree: {strays}"


def test_symbolics_is_importable_but_only_by_tests() -> None:
    """Tests MAY import symbolics; runtime MUST NOT (enforced by import-linter)."""
    module = importlib.import_module("nanofem.symbolics")
    assert "NEVER imported by runtime" in (module.__doc__ or "")
