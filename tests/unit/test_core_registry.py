"""Unit tests for Registry (SDS Section 12 plugin lookup)."""

from __future__ import annotations

import pytest

from nanofem.core.registry import Registry
from nanofem.utils.exceptions import InputValidationError


class _Dummy:
    """A trivial class to register."""


def test_register_and_get_round_trip() -> None:
    registry = Registry()
    registry.register("group.name", _Dummy, builtin=True)
    assert registry.get("group.name") is _Dummy
    assert "group.name" in registry
    assert registry.keys() == ("group.name",)


def test_keys_are_sorted_deterministically() -> None:
    registry = Registry()
    registry.register("b.two", _Dummy, builtin=True)
    registry.register("a.one", _Dummy, builtin=True)
    assert registry.keys() == ("a.one", "b.two")


def test_rejects_non_namespaced_key() -> None:
    registry = Registry()
    with pytest.raises(InputValidationError, match="namespaced"):
        registry.register("bare", _Dummy, builtin=True)


def test_reserved_prefix_requires_builtin_flag() -> None:
    registry = Registry(reserved_prefix="nanofem")
    with pytest.raises(InputValidationError, match="reserved"):
        registry.register("nanofem.bar", _Dummy)
    registry.register("nanofem.bar", _Dummy, builtin=True)  # allowed with the flag


def test_duplicate_registration_raises() -> None:
    registry = Registry()
    registry.register("group.name", _Dummy, builtin=True)
    with pytest.raises(InputValidationError, match="already registered"):
        registry.register("group.name", _Dummy, builtin=True)


def test_unknown_key_lookup_lists_registered() -> None:
    registry = Registry()
    registry.register("group.name", _Dummy, builtin=True)
    with pytest.raises(InputValidationError, match="group.name"):
        registry.get("group.missing")
