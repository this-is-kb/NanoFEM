"""String-keyed class registry: the plugin mechanism (SDS Section 12).

Keys are namespaced 'group.name'; duplicate registration is an error, never
a silent override; the 'nanofem.' namespace is reserved for built-ins.
"""

from __future__ import annotations

import re

from nanofem.utils.exceptions import InputValidationError

_KEY_PATTERN = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")


class Registry:
    """Registration and lookup for elements, theories, constitutive laws, kernels, solvers."""

    def __init__(self, reserved_prefix: str = "nanofem") -> None:
        self._classes: dict[str, type[object]] = {}
        self._reserved = reserved_prefix

    def register(self, key: str, cls: type[object], builtin: bool = False) -> None:
        """Register a class; namespaced keys only; collisions raise (SDS Section 12)."""
        if not _KEY_PATTERN.match(key):
            raise InputValidationError(
                f"registry key {key!r} must be namespaced lowercase 'group.name'"
            )
        if key.split(".", 1)[0] == self._reserved and not builtin:
            raise InputValidationError(f"namespace '{self._reserved}.' is reserved for built-ins")
        if key in self._classes:
            raise InputValidationError(f"registry key {key!r} is already registered")
        self._classes[key] = cls

    def get(self, key: str) -> type[object]:
        """Look up a class; unknown keys name what is registered."""
        try:
            return self._classes[key]
        except KeyError:
            known = ", ".join(self.keys()) or "(none)"
            raise InputValidationError(
                f"unknown registry key {key!r}; registered: {known}"
            ) from None

    def keys(self) -> tuple[str, ...]:
        """All registered keys, sorted (determinism, SDS C-5)."""
        return tuple(sorted(self._classes))

    def __contains__(self, key: str) -> bool:
        return key in self._classes
