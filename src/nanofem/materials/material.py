"""Material: validated, immutable property record (ADR-004, SDS Section 6).

Zero constitutive mathematics. Bounds validate physics, not habit: auxetic
nu < 0 is legal within (-1, 0.5); surface moduli may be negative
(Miller-Shenoy). The E/nu/G redundancy check is an SDS Section 6 mandated
consistency rule, not a constitutive equation (dev note N-10).
"""

from __future__ import annotations

from collections.abc import Callable

from nanofem.utils.exceptions import InputValidationError
from nanofem.utils.validation import (
    require_finite,
    require_identifier,
    require_in_open_interval,
    require_non_negative,
    require_positive,
)

_BOUNDS: dict[str, Callable[[float, str], None]] = {
    "E": require_positive,
    "G": require_positive,
    "nu": lambda v, n: require_in_open_interval(v, -1.0, 0.5, n),
    "rho": require_non_negative,
    "alpha_thermal": require_finite,
    "eta_damping": require_non_negative,
    "e0a": require_non_negative,  # 0 = local limit, deliberately legal (SDS Section 6)
    "l_sg": require_non_negative,
    "l_cs": require_non_negative,
    "mu_s": require_finite,  # may be negative (Miller-Shenoy)
    "lambda_s": require_finite,
    "tau0": require_finite,
    "rho_s": require_non_negative,
}

_G_CONSISTENCY_RTOL = 1e-9


class Material:
    """Typed property store; unknown keys must be namespaced ('group.key')."""

    def __init__(self, name: str, **properties: float) -> None:
        require_identifier(name, "material name")
        if not properties:
            raise InputValidationError(f"material '{name}' defines no properties")
        self._name = name
        validated: dict[str, float] = {}
        for key, value in properties.items():
            check = _BOUNDS.get(key)
            if check is not None:
                check(value, f"material '{name}': {key}")
            elif "." in key:
                require_finite(value, f"material '{name}': {key}")
            else:
                raise InputValidationError(
                    f"material '{name}': unknown property '{key}'; canonical keys are "
                    f"{sorted(_BOUNDS)}; user keys must be namespaced 'group.key'"
                )
            validated[key] = float(value)
        if {"E", "nu", "G"} <= validated.keys():
            expected = validated["E"] / (2.0 * (1.0 + validated["nu"]))
            if abs(validated["G"] - expected) > _G_CONSISTENCY_RTOL * abs(expected):
                raise InputValidationError(
                    f"material '{name}': G={validated['G']} inconsistent with "
                    f"E/(2(1+nu))={expected} (SDS Section 6 redundancy check)"
                )
        self._properties = validated

    @property
    def name(self) -> str:
        """Material name (registry key inside a Model)."""
        return self._name

    def defines(self, key: str) -> bool:
        """Whether a property key is defined."""
        return key in self._properties

    @property
    def keys(self) -> tuple[str, ...]:
        """Defined property keys, sorted."""
        return tuple(sorted(self._properties))

    def value(self, key: str) -> float:
        """Constant property value; misses list what is defined."""
        try:
            return self._properties[key]
        except KeyError:
            raise InputValidationError(
                f"material '{self._name}' does not define '{key}'; defined: {list(self.keys)}"
            ) from None

    def value_at(self, key: str, temperature: float) -> float:
        """Temperature-dependent lookup placeholder (SDS Section 6)."""
        raise NotImplementedError("TODO(future): temperature-dependent property tables")

    def to_dict(self) -> dict[str, object]:
        """JSON-safe record."""
        return {"name": self._name, "properties": dict(sorted(self._properties.items()))}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Material:
        """Rebuild from a to_dict record."""
        props = data["properties"]
        assert isinstance(props, dict)
        return cls(str(data["name"]), **{str(k): float(v) for k, v in props.items()})
