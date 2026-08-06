"""Phase-0 import tests: every package and module imports cleanly.

Success criterion: the architecture is *importable* end to end, with zero
numerical implementation behind it.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import nanofem

PACKAGES: tuple[str, ...] = (
    "nanofem",
    "nanofem.core",
    "nanofem.mesh",
    "nanofem.geometry",
    "nanofem.materials",
    "nanofem.physics",
    "nanofem.physics.elasticity",
    "nanofem.physics.eringen",
    "nanofem.physics.strain_gradient",
    "nanofem.physics.couple_stress",
    "nanofem.physics.surface",
    "nanofem.physics.piezoelectric",
    "nanofem.physics.thermoelastic",
    "nanofem.kernels",
    "nanofem.state",
    "nanofem.elements",
    "nanofem.elements.structural",
    "nanofem.elements.continuum",
    "nanofem.constraints",
    "nanofem.numerics",
    "nanofem.numerics.reference",
    "nanofem.numerics.interpolation",
    "nanofem.numerics.quadrature",
    "nanofem.numerics.mapping",
    "nanofem.numerics.operators",
    "nanofem.numerics.tensors",
    "nanofem.numerics.assembly",
    "nanofem.numerics.linalg",
    "nanofem.numerics.timeintegration",
    "nanofem.numerics.search",
    "nanofem.numerics.math",
    "nanofem.analysis",
    "nanofem.analysis.optimization",
    "nanofem.postprocess",
    "nanofem.io",
    "nanofem.symbolics",
    "nanofem.utils",
)


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    """Every declared package imports."""
    module = importlib.import_module(name)
    assert module.__name__ == name


def test_every_module_imports() -> None:
    """Walk the installed tree: every module under nanofem imports."""
    failures: list[str] = []
    for info in pkgutil.walk_packages(nanofem.__path__, prefix="nanofem."):
        try:
            importlib.import_module(info.name)
        except Exception as exc:  # noqa: BLE001 - collecting all failures for the report
            failures.append(f"{info.name}: {exc!r}")
    assert not failures, "modules failed to import:\n" + "\n".join(failures)


def test_version_is_declared() -> None:
    """The top-level package declares its version."""
    assert isinstance(nanofem.__version__, str)
    assert nanofem.__version__


def test_public_api_is_curated_and_importable() -> None:
    """Phase 1: every name in __all__ resolves; the surface is the object model."""
    assert "__version__" in nanofem.__all__
    for name in nanofem.__all__:
        assert hasattr(nanofem, name), f"__all__ names missing attribute: {name}"
    for expected in ("Model", "Mesh", "Material", "FieldSpec", "DofHandler"):
        assert expected in nanofem.__all__
