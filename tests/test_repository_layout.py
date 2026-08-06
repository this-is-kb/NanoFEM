"""Phase-0 layout tests: the repository tree matches the frozen SDS.

The SDS folder hierarchy becomes an executable assertion. These tests assume
a repository checkout (they locate the root relative to this file).
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DIRS: tuple[str, ...] = (
    ".github/workflows",
    "docs/source/tutorials",
    "docs/source/howto",
    "docs/source/theory",
    "docs/source/api",
    "docs/source/adr",
    "docs/design",
    "docs/dev",
    "examples",
    "research/validation/beam",
    "research/validation/frame",
    "research/validation/plane",
    "research/validation/plate",
    "research/validation/shell",
    "research/validation/nonlocal",
    "research/validation/gradient",
    "research/validation/auxetic",
    "research/validation/multiphysics",
    "research/benchmarks",
    "research/papers",
    "templates/plugin/src/nanofem_example_plugin",
    "tests/unit",
    "tests/symbolic",
    "tests/element",
    "tests/verification",
    "tests/convergence",
    "tests/regression",
    "src/nanofem/numerics/operators",
    "src/nanofem/numerics/tensors",
    "src/nanofem/physics/eringen",
)

EXPECTED_FILES: tuple[str, ...] = (
    "pyproject.toml",
    "setup.cfg",
    "requirements.txt",
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "CITATION.cff",
    ".gitignore",
    ".pre-commit-config.yaml",
    ".github/workflows/ci.yml",
    ".github/workflows/docs.yml",
    ".github/workflows/research-nightly.yml",
    "docs/design/ARCHITECTURE.md",
    "docs/design/ARCHITECTURE_v2.md",
    "docs/design/NanoFEM_SDS.md",
    "docs/design/REFERENCE_ELEMENTS.md",
    "docs/design/INTERPOLATION.md",
    "docs/design/SHAPE_FUNCTIONS.md",
    "docs/design/GEOMETRIC_MAPPING.md",
    "docs/design/QUADRATURE.md",
    "docs/dev/notes.md",
    "src/nanofem/py.typed",
)


@pytest.mark.parametrize("rel", EXPECTED_DIRS)
def test_expected_directory_exists(rel: str) -> None:
    """Each SDS-mandated directory exists."""
    assert (ROOT / rel).is_dir(), f"missing directory: {rel}"


@pytest.mark.parametrize("rel", EXPECTED_FILES)
def test_expected_file_exists(rel: str) -> None:
    """Each phase-0 deliverable file exists."""
    assert (ROOT / rel).is_file(), f"missing file: {rel}"


def test_license_is_mit() -> None:
    """Phase-0 requirement: MIT license."""
    assert "MIT License" in (ROOT / "LICENSE").read_text(encoding="utf-8")


def test_declared_versions_agree() -> None:
    """pyproject.toml and nanofem.__version__ must not drift apart (dev note N-15).

    They had, silently, until phase 2. This makes the check mechanical.
    """
    import tomllib

    import nanofem

    with (ROOT / "pyproject.toml").open("rb") as handle:
        declared = str(tomllib.load(handle)["project"]["version"])
    assert (
        declared == nanofem.__version__
    ), f"pyproject.toml declares {declared} but nanofem.__version__ is {nanofem.__version__}"
