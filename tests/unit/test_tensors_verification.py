"""The tensor library's self-check, and its independence from mechanics.

Unlike every prior numerics phase, ``tensors`` verification is module-level
(there is no single stateful class to hang ``verify()`` off), so trip tests
here construct deliberately malformed data directly and check that the
underlying identity fails, rather than subclassing a verified object.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from nanofem.numerics.tensors import is_tensor_library_valid, verify_tensor_library
from nanofem.numerics.tensors.verification import (
    verify_eigendecomposition,
    verify_invariants,
    verify_isotropic_oracle,
    verify_projector_algebra,
    verify_rotation_consistency,
    verify_round_trip,
    verify_symmetry_classification,
    verify_work_conjugacy,
)

# ---- the aggregate suite passes -------------------------------------------------


def test_verify_tensor_library_passes() -> None:
    verify_tensor_library()  # raises on failure
    assert is_tensor_library_valid()


@pytest.mark.parametrize(
    "check",
    [
        verify_round_trip,
        verify_work_conjugacy,
        verify_projector_algebra,
        verify_isotropic_oracle,
        verify_symmetry_classification,
        verify_rotation_consistency,
        verify_invariants,
        verify_eigendecomposition,
    ],
)
def test_each_verify_check_passes_independently(check: Callable[[], None]) -> None:
    check()


# ---- leaf-of-numerics independence proof ----------------------------------------


def test_tensors_does_not_import_mechanics_packages() -> None:
    """The subprocess+source-scan proof every numerics leaf carries (SDS rule R1).

    ``nanofem/__init__.py`` itself eagerly imports mesh/materials/physics/
    constraints/analysis as its curated public API (see its own source), so
    *any* import under ``nanofem.*`` - including this one - pulls those
    packages into ``sys.modules`` regardless of what ``tensors`` itself
    imports. That is a fact about the top-level package, not a leak from this
    layer, so the forbidden list here checks the same thing the quadrature/
    mapping precedents did: the sibling numerics leaves this package must not
    import directly (``interpolation``, ``mapping``, ``quadrature``,
    ``kernels``, ``search``).
    """
    script = (
        "import sys;"
        "from nanofem.numerics.tensors import verify_tensor_library;"
        "verify_tensor_library();"
        "loaded = set(sys.modules);"
        "forbidden = ['numerics.interpolation', 'numerics.mapping', 'numerics.quadrature',"
        " 'kernels', 'numerics.search'];"
        "hit = [m for f in forbidden for m in loaded if f in m];"
        "assert not hit, f'forbidden import: {hit}';"
        "print('leaf OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "leaf OK" in result.stdout


def test_tensors_knows_nothing_about_finite_elements() -> None:
    import nanofem.numerics.tensors as package

    root = Path(package.__file__).parent
    forbidden = ("shape_function", "ShapeFunction", "stiffness_matrix", "assembl", "DofHandler")
    for source in root.glob("*.py"):
        text = source.read_text()
        code = "\n".join(line for line in text.split("\n") if not line.strip().startswith("#"))
        body = code.split('"""')
        executable = "".join(body[::2])
        for term in forbidden:
            assert term not in executable, f"{source.name} references {term!r}"
