"""Per-operator ``verify()``, registry-level checks, the future placeholder, and independence.

Trip tests here subclass or feed malformed data directly to a real operator's
``verify()``/recipe function, matching the class-based trip-test pattern
every prior numerics phase used (unlike ``tensors``, these ARE real classes).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nanofem.numerics.operators import (
    OPERATOR_CATALOG,
    OPERATOR_REGISTRY,
    NonlocalIntegralOperator,
    available_operators,
    is_operator_library_valid,
    verify_operator_library,
)
from nanofem.numerics.operators.base import Continuity, derived_continuity
from nanofem.numerics.operators.verification import (
    verify_continuity_derivation,
    verify_cross_operator_consistency,
    verify_registry_self_consistency,
)

# ---- the aggregate suite and every registered class passes ----------------------


def test_verify_operator_library_passes() -> None:
    verify_operator_library()
    assert is_operator_library_valid()


@pytest.mark.parametrize("name", list(OPERATOR_REGISTRY))
def test_each_registered_operator_verifies_itself(name: str) -> None:
    operator = OPERATOR_REGISTRY[name]()
    operator.verify()
    assert operator.is_valid()


def test_registry_covers_every_catalog_entry_except_the_future_placeholder() -> None:
    assert set(available_operators()) == OPERATOR_CATALOG - {"nonlocal_integral"}


def test_verify_registry_self_consistency_and_continuity_pass() -> None:
    verify_registry_self_consistency()
    verify_continuity_derivation()
    verify_cross_operator_consistency()


# ---- derivative-order / continuity cross-checks ----------------------------------


def test_second_gradient_forces_c1_continuity() -> None:
    assert derived_continuity(("second_gradient",)) is Continuity.C1


def test_first_derivative_operators_stay_c0() -> None:
    assert derived_continuity(("gradient", "symmetric_gradient", "divergence")) is Continuity.C0


# ---- the future placeholder -------------------------------------------------------


def test_nonlocal_integral_operator_refuses_construction_and_names_its_blocker() -> None:
    with pytest.raises(NotImplementedError, match="declared placeholder"):
        NonlocalIntegralOperator()


def test_nonlocal_integral_is_not_in_the_registry() -> None:
    assert "nonlocal_integral" not in OPERATOR_REGISTRY
    assert "nonlocal_integral" in OPERATOR_CATALOG  # still a real catalog entry


# ---- leaf-of-numerics independence proof ------------------------------------------


def test_operators_does_not_import_mechanics_or_undeclared_numerics_packages() -> None:
    """The subprocess+source-scan proof every numerics leaf carries (SDS rule R1).

    ``nanofem/__init__.py`` itself eagerly imports mesh/materials/physics/
    constraints/analysis as its curated public API, so any import under
    ``nanofem.*`` pulls those packages into ``sys.modules`` regardless of what
    ``operators`` itself imports - that is a fact about the top-level package,
    not a leak from this layer. The forbidden list here checks what the
    quadrature/mapping precedents checked: the sibling numerics leaves this
    package must not import directly.
    """
    script = (
        "import sys;"
        "from nanofem.numerics.operators import verify_operator_library;"
        "verify_operator_library();"
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


def test_operators_knows_nothing_about_assembly_or_meshes() -> None:
    import nanofem.numerics.operators as package

    root = Path(package.__file__).parent
    forbidden = ("DofHandler", "assembl", "SparsityPattern", "Material", "ShapeFunctionFamily")
    for source in root.glob("*.py"):
        text = source.read_text()
        code = "\n".join(line for line in text.split("\n") if not line.strip().startswith("#"))
        body = code.split('"""')
        executable = "".join(body[::2])
        for term in forbidden:
            assert term not in executable, f"{source.name} references {term!r}"


# ---- shape/error trip tests -------------------------------------------------------


def test_helmholtz_rejects_a_negative_length_scale() -> None:
    from nanofem.numerics.operators import helmholtz_matrix
    from nanofem.utils.exceptions import InputValidationError

    values = np.zeros((1, 2))
    gradients = np.zeros((1, 2, 2))
    weights = np.array([1.0])
    volume_scale = np.array([1.0])
    with pytest.raises(InputValidationError):
        helmholtz_matrix(values, gradients, weights, volume_scale, -0.1)


def test_second_gradient_rejects_a_non_square_hessian_block() -> None:
    from nanofem.numerics.operators import OperatorShapeError, second_gradient_tensor

    with pytest.raises(OperatorShapeError):
        second_gradient_tensor(np.zeros((2, 3, 2, 3)))
