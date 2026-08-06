"""nanofem.numerics.operators.

Discrete operator library: stateless recipes from tabulated data (SDS Section 8, ADR-015).

Responsibilities
----------------
- Gradient family, Helmholtz, nonlocal pairing, surface gradient, second gradient, Voigt,
  transforms
- Each operator declares its highest interpolation derivative -> theories derive continuity

Future modules
--------------
- base.py (implemented: Continuity, OPERATOR_CATALOG, derived_continuity, DiscreteOperator)
- errors.py (implemented: OperatorError, OperatorShapeError, UnsupportedDimensionError)
- gradient.py, divergence.py, curl.py, symmetric_gradient.py, laplacian.py, helmholtz.py,
  surface_gradient.py, second_gradient.py, voigt_map.py, transformation.py (all implemented)
- future.py (implemented: NonlocalIntegralOperator - a declared, blocked placeholder;
  see its docstring for what it needs before it can be a real recipe)
- registry.py, verification.py (implemented)

TODO
----
- TODO: nonlocal_integral becomes a real recipe once kernels/ and numerics/search/ exist
"""

from __future__ import annotations

from nanofem.numerics.operators.base import (
    OPERATOR_CATALOG,
    OPERATOR_DERIVATIVE_ORDER,
    Continuity,
    DiscreteOperator,
    derived_continuity,
)
from nanofem.numerics.operators.curl import CurlOperator, curl_matrix
from nanofem.numerics.operators.divergence import DivergenceOperator, divergence_matrix
from nanofem.numerics.operators.errors import (
    OperatorError,
    OperatorShapeError,
    UnsupportedDimensionError,
)
from nanofem.numerics.operators.future import NonlocalIntegralOperator
from nanofem.numerics.operators.gradient import GradientOperator, gradient_matrix
from nanofem.numerics.operators.helmholtz import HelmholtzOperator, helmholtz_matrix, mass_term
from nanofem.numerics.operators.laplacian import LaplacianOperator, laplacian_matrix
from nanofem.numerics.operators.registry import OPERATOR_REGISTRY, available_operators
from nanofem.numerics.operators.second_gradient import (
    SecondGradientOperator,
    second_gradient_tensor,
)
from nanofem.numerics.operators.surface_gradient import (
    SurfaceGradientOperator,
    surface_gradient_matrix,
    surface_projector,
)
from nanofem.numerics.operators.symmetric_gradient import (
    SymmetricGradientOperator,
    symmetric_gradient_matrix,
)
from nanofem.numerics.operators.transformation import (
    TransformationKind,
    TransformationOperator,
    tensor_transformation_matrix_voigt,
    vector_transformation_matrix,
)
from nanofem.numerics.operators.verification import (
    is_operator_library_valid,
    verify_operator_library,
)
from nanofem.numerics.operators.voigt_map import (
    VoigtMapOperator,
    deviatoric_projector_voigt,
    identity_vector,
    trace_row_voigt,
)

__all__ = [
    # base vocabulary
    "Continuity",
    "OPERATOR_CATALOG",
    "OPERATOR_DERIVATIVE_ORDER",
    "derived_continuity",
    "DiscreteOperator",
    # errors
    "OperatorError",
    "OperatorShapeError",
    "UnsupportedDimensionError",
    # recipes (functions)
    "gradient_matrix",
    "divergence_matrix",
    "curl_matrix",
    "symmetric_gradient_matrix",
    "laplacian_matrix",
    "helmholtz_matrix",
    "mass_term",
    "surface_gradient_matrix",
    "surface_projector",
    "second_gradient_tensor",
    "identity_vector",
    "deviatoric_projector_voigt",
    "trace_row_voigt",
    "vector_transformation_matrix",
    "tensor_transformation_matrix_voigt",
    "TransformationKind",
    # recipes (DiscreteOperator classes)
    "GradientOperator",
    "DivergenceOperator",
    "CurlOperator",
    "SymmetricGradientOperator",
    "LaplacianOperator",
    "HelmholtzOperator",
    "SurfaceGradientOperator",
    "SecondGradientOperator",
    "VoigtMapOperator",
    "TransformationOperator",
    # declared placeholder
    "NonlocalIntegralOperator",
    # registry
    "OPERATOR_REGISTRY",
    "available_operators",
    # verification
    "verify_operator_library",
    "is_operator_library_valid",
]
