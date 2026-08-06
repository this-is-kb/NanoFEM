"""The operator registry: name -> ``DiscreteOperator`` class (SDS Section 8).

Excludes ``nonlocal_integral``, the declared placeholder in ``future.py`` -
mirroring how ``AVAILABLE_QUADRATURES`` excludes ``GaussJacobiQuadrature`` and
friends. A registry entry is a promise the class can be instantiated and
verified today.
"""

from __future__ import annotations

from nanofem.numerics.operators.base import DiscreteOperator
from nanofem.numerics.operators.curl import CurlOperator
from nanofem.numerics.operators.divergence import DivergenceOperator
from nanofem.numerics.operators.gradient import GradientOperator
from nanofem.numerics.operators.helmholtz import HelmholtzOperator
from nanofem.numerics.operators.laplacian import LaplacianOperator
from nanofem.numerics.operators.second_gradient import SecondGradientOperator
from nanofem.numerics.operators.surface_gradient import SurfaceGradientOperator
from nanofem.numerics.operators.symmetric_gradient import SymmetricGradientOperator
from nanofem.numerics.operators.transformation import TransformationOperator
from nanofem.numerics.operators.voigt_map import VoigtMapOperator

#: Every SDS Section 8 operator with a real recipe, keyed by its ``OPERATOR_CATALOG`` name.
OPERATOR_REGISTRY: dict[str, type[DiscreteOperator]] = {
    "gradient": GradientOperator,
    "symmetric_gradient": SymmetricGradientOperator,
    "divergence": DivergenceOperator,
    "curl": CurlOperator,
    "laplacian": LaplacianOperator,
    "helmholtz": HelmholtzOperator,
    "surface_gradient": SurfaceGradientOperator,
    "second_gradient": SecondGradientOperator,
    "voigt_map": VoigtMapOperator,
    "transformation": TransformationOperator,
}


def available_operators() -> tuple[str, ...]:
    """The names of every operator with a real recipe, in registry order."""
    return tuple(OPERATOR_REGISTRY)
