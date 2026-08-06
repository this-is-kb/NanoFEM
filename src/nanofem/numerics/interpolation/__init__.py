"""nanofem.numerics.interpolation.

The interpolation framework: finite elements described in the classical
Ciarlet sense as a triple (reference domain, polynomial space, degrees of
freedom), together with everything derivable from that triple *without*
forming a basis.

This package holds both sides of a phase boundary, and the boundary has a
precise algebraic statement. The generalized Vandermonde
``M[k, j] = l_k(m_j)`` pairs every DOF functional with every monomial of the
space. ``Interpolation`` builds ``M`` and proves it invertible - which is
exactly what unisolvence, the Kronecker property, and linear independence all
reduce to. ``M^-1`` is the coefficient matrix of the shape functions in the
monomial basis; ``ShapeFunctions`` (a later phase) inverts it and tabulates
the result. Nothing here evaluates or differentiates a shape function.

Responsibilities
----------------
- Polynomial spaces (``P_k``, ``Q_k``) as monomial exponent sets, with the
  order / completeness-degree / maximum-total-degree distinction kept honest
- Degrees of freedom as functionals; interpolation nodes with entity
  associations derived from the reference element (SDS C-3)
- The Lagrange (C0) and Hermite (C1) families on line, triangle, and
  quadrilateral; declared placeholders for serendipity, hierarchical, and
  spectral
- Verification: polynomial degree and completeness, linear independence,
  Kronecker duality, constant reproduction, node ordering

TODO
----
- TODO(phase-4): construct the nodal basis by inverting the unisolvence
  matrix, and implement ``ShapeFunctions.evaluate`` / ``derivatives``
- TODO(phase-4+): the serendipity, hierarchical, and spectral families
  (see ``future.py`` for what each still needs)
"""

from __future__ import annotations

from nanofem.numerics.interpolation.base import Interpolation, ShapeFunctions
from nanofem.numerics.interpolation.dofs import DofFunctional, InterpolationNode
from nanofem.numerics.interpolation.enums import (
    DofKind,
    InterpolationFamily,
    PolynomialSpaceType,
)
from nanofem.numerics.interpolation.errors import (
    InterpolationError,
    NodeOrderingError,
    PolynomialSpaceError,
    UnisolvenceError,
)
from nanofem.numerics.interpolation.future import (
    HierarchicalInterpolation,
    SerendipityInterpolation,
    SpectralInterpolation,
)
from nanofem.numerics.interpolation.hermite import HermiteInterpolation
from nanofem.numerics.interpolation.lagrange import LagrangeInterpolation
from nanofem.numerics.interpolation.polynomial import PolynomialSpace, monomial_label
from nanofem.numerics.interpolation.registry import (
    AVAILABLE_INTERPOLATIONS,
    INTERPOLATION_FAMILIES,
    available_interpolations,
    interpolation,
    interpolation_from_dict,
)
from nanofem.numerics.interpolation.shape_functions import (
    SHAPE_FUNCTION_FAMILIES,
    HermiteShapeFunctions,
    LagrangeShapeFunctions,
    ShapeFunctionError,
    ShapeFunctionFamily,
    shape_functions,
)
from nanofem.numerics.interpolation.tabulation import (
    PointsLike,
    Tabulation,
    as_points,
    monomial_table,
)

__all__ = [
    # enumerations
    "InterpolationFamily",
    "DofKind",
    "PolynomialSpaceType",
    # errors
    "InterpolationError",
    "PolynomialSpaceError",
    "UnisolvenceError",
    "NodeOrderingError",
    # building blocks
    "PolynomialSpace",
    "monomial_label",
    "InterpolationNode",
    "DofFunctional",
    # contracts
    "Interpolation",
    # implemented families
    "LagrangeInterpolation",
    "HermiteInterpolation",
    # declared placeholders
    "SerendipityInterpolation",
    "HierarchicalInterpolation",
    "SpectralInterpolation",
    # registry
    "INTERPOLATION_FAMILIES",
    "AVAILABLE_INTERPOLATIONS",
    "interpolation",
    "available_interpolations",
    "interpolation_from_dict",
    # shape functions
    "ShapeFunctions",
    "ShapeFunctionFamily",
    "LagrangeShapeFunctions",
    "HermiteShapeFunctions",
    "ShapeFunctionError",
    "SHAPE_FUNCTION_FAMILIES",
    "shape_functions",
    "Tabulation",
    "PointsLike",
    "monomial_table",
    "as_points",
]
