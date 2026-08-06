"""nanofem.numerics.quadrature.

Integration rules with declared exactness on reference cells (SDS 2.5).

A quadrature rule is a finite set of points and weights on a reference domain
such that ``sum_q w_q f(x_q)`` approximates ``integral f``, exactly for every
polynomial up to a declared degree. That is all a rule is, and all it knows: it
has no notion of a shape function, a finite element, a constitutive model, or an
assembly. It does not multiply weights by a Jacobian and it does not integrate
over a physical element - those compose a rule with a mapping, which is the
element layer's job.

A leaf of the numerics layer, deliberately
------------------------------------------
This package imports the reference domains and nothing else from the library.
The independence is structural rather than stylistic: phase 3 recorded that the
spectral interpolation family cannot precede quadrature, because its nodes *are*
Gauss-Lobatto-Legendre points. When that family lands, ``numerics.interpolation``
will import ``numerics.quadrature``, so the dependency must not already run the
other way. Monomial evaluation and the closed-form integrals used to verify
exactness are therefore written here rather than borrowed.

Responsibilities
----------------
- Gauss families; rule selection by cell and order; explicit reduced rules only
- Gauss-Legendre and Gauss-Lobatto on the line; their tensor products on the
  square; Dunavant's symmetric rules on the triangle
- Integration of scalar functions, moments, measure and centroid
- Exactness against analytic closed forms, and maximality of the declared degree
- Weight normalization and positivity; symmetry orbits derived from the points

TODO
----
- TODO(phase-6+): Dunavant degrees 6-20 (needs the six-fold orbit)
- TODO(phase-6+): the Gauss-Jacobi, adaptive, and sparse-grid families in
  ``future.py``
- TODO(phase-7): facet rules, once facet mappings exist
"""

from __future__ import annotations

from nanofem.numerics.quadrature.dunavant import SUPPORTED_DEGREES, DunavantQuadrature
from nanofem.numerics.quadrature.enums import QuadratureFamily
from nanofem.numerics.quadrature.errors import (
    ExactnessError,
    QuadratureError,
    SymmetryError,
    WeightError,
)
from nanofem.numerics.quadrature.factory import (
    AVAILABLE_QUADRATURES,
    DEFAULT_FAMILIES,
    QuadratureFactory,
    available_quadratures,
    quadrature,
)
from nanofem.numerics.quadrature.future import (
    AdaptiveQuadrature,
    GaussJacobiQuadrature,
    SparseGridQuadrature,
)
from nanofem.numerics.quadrature.gauss import (
    GaussLegendreQuadrature,
    GaussLobattoQuadrature,
)
from nanofem.numerics.quadrature.moments import (
    exact_monomial_integral,
    monomial_exponents,
    monomial_exponents_up_to,
    monomial_values,
)
from nanofem.numerics.quadrature.rules import QuadratureRule
from nanofem.numerics.quadrature.symmetry import AffineSymmetry, symmetry_group
from nanofem.numerics.quadrature.tensor import TensorProductQuadrature

__all__ = [
    # enumeration
    "QuadratureFamily",
    # errors
    "QuadratureError",
    "ExactnessError",
    "WeightError",
    "SymmetryError",
    # contract
    "QuadratureRule",
    # implemented families
    "GaussLegendreQuadrature",
    "GaussLobattoQuadrature",
    "TensorProductQuadrature",
    "DunavantQuadrature",
    "SUPPORTED_DEGREES",
    # declared placeholders
    "GaussJacobiQuadrature",
    "AdaptiveQuadrature",
    "SparseGridQuadrature",
    # selection
    "quadrature",
    "QuadratureFactory",
    "available_quadratures",
    "AVAILABLE_QUADRATURES",
    "DEFAULT_FAMILIES",
    # the verification standard
    "exact_monomial_integral",
    "monomial_values",
    "monomial_exponents",
    "monomial_exponents_up_to",
    # symmetry
    "symmetry_group",
    "AffineSymmetry",
]
