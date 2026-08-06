"""Strongly typed enumerations for the interpolation framework.

These name the interpolation families, the kinds of degree-of-freedom
functional, and the polynomial space types. They describe *metadata*: no
shape function, gradient, or evaluation machinery is implied.
"""

from __future__ import annotations

from enum import Enum, unique


@unique
class InterpolationFamily(Enum):
    """A family of interpolation on a reference element.

    ``LAGRANGE`` and ``HERMITE`` are implemented; the remaining three are
    declared so registries and interfaces anticipate them (see ``future.py``
    for what each still needs).
    """

    LAGRANGE = "lagrange"
    HERMITE = "hermite"
    SERENDIPITY = "serendipity"
    HIERARCHICAL = "hierarchical"
    SPECTRAL = "spectral"

    @property
    def is_implemented(self) -> bool:
        """Whether a concrete implementation exists in this phase."""
        return self in _IMPLEMENTED_FAMILIES

    @property
    def is_nodal_family(self) -> bool:
        """Whether this family's DOFs are point functionals (values/derivatives).

        ``HIERARCHICAL`` is the exception: its DOFs are modal coefficients on
        entities, not point evaluations, so it has no interpolation nodes.
        """
        return self is not InterpolationFamily.HIERARCHICAL


_IMPLEMENTED_FAMILIES: frozenset[InterpolationFamily] = frozenset(
    {InterpolationFamily.LAGRANGE, InterpolationFamily.HERMITE}
)


@unique
class DofKind(Enum):
    """The kind of functional a degree of freedom applies.

    ``POINT_VALUE`` is ``u -> u(x)``; ``POINT_DERIVATIVE`` is
    ``u -> D^alpha u(x)`` for a non-zero multi-index ``alpha``. ``MOMENT``
    (``u -> integral of u against a weight``) is named for the hierarchical
    and serendipity families and is not used in this phase.
    """

    POINT_VALUE = "point_value"
    POINT_DERIVATIVE = "point_derivative"
    MOMENT = "moment"


@unique
class PolynomialSpaceType(Enum):
    """The construction rule of a polynomial space.

    - ``TOTAL_DEGREE`` (``P_k``): every monomial of total degree <= k.
      The natural space of simplices.
    - ``TENSOR_PRODUCT`` (``Q_k``): every monomial of degree <= k in each
      variable separately. The natural space of tensor-product cells.
    - ``SERENDIPITY`` (``S_k``): ``P_k`` plus selected higher monomials,
      trading interior nodes for a smaller space. Declared, not built.
    """

    TOTAL_DEGREE = "P"
    TENSOR_PRODUCT = "Q"
    SERENDIPITY = "S"

    @property
    def symbol(self) -> str:
        """The conventional single-letter symbol of the space (``P``, ``Q``, ``S``)."""
        return self.value
