"""Strongly typed enumeration of quadrature families.

Names how a rule's points and weights are constructed, which is what determines
its exactness, its point count, and whether its weights are positive.
"""

from __future__ import annotations

from enum import Enum, unique


@unique
class QuadratureFamily(Enum):
    """The construction rule of a quadrature scheme."""

    GAUSS_LEGENDRE = "gauss_legendre"
    GAUSS_LOBATTO = "gauss_lobatto"
    TENSOR_PRODUCT = "tensor_product"
    DUNAVANT = "dunavant"
    GAUSS_JACOBI = "gauss_jacobi"
    ADAPTIVE = "adaptive"
    SPARSE_GRID = "sparse_grid"

    @property
    def is_implemented(self) -> bool:
        """Whether a concrete rule exists for this family in this phase."""
        return self in _IMPLEMENTED

    @property
    def includes_endpoints(self) -> bool:
        """Whether the family places points on the boundary of its domain.

        The distinction that matters downstream: a Gauss rule's points are all
        interior, so a field may be evaluated at them even where the boundary is
        awkward. A Lobatto rule includes the endpoints, which is exactly why it
        can collocate with nodal points and diagonalize a mass matrix.
        """
        return self is QuadratureFamily.GAUSS_LOBATTO


_IMPLEMENTED: frozenset[QuadratureFamily] = frozenset(
    {
        QuadratureFamily.GAUSS_LEGENDRE,
        QuadratureFamily.GAUSS_LOBATTO,
        QuadratureFamily.TENSOR_PRODUCT,
        QuadratureFamily.DUNAVANT,
    }
)
