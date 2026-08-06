"""Exact monomial integrals over the reference domains, and monomial evaluation.

The verification standard for every rule in this package. A quadrature rule is
correct exactly when its weighted sum reproduces these closed forms for every
polynomial within its exactness degree, so these are computed *analytically* and
never by quadrature - a rule checked against itself proves nothing.

Independence from the interpolation layer
-----------------------------------------
Monomial evaluation is written here rather than imported from
``numerics.interpolation``, and this is a hard architectural constraint rather
than a preference. Phase 3 recorded that the spectral interpolation family
cannot precede quadrature, because its nodes *are* Gauss-Lobatto-Legendre
points: when that family lands, ``numerics.interpolation`` will import
``numerics.quadrature``. Were the dependency to run the other way as well, the
two packages could not both be imported. Quadrature therefore knows nothing
about shape functions, interpolation, or polynomial spaces - it integrates
scalar functions on a reference domain and nothing more.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.reference.element import ReferenceElement
from nanofem.numerics.reference.enums import CellType
from nanofem.utils.exceptions import InputValidationError

#: Reference domains built as products of intervals; the line is [-1, 1].
_CUBE_CELLS: frozenset[CellType] = frozenset(
    {CellType.LINE, CellType.QUADRILATERAL, CellType.HEXAHEDRON}
)

#: Reference domains that are unit simplices with a vertex at the origin.
_SIMPLEX_CELLS: frozenset[CellType] = frozenset({CellType.TRIANGLE, CellType.TETRAHEDRON})


def monomial_values(exponents: tuple[int, ...], points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate ``x^exponents`` at each point, shape ``(n_points,)``.

    The plainest possible power product. It needs no derivatives, so it carries
    none of the factorial machinery the interpolation layer's tabulator needs.
    """
    powers = np.asarray(exponents, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != powers.size:
        raise InputValidationError(
            f"points must have shape (n, {powers.size}) to match the exponent tuple, "
            f"got {points.shape}"
        )
    return np.asarray(np.prod(points**powers, axis=1), dtype=np.float64)


def exact_monomial_integral(element: ReferenceElement, exponents: tuple[int, ...]) -> float:
    """The exact integral of ``x^exponents`` over the reference domain.

    Two closed forms cover every domain in the library.

    **Products of intervals** (the line ``[-1, 1]``, the square, the cube). The
    integral separates, and each factor is ``2/(a+1)`` for an even power and
    zero for an odd one, since an odd power is antisymmetric about the origin.
    One odd exponent therefore annihilates the whole product.

    **Unit simplices** (the triangle with vertices at the origin and the axes).
    ``integral of prod x_i^a_i = (prod a_i!) / (sum a_i + d)!`` for a
    ``d``-simplex, which follows from repeated integration by parts, or from the
    Dirichlet integral. For the triangle this is ``p! q! / (p + q + 2)!``, and
    at ``p = q = 0`` it returns ``1/2``, the area.

    The line belongs to the first family, not the second: it is both a
    1-simplex and a 1-cube, but its reference domain is ``[-1, 1]``, so the
    interval formula is the one that applies.
    """
    cell = element.cell_type
    if len(exponents) != element.topological_dimension:
        raise InputValidationError(
            f"exponent tuple {exponents} has {len(exponents)} entries, expected "
            f"{element.topological_dimension} for a {cell.value}"
        )
    if any(power < 0 for power in exponents):
        raise InputValidationError(f"exponent tuple {exponents} has a negative entry")

    if cell in _CUBE_CELLS:
        total = 1.0
        for power in exponents:
            if power % 2 == 1:
                return 0.0
            total *= 2.0 / (power + 1)
        return total
    if cell in _SIMPLEX_CELLS:
        numerator = 1.0
        for power in exponents:
            numerator *= float(math.factorial(power))
        denominator = float(math.factorial(sum(exponents) + element.topological_dimension))
        return numerator / denominator
    raise NotImplementedError(
        f"no closed-form monomial integral for a {cell.value}; the prism and pyramid "
        f"need their own formulas and arrive with those reference elements"
    )


def monomial_exponents(degree: int, n_variables: int) -> tuple[tuple[int, ...], ...]:
    """Every exponent tuple of total degree exactly ``degree`` in ``n_variables``.

    Ordered deterministically so a failure always names the same monomial.
    """
    if degree < 0:
        raise InputValidationError(f"degree must be >= 0, got {degree}")
    if n_variables < 1:
        raise InputValidationError(f"n_variables must be >= 1, got {n_variables}")
    if n_variables == 1:
        return ((degree,),)
    found: list[tuple[int, ...]] = []
    for first in range(degree + 1):
        for rest in monomial_exponents(degree - first, n_variables - 1):
            found.append((first, *rest))
    return tuple(sorted(found))


def monomial_exponents_up_to(degree: int, n_variables: int) -> tuple[tuple[int, ...], ...]:
    """Every exponent tuple of total degree at most ``degree``."""
    found: list[tuple[int, ...]] = []
    for total in range(degree + 1):
        found.extend(monomial_exponents(total, n_variables))
    return tuple(found)
