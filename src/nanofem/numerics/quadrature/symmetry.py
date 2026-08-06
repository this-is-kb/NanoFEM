"""The symmetry group of a reference domain, as affine maps.

A symmetry of a reference element is an affine map that sends the domain onto
itself. Since an affine map is fixed by where it sends the vertices, and any
symmetry must permute the vertices, the whole group can be *derived*: try every
vertex permutation, fit an affine map to it, and keep the fits that are exact
and measure preserving. Nothing is hard-coded per cell, so the triangle's six
permutations, the square's eight dihedral maps, and the line's reflection all
fall out of one loop - and so will the tetrahedron's and the hexahedron's.

Quadrature uses this to check that a rule declaring symmetry really is invariant
under the element's symmetries: every point must map onto another point carrying
the same weight. That is what makes a symmetric rule's exactness independent of
how the element's vertices happen to be numbered.

This module deliberately does not import ``numerics.mapping``, even though the
affine fit is the same idea. Quadrature must stay a leaf of the numerics layer
(see ``moments.py`` for why), and the fit here is six lines rather than a
dependency.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cache

import numpy as np
from numpy.typing import NDArray

from nanofem.numerics.reference.element import ReferenceElement
from nanofem.numerics.reference.registry import reference_element

#: Residual above which an affine vertex fit is judged inexact.
_FIT_TOL: float = 1.0e-10

#: How far ``|det A|`` may stray from 1 before the map is not a symmetry.
_MEASURE_TOL: float = 1.0e-10


@dataclass(frozen=True)
class AffineSymmetry:
    """One symmetry of a reference domain: ``x -> A x + b``."""

    linear: NDArray[np.float64]
    translation: NDArray[np.float64]
    permutation: tuple[int, ...]

    def apply(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map a batch of points, shape ``(n, dim) -> (n, dim)``."""
        return np.asarray(points @ self.linear.T + self.translation, dtype=np.float64)

    @property
    def is_identity(self) -> bool:
        """Whether this is the trivial symmetry."""
        return self.permutation == tuple(range(len(self.permutation)))


def symmetry_group(element: ReferenceElement) -> tuple[AffineSymmetry, ...]:
    """Every affine map sending the reference domain onto itself.

    Six for the triangle (all vertex permutations), eight for the square (the
    dihedral group - the other sixteen permutations admit no affine fit), two
    for the line.
    """
    return _symmetry_group_cached(element.cell_type.value)


@cache
def _symmetry_group_cached(cell_name: str) -> tuple[AffineSymmetry, ...]:
    """Derive and memoize the group. Keyed by name so the cache stays hashable.

    Memoized rather than recomputed: a reference element is canonical per shape,
    so its symmetry group is a constant of the library. The cache is a pure
    memo on a pure function of an immutable key - it changes nothing about the
    result, only how often the search runs.
    """
    element = reference_element(cell_name)
    vertices = element.vertex_coordinates
    count = element.num_vertices
    augmented = np.hstack([vertices, np.ones((count, 1))])
    found: list[AffineSymmetry] = []
    for permutation in itertools.permutations(range(count)):
        target = vertices[list(permutation)]
        solution, *_ = np.linalg.lstsq(augmented, target, rcond=None)
        if not np.allclose(augmented @ solution, target, atol=_FIT_TOL):
            continue  # no affine map realizes this permutation
        linear = np.ascontiguousarray(solution[:-1].T)
        if abs(abs(float(np.linalg.det(linear))) - 1.0) > _MEASURE_TOL:
            continue  # an affine map that rescales is not a symmetry
        translation = np.ascontiguousarray(solution[-1])
        linear.setflags(write=False)
        translation.setflags(write=False)
        found.append(AffineSymmetry(linear, translation, permutation))
    return tuple(found)
