"""Cantilever beam, tip-loaded: the classical Timoshenko closed-form benchmark.

Unlike ``test_beam_eb_cantilever_benchmark.py`` (exact for a *single* cubic
Hermite element), a single selective-reduced-integration Timoshenko element
is **not** exact for a cantilever - this is a genuine, verified property of
the formulation (see ``test_timoshenko_beam_verification.py``'s
mesh-convergence test), not a weaker test written here. This benchmark uses
a moderately refined 8-element mesh and a correspondingly loose tolerance
(``rtol=5e-3``; the convergence data shows an 8-element mesh is within
~0.4% of exact), a preview of the later verification-suite's mesh-
convergence/cantilever-beam items.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.elements.structural.beam_timoshenko import TimoshenkoBeam

N_ELEMENTS = 8
LENGTH = 3.0
YOUNG_MODULUS = 200e9
SHEAR_MODULUS = 80e9
SECOND_MOMENT = 4.0e-6
SHEAR_AREA = 0.001
TIP_FORCE = 1_000.0


def test_cantilever_tip_load_matches_closed_form_within_mesh_tolerance() -> None:
    n_nodes = N_ELEMENTS + 1
    n_dof = 2 * n_nodes
    stiffness = np.zeros((n_dof, n_dof))
    element_length = LENGTH / N_ELEMENTS

    for element_index in range(N_ELEMENTS):
        x0 = element_index * element_length
        x1 = (element_index + 1) * element_length
        dofs = (
            2 * element_index,
            2 * element_index + 1,
            2 * element_index + 2,
            2 * element_index + 3,
        )
        beam = TimoshenkoBeam(
            element_index,
            (element_index, element_index + 1),
            np.array([[x0], [x1]]),
            dofs,
            young_modulus=YOUNG_MODULUS,
            shear_modulus=SHEAR_MODULUS,
            second_moment=SECOND_MOMENT,
            shear_area=SHEAR_AREA,
        )
        local = beam.local_stiffness()
        for a in range(4):
            for b in range(4):
                stiffness[dofs[a], dofs[b]] += local[a, b]

    # Node 0 (w, theta) fixed; tip transverse force at the last node's w DOF.
    force = np.zeros(n_dof)
    force[-2] = TIP_FORCE
    free_dofs = list(range(2, n_dof))
    reduced = stiffness[np.ix_(free_dofs, free_dofs)]
    solution = np.linalg.solve(reduced, force[free_dofs])
    tip_deflection = solution[-2]

    expected = TIP_FORCE * LENGTH**3 / (
        3.0 * YOUNG_MODULUS * SECOND_MOMENT
    ) + TIP_FORCE * LENGTH / (SHEAR_MODULUS * SHEAR_AREA)
    assert tip_deflection == pytest.approx(expected, rel=5e-3)
