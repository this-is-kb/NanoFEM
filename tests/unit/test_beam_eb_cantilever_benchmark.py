"""Cantilever beam, tip-loaded: the classical closed-form benchmark (a preview of Step 8).

A single cubic-Hermite Euler-Bernoulli element is exact for a tip point load
(the true deflection curve under this loading *is* a cubic), so a
single-element cantilever should match ``w = PL^3/(3EI)`` and
``theta = PL^2/(2EI)`` to machine precision - no assembler needed, since one
element already *is* the whole system once node 1's DOFs are eliminated by
hand (mirroring how ``test_bar_analytical.py`` never needs the assembler to
check ``Bar`` in isolation).
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.elements.structural.beam_eb import EulerBernoulliBeam


def test_cantilever_tip_load_matches_closed_form() -> None:
    length, young_modulus, second_moment, tip_force = 3.0, 200e9, 4.0e-6, 1_000.0
    beam = EulerBernoulliBeam(
        0,
        (0, 1),
        np.array([[0.0], [length]]),
        (0, 1, 2, 3),
        young_modulus=young_modulus,
        second_moment=second_moment,
    )
    stiffness = beam.local_stiffness()

    # Node 1 (w1, theta1) fixed; node 2 (w2, theta2) free, loaded by a pure
    # transverse tip force (no applied moment) - partition the 4x4 by hand.
    k_ff = stiffness[2:, 2:]
    f_f = np.array([tip_force, 0.0])
    w2, theta2 = np.linalg.solve(k_ff, f_f)

    expected_w2 = tip_force * length**3 / (3.0 * young_modulus * second_moment)
    expected_theta2 = tip_force * length**2 / (2.0 * young_modulus * second_moment)
    assert w2 == pytest.approx(expected_w2, rel=1e-9)
    assert theta2 == pytest.approx(expected_theta2, rel=1e-9)
