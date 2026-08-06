"""SDS C-1 encoded once: Voigt ordering and the engineering-shear split (ADR-016).

Voigt is the wire format of all B/D/stress/strain exchanges: strain vectors
are *kinematic* Voigt (engineering shear, gamma = 2 eps), stress vectors are
*kinetic* Voigt (plain components), so work conjugacy holds without factors.
Converters (Voigt <-> Mandel <-> full) are the ONLY sanctioned representation
bridge; hand-inserted factors of 2 are the classic cross-developer bug this
module exists to kill.
"""

from __future__ import annotations

VOIGT_ORDER: dict[int, tuple[tuple[int, int], ...]] = {
    1: ((0, 0),),
    2: ((0, 0), (1, 1), (0, 1)),
    3: ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)),
}

VOIGT_LENGTH: dict[int, int] = {1: 1, 2: 3, 3: 6}
