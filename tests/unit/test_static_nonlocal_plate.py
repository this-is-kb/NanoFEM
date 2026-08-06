"""EringenDifferentialTheory through the full Mesh -> Model -> LinearStaticAnalysis pipeline.

This is the test that actually caught the sign bug documented in
``elements/continuum/nonlocal_continuum.py`` and ``docs/dev/notes.md``: the element-level
Schur-complement/constant-strain checks (``test_nonlocal_continuum_element.py``) all passed with
the wrong sign, because a *substitution* ``e* = K_ee^-1 K_eu u`` never assembles the monolithic
matrix and is insensitive to the diagonal block's sign. Only a real, monolithic solve - a
DirichletBC-partitioned, Dirichlet-reduced, solved-and-recovered pipeline exactly like every
other Stage-3 benchmark - reproduced a displacement with the right *magnitude* but the wrong
*sign*, which is what this file locks in as a regression.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.analysis.static import LinearStaticAnalysis
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import NodalLoad
from nanofem.core.model import DomainDefinition, Model
from nanofem.elements.factory import build_elements
from nanofem.geometry.plane import PlaneGeometry
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.physics.elasticity.eringen_differential import (
    EringenDifferentialMaterial,
    EringenDifferentialTheory,
)
from nanofem.physics.elasticity.plane import PlaneStressConstitutive

YOUNG_MODULUS = 200e9
POISSON = 0.3
LENGTH = 2.0
HEIGHT = 1.0
THICKNESS = 0.01
TOTAL_FORCE = 10_000.0


def _plate_model(e0a: float) -> Model:
    coords = np.array([[0.0, 0.0], [LENGTH, 0.0], [LENGTH, HEIGHT], [0.0, HEIGHT]])
    block = CellBlock("tri3", np.array([[0, 1, 2], [0, 2, 3]]), region="plate")
    regions = (
        Region("left_edge", 0, (0, 3)),
        Region("origin", 0, (0,)),
        Region("right_edge", 0, (1, 2)),
    )
    mesh = Mesh(coords, (block,), regions)
    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=POISSON, e0a=e0a))
    model.add_section("plane", PlaneGeometry(thickness=THICKNESS))
    model.add_theory("nonlocal", EringenDifferentialTheory(dim=2))
    model.add_constitutive("nonlocal_law", EringenDifferentialMaterial(PlaneStressConstitutive()))
    model.add_domain(
        DomainDefinition(
            "plate_domain",
            "plate",
            "nonlocal",
            "steel",
            geometry="plane",
            constitutive="nonlocal_law",
        )
    )
    model.add_dirichlet(DirichletBC("left_edge", "u", ("x",), 0.0))
    model.add_dirichlet(DirichletBC("origin", "u", ("y",), 0.0))
    case = LoadCase("tension")
    case.add(NodalLoad("right_edge", "u", np.array([TOTAL_FORCE / 2.0, 0.0])))
    model.add_load_case(case)
    return model


@pytest.mark.parametrize("e0a", [0.0, 0.3])
def test_tip_displacement_matches_1d_bar_formula_and_has_the_right_sign(e0a: float) -> None:
    """A uniaxial-consistent-loaded plate reproduces P*L/(E*H*t) with the CORRECT SIGN, for
    e0a=0 (the exact classical limit) and for e0a=0.3 (a constant-strain field, which the
    element-level tests already show gives zero nonlocal correction - so the same closed form
    should hold exactly here too, through the full pipeline, not just at the element level)."""
    model = _plate_model(e0a)
    result = LinearStaticAnalysis(model).run()["tension"]
    dof_handler = result.dof_handler
    expected = TOTAL_FORCE * LENGTH / (YOUNG_MODULUS * HEIGHT * THICKNESS)
    for node in (1, 2):
        u_x = result.displacements[dof_handler.global_dof(node, "u", "x")]
        assert u_x > 0.0, "regression guard: a sign-flipped solve gives a negative u_x here"
        assert u_x == pytest.approx(expected, rel=1e-9)


def test_build_elements_dispatches_to_nonlocal_continuum_element() -> None:
    from nanofem.elements.continuum.nonlocal_continuum import NonlocalContinuumElement

    model = _plate_model(e0a=0.1)
    dof_handler = model.build_dof_handler()
    elements = build_elements(model, dof_handler)
    assert len(elements) == 2
    assert all(isinstance(e, NonlocalContinuumElement) for e in elements)
