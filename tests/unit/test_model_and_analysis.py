"""Unit tests for the Model facade and analysis metadata (requirements 15-16).

This file is the executable success criterion: Mesh -> Material -> Theory ->
FieldSpec -> Model -> Analysis, without solving anything.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from nanofem.analysis.buckling import BucklingOptions, LinearBucklingAnalysis
from nanofem.analysis.modal import ModalAnalysis, ModalOptions
from nanofem.analysis.static import LinearStaticAnalysis
from nanofem.analysis.transient import TransientAnalysis, TransientOptions
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import NodalLoad
from nanofem.core.model import DomainDefinition, Model
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.numerics.assembly.contributions import ContributionKind, OperatorRole
from nanofem.numerics.operators.base import Continuity
from nanofem.physics.base import DeclaredTheory, TheoryDeclaration
from nanofem.utils.exceptions import (
    InputValidationError,
    MeshError,
    MissingPropertyError,
    ModelError,
)


def plate_mesh() -> Mesh:
    """Two quad4 cells with node regions 'left' and 'tip'."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    block = CellBlock("quad4", np.array([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=np.int64), "body")
    return Mesh(coords, (block,), (Region("left", 0, (0, 3)), Region("tip", 0, (2, 5))))


def elasticity() -> DeclaredTheory:
    """Local small-strain elasticity declaration (SDS 4.1)."""
    return DeclaredTheory(
        TheoryDeclaration(
            name="local_elasticity",
            field_requirements=(("u", 2),),
            continuity=(("u", Continuity.C0),),
            required_properties=("E", "nu", "rho"),
            operators=("symmetric_gradient", "voigt_map"),
            roles=(
                OperatorRole.STIFFNESS,
                OperatorRole.MASS,
                OperatorRole.GEOMETRIC_STIFFNESS,
                OperatorRole.FORCE,
            ),
            kinds=(ContributionKind.CELL,),
        )
    )


def build_model() -> Model:
    """The full success-criterion chain, ready to validate."""
    model = Model(plate_mesh())
    model.add_material(Material("silicon", E=169.0e9, nu=0.22, rho=2330.0))
    model.add_theory("local_elasticity", elasticity())
    model.add_domain(DomainDefinition("plate", "body", "local_elasticity", "silicon"))
    model.add_dirichlet(DirichletBC("left", "u", ("x", "y"), 0.0))
    case = LoadCase("service")
    case.add(NodalLoad("tip", "u", np.array([0.0, -1.0e-6])))
    model.add_load_case(case)
    return model


def test_success_criterion_chain_validates_and_numbers() -> None:
    """Mesh -> Material -> Theory -> FieldSpec -> Model -> DOFs, no solving."""
    model = build_model()
    model.validate()
    (u,) = model.field_specs()
    assert u.name == "u" and u.components == ("x", "y") and u.continuity is Continuity.C0
    dh = model.build_dof_handler()
    assert dh.num_dofs == 12  # 6 nodes x 2 components
    assert dh.global_dof(5, "u", "y") == 11


def test_validation_catches_missing_property() -> None:
    """Theory requires rho; a material without it fails by name (SDS D-1)."""
    model = Model(plate_mesh())
    model.add_material(Material("norho", E=1.0, nu=0.3))
    model.add_theory("local_elasticity", elasticity())
    model.add_domain(DomainDefinition("plate", "body", "local_elasticity", "norho"))
    with pytest.raises(MissingPropertyError, match="rho"):
        model.validate()


def test_validation_catches_unknown_targets() -> None:
    """Unknown regions, theories, and fields raise; nothing is silently dropped."""
    model = build_model()
    model.add_dirichlet(DirichletBC("ghost_region", "u", ("x",), 0.0))
    with pytest.raises(MeshError, match="ghost_region"):
        model.validate()

    model2 = Model(plate_mesh())
    model2.add_material(Material("m", E=1.0, nu=0.3, rho=1.0))
    model2.add_domain(DomainDefinition("plate", "body", "nope", "m"))
    with pytest.raises(ModelError, match="nope"):
        model2.validate()

    model3 = build_model()
    model3.add_dirichlet(DirichletBC("left", "u", ("z",), 0.0))
    with pytest.raises(ModelError, match="components"):
        model3.validate()


def test_duplicate_registrations_raise() -> None:
    """Registries reject duplicates instead of silently overriding."""
    model = build_model()
    with pytest.raises(ModelError):
        model.add_material(Material("silicon", E=1.0, nu=0.3))
    with pytest.raises(ModelError):
        model.add_theory("local_elasticity", elasticity())


def test_manifest_is_json_and_fingerprint_tracks_content() -> None:
    """Manifest serializes; fingerprint is stable and content-sensitive (C-5)."""
    a, b = build_model(), build_model()
    assert a.fingerprint() == b.fingerprint()
    json.dumps(a.to_manifest())  # must be JSON-safe
    c = build_model()
    c.add_material(Material("steel", E=200.0e9, nu=0.3, rho=7850.0))
    assert c.fingerprint() != a.fingerprint()


def test_analyses_carry_metadata_and_refuse_to_solve() -> None:
    """Orchestration metadata only: describe() works, run() raises (phase 1).

    ``LinearStaticAnalysis`` is excluded from the blanket ``run()``-raises
    loop: it is real as of the walking skeleton (SDS 2.18), covered
    separately in test_static_analytical.py. The 2-D ``build_model()`` plate
    here has no section/Bar-compatible domain, so it cannot exercise a real
    solve anyway - this test still checks its metadata via describe().
    """
    model = build_model()
    analyses = (
        ModalAnalysis(model, ModalOptions(num_modes=4)),
        LinearBucklingAnalysis(model, BucklingOptions(preload_case="service")),
        TransientAnalysis(model, TransientOptions(t_end=1.0, dt=0.01)),
    )
    for analysis in analyses:
        analysis.validate()
        description = analysis.describe()
        assert description["analysis"] and description["required_roles"]
        assert OperatorRole.STIFFNESS in analysis.required_roles()
        with pytest.raises(NotImplementedError):
            analysis.run()

    static = LinearStaticAnalysis(model)
    static.validate()
    description = static.describe()
    assert description["analysis"] and description["required_roles"]
    assert OperatorRole.STIFFNESS in static.required_roles()

    with pytest.raises(InputValidationError):
        ModalOptions(num_modes=0)
    with pytest.raises(InputValidationError):
        TransientOptions(t_end=1.0, dt=2.0)
