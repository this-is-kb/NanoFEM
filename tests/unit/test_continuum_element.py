"""ContinuumElement: the generic integration path, against Bar and hand-derived closed forms.

The stiffness check closes the loop test_bar_verification.py opened: that
test builds the composed pipeline from scratch as a one-off oracle for
``Bar``; this file compares two independent pieces of *production* code
(``ContinuumElement`` vs. ``Bar``) directly, which is a different statement.
test_bar_verification.py is intentionally left untouched (see its own
docstring) so a bug shared between the two would still be caught by neither
alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.constraints.loads import BodyForce
from nanofem.core.dof_handler import DofHandler
from nanofem.core.fields import displacement
from nanofem.elements.continuum.continuum import ContinuumBodyForceProvider, ContinuumElement
from nanofem.elements.dof_utils import build_local_dof_map
from nanofem.elements.structural.bar import Bar
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.numerics.assembly.contributions import ContributionKind, OperatorRole
from nanofem.numerics.operators.base import Continuity
from nanofem.numerics.reference.enums import CellType
from nanofem.physics.base import DeclaredTheory, Locality, Theory, TheoryDeclaration
from nanofem.physics.elasticity.isotropic import IsotropicElasticConstitutive, IsotropicElasticity
from nanofem.utils.exceptions import InputValidationError


def _element(
    x0: float,
    x1: float,
    e: float,
    a: float,
    *,
    rho: float | None = None,
    theory: Theory | None = None,
) -> ContinuumElement:
    material = Material("m", E=e) if rho is None else Material("m", E=e, rho=rho)
    return ContinuumElement(
        cell_id=0,
        node_ids=(0, 1),
        coordinates=np.array([[x0], [x1]]),
        global_dofs=np.array([0, 1], dtype=np.int64),
        cell_type=CellType.LINE,
        interpolation_order=1,
        theory=theory if theory is not None else IsotropicElasticity(dim=1),
        constitutive=IsotropicElasticConstitutive(dim=1),
        material=material,
        section_measure=a,
    )


@pytest.mark.parametrize(
    ("x0", "x1", "e", "a"), [(0.0, 2.5, 71.7e9, 0.0025), (1.0, 4.0, 200e9, 1e-4)]
)
def test_stiffness_matches_bar(x0: float, x1: float, e: float, a: float) -> None:
    continuum = _element(x0, x1, e, a)
    bar = Bar(0, (0, 1), np.array([[x0], [x1]]), (0, 1), young_modulus=e, area=a)
    np.testing.assert_allclose(continuum.local_stiffness(), bar.local_stiffness(), rtol=1e-9)


def test_mass_matches_hand_derivation() -> None:
    x0, x1, e, a, rho = 0.0, 2.0, 200e9, 0.001, 7850.0
    continuum = _element(x0, x1, e, a, rho=rho)
    length = x1 - x0
    expected = rho * a * length / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    np.testing.assert_allclose(continuum.local_mass(), expected)


def test_body_force_matches_hand_derivation() -> None:
    x0, x1, e, a = 0.0, 2.0, 200e9, 0.001
    continuum = _element(x0, x1, e, a)
    length = x1 - x0
    b = 5.0
    expected = b * a * length / 2.0 * np.array([1.0, 1.0])
    np.testing.assert_allclose(continuum.local_body_force(np.array([b])), expected)


def test_dof_signature_matches_bar_convention() -> None:
    continuum = _element(0.0, 1.0, 1.0, 1.0)
    assert continuum.dof_signature().dof_names_per_node == (("u.x",), ("u.x",))


def test_contributions_emit_stiffness_and_mass_but_not_force() -> None:
    continuum = _element(0.0, 1.0, 1.0, 1.0, rho=1.0)
    (stiffness,) = list(continuum.contributions(OperatorRole.STIFFNESS))
    assert stiffness.kind is ContributionKind.CELL
    np.testing.assert_allclose(stiffness.block, continuum.local_stiffness())
    (mass,) = list(continuum.contributions(OperatorRole.MASS))
    np.testing.assert_allclose(mass.block, continuum.local_mass())
    assert list(continuum.contributions(OperatorRole.FORCE)) == []


def _stiffness_only_theory() -> DeclaredTheory:
    declaration = TheoryDeclaration(
        name="axial_stiffness_only",
        field_requirements=(("u", 1),),
        continuity=(("u", Continuity.C0),),
        required_properties=("E",),
        operators=("symmetric_gradient", "voigt_map"),
        roles=(OperatorRole.STIFFNESS,),
        kinds=(ContributionKind.CELL,),
        locality=Locality.LOCAL,
    )
    return DeclaredTheory(declaration)


def test_mass_role_absent_never_requires_rho() -> None:
    continuum = _element(0.0, 1.0, 1.0, 1.0, theory=_stiffness_only_theory())
    assert list(continuum.contributions(OperatorRole.MASS)) == []
    continuum.local_stiffness()  # must not raise even though the material has no "rho"


def test_rejects_multi_field_theory() -> None:
    declaration = TheoryDeclaration(
        name="two_fields",
        field_requirements=(("u", 1), ("T", 1)),
        continuity=(("u", Continuity.C0), ("T", Continuity.C0)),
        required_properties=("E",),
        operators=("symmetric_gradient", "voigt_map"),
        roles=(OperatorRole.STIFFNESS,),
        kinds=(ContributionKind.CELL,),
    )
    with pytest.raises(InputValidationError, match="single-field"):
        _element(0.0, 1.0, 1.0, 1.0, theory=DeclaredTheory(declaration))


def test_body_force_provider_sums_to_global_equilibrium() -> None:
    coordinates = np.array([[0.0], [1.0], [2.5]])
    block = CellBlock("line2", np.array([[0, 1], [1, 2]]), region="bar")
    mesh = Mesh(coordinates, (block,), (Region("all", 1, (0, 1)),))
    dof_handler = DofHandler.generate(mesh, (displacement(1),))

    theory = IsotropicElasticity(dim=1)
    constitutive = IsotropicElasticConstitutive(dim=1)
    material = Material("m", E=200e9)
    area = 0.001
    body_force_value = 5.0

    elements = {}
    for cell_id in mesh.cells_in_region("all"):
        cell = mesh.cell(cell_id)
        node_a, node_b = cell.connectivity
        coords = np.array([mesh.node(node_a).coordinates, mesh.node(node_b).coordinates])
        global_dofs = build_local_dof_map(dof_handler, (node_a, node_b), "u", ("x",))
        elements[cell_id] = ContinuumElement(
            cell_id,
            (node_a, node_b),
            coords,
            global_dofs,
            CellType.LINE,
            1,
            theory,
            constitutive,
            material,
            section_measure=area,
        )

    body_force = BodyForce("all", "u", np.array([body_force_value]))
    provider = ContinuumBodyForceProvider(body_force, mesh, elements)
    total = np.zeros(dof_handler.num_dofs)
    for contribution in provider.contributions(OperatorRole.FORCE):
        np.add.at(total, contribution.row_dofs, contribution.block)

    total_length = 2.5
    assert total.sum() == pytest.approx(body_force_value * area * total_length)
    assert list(provider.contributions(OperatorRole.STIFFNESS)) == []
