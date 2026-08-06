"""``NonlocalAxialLoad``/``NonlocalAxialLoadProvider``: the Eringen differential bar's load term.

Two independent checks: (1) a single-element closed-form check of the two integrals the
provider computes (the classical consistent load for a *linearly-varying* nodal intensity,
plus the nonlocal gradient correction, both worked out by hand for a 2-node line), and (2) the
``nonlocal_parameter=0`` reduction, which must reproduce the classical consistent-load-only
result exactly (SDS Section 5's "null response"-style consistency check, applied to a load
provider rather than a constitutive law). The full mesh-convergence benchmark against the
closed-form fixed-fixed nonlocal bar solution lives in
``test_nonlocal_bar_benchmark.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.constraints.loads import NonlocalAxialLoad
from nanofem.constraints.nonlocal_load import NonlocalAxialLoadProvider
from nanofem.core.dof_handler import DofHandler
from nanofem.core.fields import FieldSpec
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.numerics.assembly.contributions import ContributionKind, OperatorRole
from nanofem.utils.exceptions import InputValidationError, ModelError

LENGTH = 2.0
Q_A = 100.0
Q_B = 260.0
MU = 0.05


def _single_bar_mesh() -> Mesh:
    coords = np.array([[0.0], [LENGTH]])
    block = CellBlock("line2", np.array([[0, 1]]), region="bar")
    return Mesh(coords, (block,), (Region("all", 0, (0, 1)),))


def _provider(mu: float) -> NonlocalAxialLoadProvider:
    mesh = _single_bar_mesh()
    dof_handler = DofHandler.generate(mesh, (FieldSpec("u", ("x",)),))
    load = NonlocalAxialLoad("bar", "u", np.array([Q_A, Q_B]), mu)
    return NonlocalAxialLoadProvider(load, mesh, dof_handler, ("x",))


def test_matches_the_hand_derived_closed_form_for_a_single_element() -> None:
    """Linearly-varying nodal intensity on one 2-node element: both terms have a clean closed
    form. Classical: L/6 * [2*Q_A + Q_B, Q_A + 2*Q_B]. Nonlocal: mu/L * [-(Q_B-Q_A), Q_B-Q_A]."""
    provider = _provider(MU)
    (contribution,) = list(provider.contributions(OperatorRole.FORCE))
    assert contribution.kind is ContributionKind.CELL

    f_classical = (LENGTH / 6.0) * np.array([2.0 * Q_A + Q_B, Q_A + 2.0 * Q_B])
    delta_q = Q_B - Q_A
    f_nonlocal = (MU / LENGTH) * np.array([-delta_q, delta_q])
    expected = f_classical + f_nonlocal
    np.testing.assert_allclose(contribution.block, expected, rtol=1e-12)

    empty = list(provider.contributions(OperatorRole.STIFFNESS))
    assert empty == []


def test_zero_nonlocal_parameter_reduces_to_the_classical_consistent_load() -> None:
    """mu=0 must reproduce the classical linear-element consistent load exactly - no residue."""
    provider = _provider(0.0)
    (contribution,) = list(provider.contributions(OperatorRole.FORCE))
    f_classical = (LENGTH / 6.0) * np.array([2.0 * Q_A + Q_B, Q_A + 2.0 * Q_B])
    np.testing.assert_allclose(contribution.block, f_classical, rtol=1e-12)


def test_uniform_intensity_gives_zero_nonlocal_correction() -> None:
    """A spatially-*uniform* q(x) has q'=0 everywhere: the nonlocal term vanishes identically -
    the discrete analogue of the "Peddieson paradox" this class's own docstring names."""
    mesh = _single_bar_mesh()
    dof_handler = DofHandler.generate(mesh, (FieldSpec("u", ("x",)),))
    load = NonlocalAxialLoad("bar", "u", np.array([150.0, 150.0]), MU)
    provider = NonlocalAxialLoadProvider(load, mesh, dof_handler, ("x",))
    (contribution,) = list(provider.contributions(OperatorRole.FORCE))
    f_classical_uniform = 150.0 * LENGTH / 2.0 * np.array([1.0, 1.0])
    np.testing.assert_allclose(contribution.block, f_classical_uniform, rtol=1e-12)


def test_nodal_intensity_length_mismatch_raises() -> None:
    mesh = _single_bar_mesh()
    dof_handler = DofHandler.generate(mesh, (FieldSpec("u", ("x",)),))
    load = NonlocalAxialLoad("bar", "u", np.array([1.0, 2.0, 3.0]), MU)
    provider = NonlocalAxialLoadProvider(load, mesh, dof_handler, ("x",))
    with pytest.raises(ModelError, match="entries"):
        list(provider.contributions(OperatorRole.FORCE))


def test_multi_component_field_raises() -> None:
    mesh = _single_bar_mesh()
    dof_handler = DofHandler.generate(mesh, (FieldSpec("u", ("x", "y")),))
    load = NonlocalAxialLoad("bar", "u", np.array([Q_A, Q_B]), MU)
    provider = NonlocalAxialLoadProvider(load, mesh, dof_handler, ("x", "y"))
    with pytest.raises(ModelError, match="1-D axial"):
        list(provider.contributions(OperatorRole.FORCE))


def test_negative_nonlocal_parameter_rejected() -> None:
    with pytest.raises(InputValidationError):
        NonlocalAxialLoad("bar", "u", np.array([Q_A, Q_B]), -1.0)
