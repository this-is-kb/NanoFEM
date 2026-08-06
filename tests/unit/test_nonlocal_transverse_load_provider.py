"""``NonlocalTransverseLoad``/``NonlocalTransverseLoadProvider``: the Eringen differential
Euler-Bernoulli beam's load term.

Mirrors ``test_nonlocal_axial_load_provider.py``'s structure. Because a Hermite cubic basis
makes a fully-worked closed form impractical for a unit test (unlike the bar's linear Lagrange
case), the "independent check" here is a from-scratch quadrature computation built directly in
this file using the same real Hermite/mapping stack the provider itself uses internally, but
written separately (not calling into the provider's own code) - an independent re-derivation,
not a hand-worked formula, which is still a genuine check against a coding mistake in the
provider (it would have to be wrong the same way in both places to go undetected).
"""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.constraints.loads import NonlocalTransverseLoad
from nanofem.constraints.nonlocal_load import NonlocalTransverseLoadProvider
from nanofem.core.dof_handler import DofHandler
from nanofem.core.fields import FieldSpec
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.numerics.assembly.contributions import ContributionKind, OperatorRole
from nanofem.utils.exceptions import InputValidationError, ModelError

LENGTH = 3.0
Q_A = 100.0
Q_B = 260.0
MU = 0.05


def _single_beam_mesh() -> Mesh:
    coords = np.array([[0.0], [LENGTH]])
    block = CellBlock("line2", np.array([[0, 1]]), region="beam")
    return Mesh(coords, (block,), (Region("all", 0, (0, 1)),))


def _provider(mu: float, q_a: float = Q_A, q_b: float = Q_B) -> NonlocalTransverseLoadProvider:
    mesh = _single_beam_mesh()
    dof_handler = DofHandler.generate(mesh, (FieldSpec("u", ("y",)), FieldSpec("r", ("z",))))
    load = NonlocalTransverseLoad("beam", "u", np.array([q_a, q_b]), mu)
    return NonlocalTransverseLoadProvider(load, mesh, dof_handler)


def _independent_reference_block(mu: float, q_a: float, q_b: float) -> np.ndarray:
    """A from-scratch quadrature computation, not calling the provider's own code."""
    from nanofem.numerics.interpolation import HermiteInterpolation, shape_functions
    from nanofem.numerics.mapping import AffineMapping
    from nanofem.numerics.quadrature import quadrature
    from nanofem.numerics.reference.enums import CellType
    from nanofem.physics.elasticity.euler_bernoulli import _reference_derivative_scale

    interpolation = HermiteInterpolation(CellType.LINE, order=3)
    basis = shape_functions(interpolation)
    rule = quadrature(CellType.LINE, order=6)
    mapping = AffineMapping(CellType.LINE, [[0.0], [LENGTH]])
    jacobian = mapping.jacobian_determinant(rule.points)
    dof_orders = tuple(dof.derivative[0] for dof in interpolation.dofs)
    scale = _reference_derivative_scale(dof_orders, float(jacobian[0]))

    ref_grad = basis.derivatives(rule.points)
    phys_grad = mapping.physical_gradient(ref_grad, rule.points)[:, :, 0]
    values = basis.evaluate(rule.points) * scale
    w_prime = phys_grad * scale

    q_local = np.array([q_a, 0.0, q_b, 0.0])
    q_vals = values @ q_local
    q_prime_vals = w_prime @ q_local

    f_classical = np.einsum("q,q,qa,q->a", rule.weights, jacobian, values, q_vals)
    f_nonlocal = mu * np.einsum("q,q,qa,q->a", rule.weights, jacobian, w_prime, q_prime_vals)
    return np.asarray(f_classical + f_nonlocal)


@pytest.mark.parametrize("mu", [0.0, MU])
def test_matches_an_independent_quadrature_computation(mu: float) -> None:
    provider = _provider(mu)
    (contribution,) = list(provider.contributions(OperatorRole.FORCE))
    assert contribution.kind is ContributionKind.CELL
    expected = _independent_reference_block(mu, Q_A, Q_B)
    np.testing.assert_allclose(contribution.block, expected, rtol=1e-10)

    empty = list(provider.contributions(OperatorRole.STIFFNESS))
    assert empty == []


def test_uniform_intensity_gives_zero_nonlocal_correction() -> None:
    """A uniform q(x) has q'=0: the beam analogue of the Peddieson-paradox null effect."""
    provider = _provider(MU, q_a=150.0, q_b=150.0)
    (contribution,) = list(provider.contributions(OperatorRole.FORCE))
    zero_mu_provider = _provider(0.0, q_a=150.0, q_b=150.0)
    (zero_mu_contribution,) = list(zero_mu_provider.contributions(OperatorRole.FORCE))
    np.testing.assert_allclose(contribution.block, zero_mu_contribution.block, rtol=1e-10)


def test_nodal_intensity_length_mismatch_raises() -> None:
    mesh = _single_beam_mesh()
    dof_handler = DofHandler.generate(mesh, (FieldSpec("u", ("y",)), FieldSpec("r", ("z",))))
    load = NonlocalTransverseLoad("beam", "u", np.array([1.0, 2.0, 3.0]), MU)
    provider = NonlocalTransverseLoadProvider(load, mesh, dof_handler)
    with pytest.raises(ModelError, match="entries"):
        list(provider.contributions(OperatorRole.FORCE))


def test_wrong_field_raises() -> None:
    mesh = _single_beam_mesh()
    dof_handler = DofHandler.generate(mesh, (FieldSpec("u", ("y",)), FieldSpec("r", ("z",))))
    load = NonlocalTransverseLoad("beam", "r", np.array([Q_A, Q_B]), MU)
    provider = NonlocalTransverseLoadProvider(load, mesh, dof_handler)
    with pytest.raises(ModelError, match="'u'"):
        list(provider.contributions(OperatorRole.FORCE))


def test_negative_nonlocal_parameter_rejected() -> None:
    with pytest.raises(InputValidationError):
        NonlocalTransverseLoad("beam", "u", np.array([Q_A, Q_B]), -1.0)
