"""v0.8.0 success criterion, executable: the walking skeleton.

A steel bar, fixed at one end, loaded at the tip - built through every seam
named in ARCHITECTURE_v2.md's "phase-0 walking skeleton" milestone: Mesh ->
Material -> CrossSection -> Theory -> Model -> DofHandler -> elements ->
Assembler -> ConstraintHandler -> LinearStaticAnalysis -> SparseDirectSolver.

The payoff line at the end checks three things against independent hand
computations: the tip displacement equals the classical closed form
``u = PL/(EA)``, the fixed-end reaction balances the applied load, and the
element's closed-form stiffness matches the fully composed operators+tensors
path (ADR-002's equivalence requirement, SDS E-5).
"""

from __future__ import annotations

import numpy as np

from nanofem.analysis.static import LinearStaticAnalysis
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import NodalLoad
from nanofem.core.model import DomainDefinition, Model
from nanofem.elements.structural.bar import Bar
from nanofem.geometry.standard import CircularSection
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.numerics.interpolation import LagrangeInterpolation, shape_functions
from nanofem.numerics.mapping import AffineMapping
from nanofem.numerics.operators.symmetric_gradient import symmetric_gradient_matrix
from nanofem.numerics.quadrature import quadrature
from nanofem.numerics.reference.enums import CellType
from nanofem.physics.elasticity.isotropic import IsotropicElasticConstitutive, IsotropicElasticity

YOUNG_MODULUS = 200.0e9  # Pa, steel
RADIUS = 0.01  # m
LENGTH = 1.5  # m
TIP_FORCE = 1_000.0  # N


def build_model() -> Model:
    """Mesh -> Material -> CrossSection -> Theory -> Model, ready to solve."""
    coordinates = np.array([[0.0], [LENGTH]])
    block = CellBlock("line2", np.array([[0, 1]]), region="bar")
    mesh = Mesh(coordinates, (block,), (Region("fixed", 0, (0,)), Region("tip", 0, (1,))))

    model = Model(mesh)
    model.add_material(Material("steel", E=YOUNG_MODULUS, nu=0.3, rho=7850.0))
    model.add_section("circular", CircularSection(radius=RADIUS))
    model.add_theory("axial", IsotropicElasticity())
    model.add_domain(DomainDefinition("bar_domain", "bar", "axial", "steel", "circular"))
    model.add_dirichlet(DirichletBC("fixed", "u", ("x",), 0.0))

    service = LoadCase("service")
    service.add(NodalLoad("tip", "u", np.array([TIP_FORCE])))
    model.add_load_case(service)
    return model


def composed_path_stiffness(area: float) -> np.ndarray:
    """The general FE pipeline (mapping + interpolation + quadrature + B + D), from scratch.

    Built independently of ``Bar`` to discharge ADR-002's "declare
    equivalence... verification tests enforce it" - area is applied by hand,
    since a continuum Theory integrates over length only.
    """
    mapping = AffineMapping(CellType.LINE, [[0.0], [LENGTH]])
    basis = shape_functions(LagrangeInterpolation(CellType.LINE, order=1))
    rule = quadrature(CellType.LINE, order=1)

    reference_gradients = basis.derivatives(rule.points)
    physical_gradients = mapping.physical_gradient(reference_gradients, rule.points)
    b_matrix = symmetric_gradient_matrix(physical_gradients)
    jacobian = mapping.jacobian_determinant(rule.points)

    model = IsotropicElasticConstitutive()
    n_qp = rule.points.shape[0]
    strains = np.zeros((1, n_qp, 1))
    properties = {"E": np.full((1, n_qp), YOUNG_MODULUS)}
    _, tangent = model.respond_batch(strains, properties)

    k = np.zeros((2, 2))
    for q in range(n_qp):
        b_row = b_matrix[q, 0, :, 0]
        d_scalar = float(tangent[0, q, 0, 0])
        k += rule.weights[q] * jacobian[q] * d_scalar * np.outer(b_row, b_row)
    return area * k


def main() -> None:
    """Build, solve, and check the walking-skeleton bar against closed forms."""
    print("=== build the model ===")
    model = build_model()
    area = model.sections["circular"].area()
    print(f"  E = {YOUNG_MODULUS:.3e} Pa, A = {area:.6e} m^2, L = {LENGTH} m")

    print("\n=== solve ===")
    result = LinearStaticAnalysis(model).run()["service"]
    dof_handler = result.dof_handler
    u_fixed = result.displacements[dof_handler.global_dof(0, "u", "x")]
    u_tip = result.displacements[dof_handler.global_dof(1, "u", "x")]
    print(f"  u_fixed = {u_fixed:.6e} m")
    print(f"  u_tip   = {u_tip:.6e} m")
    print(f"  reaction at fixed end = {result.reactions[0]:.6f} N")

    print("\n=== payoff: check against independent closed forms ===")
    u_expected = TIP_FORCE * LENGTH / (YOUNG_MODULUS * area)
    print(f"  u_tip vs P*L/(E*A):        {u_tip:.9e}  vs  {u_expected:.9e}")
    assert np.isclose(u_tip, u_expected, rtol=1e-9)

    print(f"  reaction vs -P:            {result.reactions[0]:.6f}  vs  {-TIP_FORCE:.6f}")
    assert np.isclose(result.reactions[0], -TIP_FORCE, rtol=1e-9)

    bar = Bar(
        0, (0, 1), np.array([[0.0], [LENGTH]]), (0, 1), young_modulus=YOUNG_MODULUS, area=area
    )
    composed = composed_path_stiffness(area)
    print("  closed-form K   =\n" + str(bar.local_stiffness()))
    print("  composed-path K =\n" + str(composed))
    assert np.allclose(bar.local_stiffness(), composed, rtol=1e-9)

    print("\nall three checks passed - the walking skeleton is complete (v0.8.0)")


if __name__ == "__main__":
    main()
