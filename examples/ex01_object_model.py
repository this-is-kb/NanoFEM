"""Phase-1 success criterion, executable: Mesh -> Material -> Theory -> FieldSpec
-> Model -> Analysis, without solving anything.

Builds a two-cell silicon nanoplate model, validates it, numbers DOFs
deterministically, registers four analyses, and round-trips the model
manifest and DOF numbering through JSON. Zero FEM mathematics.
"""

from __future__ import annotations

import json

import numpy as np

from nanofem.analysis.buckling import BucklingOptions, LinearBucklingAnalysis
from nanofem.analysis.modal import ModalAnalysis, ModalOptions
from nanofem.analysis.static import LinearStaticAnalysis
from nanofem.analysis.transient import TransientAnalysis, TransientOptions
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import NodalLoad
from nanofem.constraints.time_functions import ConstantTF
from nanofem.core.dof_handler import DofHandler
from nanofem.core.model import DomainDefinition, Model
from nanofem.materials.material import Material
from nanofem.mesh.mesh import CellBlock, Mesh
from nanofem.mesh.region import Region
from nanofem.numerics.assembly.contributions import ContributionKind, OperatorRole
from nanofem.numerics.operators.base import Continuity
from nanofem.physics.base import DeclaredTheory, TheoryDeclaration


def main() -> None:
    """Run the object-model chain end to end and print a run summary."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    block = CellBlock("quad4", np.array([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=np.int64), "body")
    mesh = Mesh(coords, (block,), (Region("left", 0, (0, 3)), Region("tip", 0, (2, 5))))

    silicon = Material("silicon", E=169.0e9, nu=0.22, rho=2330.0)

    elasticity = DeclaredTheory(
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

    model = Model(mesh)
    model.add_material(silicon)
    model.add_theory("local_elasticity", elasticity)
    model.add_domain(DomainDefinition("plate", "body", "local_elasticity", "silicon"))
    model.add_dirichlet(DirichletBC("left", "u", ("x", "y"), 0.0))
    service = LoadCase("service")
    service.add(NodalLoad("tip", "u", np.array([0.0, -1.0e-6])), time_function=ConstantTF(1.0))
    model.add_load_case(service)

    model.validate()
    dof_handler = model.build_dof_handler()

    analyses = (
        LinearStaticAnalysis(model),
        ModalAnalysis(model, ModalOptions(num_modes=6)),
        LinearBucklingAnalysis(model, BucklingOptions(preload_case="service")),
        TransientAnalysis(model, TransientOptions(t_end=1.0e-6, dt=1.0e-8)),
    )
    for analysis in analyses:
        analysis.validate()

    manifest = json.dumps(model.to_manifest(), sort_keys=True)
    reloaded = DofHandler.import_numbering(dof_handler.export_numbering())

    print(f"fields          : {[f.name for f in model.field_specs()]}")
    print(f"global DOFs     : {dof_handler.num_dofs}")
    print(f"dof(node5,u,y)  : {dof_handler.global_dof(5, 'u', 'y')}")
    print(f"model fingerprint: {model.fingerprint()[:16]}...")
    print(f"manifest bytes  : {len(manifest)} (JSON round-trip OK)")
    print(
        f"numbering round-trip fingerprints match: "
        f"{reloaded.fingerprint() == dof_handler.fingerprint()}"
    )
    for analysis in analyses:
        d = analysis.describe()
        print(f"analysis {d['analysis']:<24} roles={d['required_roles']}")
    print("no matrices were assembled and nothing was solved - phase 1 complete")


if __name__ == "__main__":
    main()
