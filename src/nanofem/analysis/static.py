"""Linear static analysis: K u = f per load case (SDS 2.18, the walking skeleton)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from nanofem.analysis.base import AnalysisBase
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.handler import ConstraintHandler
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import (
    NodalLoad,
    NodalLoadProvider,
    NonlocalAxialLoad,
    NonlocalTransverseLoad,
    TractionLoad,
)
from nanofem.constraints.nonlocal_load import (
    NonlocalAxialLoadProvider,
    NonlocalTransverseLoadProvider,
)
from nanofem.constraints.traction import TractionLoadProvider
from nanofem.core.dof_handler import DofHandler
from nanofem.core.model import Model
from nanofem.elements.factory import build_elements
from nanofem.numerics.assembly.assembler import Assembler
from nanofem.numerics.assembly.contributions import ContributionProvider, OperatorRole
from nanofem.numerics.assembly.sparsity import SparsityPattern
from nanofem.numerics.assembly.system import GlobalSystem, ReducedSystem
from nanofem.numerics.linalg.linear import SparseDirectSolver
from nanofem.utils.exceptions import ModelError


@dataclass(frozen=True)
class StaticOptions:
    """Which registered load cases to run (empty = all)."""

    load_cases: tuple[str, ...] = ()


@dataclass(frozen=True)
class StaticResult:
    """One load case's solved state: full displacement vector, reactions at constrained DOFs."""

    displacements: NDArray[np.float64]
    reactions: NDArray[np.float64]
    dof_handler: DofHandler


class LinearStaticAnalysis(AnalysisBase):
    """Assemble STIFFNESS + FORCE, constrain, solve, recover, package."""

    def __init__(self, model: Model, options: StaticOptions | None = None) -> None:
        super().__init__(model)
        self.options = options if options is not None else StaticOptions()

    def required_roles(self) -> tuple[OperatorRole, ...]:
        """Static recipe roles."""
        return (OperatorRole.STIFFNESS, OperatorRole.FORCE)

    def validate(self) -> None:
        """Base checks plus load-case name resolution."""
        super().validate()
        unknown = [c for c in self.options.load_cases if c not in self.model.load_case_names]
        if unknown:
            raise ModelError(
                f"static analysis references unknown load cases {unknown} "
                f"(registered: {list(self.model.load_case_names)})"
            )

    def run(self) -> dict[str, StaticResult]:
        """Assemble once, then solve every selected load case against the same K.

        STIFFNESS and the sparsity pattern are shared across load cases
        (SDS 2.13: pattern identity is contract); each load case gets its own
        FORCE assembly and constraint reduction.
        """
        self.validate()
        model = self.model
        dof_handler = model.build_dof_handler()
        elements = build_elements(model, dof_handler)
        pattern = SparsityPattern.from_providers(
            elements, OperatorRole.STIFFNESS, dof_handler.num_dofs
        )
        assembler = Assembler(pattern)
        stiffness = cast(csr_matrix, assembler.assemble(elements, OperatorRole.STIFFNESS))
        # Model.dirichlet_bcs is typed as the narrow ConstraintLike protocol (core cannot
        # import constraints.DirichletBC, N-9); every entry Model.add_dirichlet ever receives
        # in this phase is a real DirichletBC, which ConstraintHandler needs concretely.
        dirichlet_bcs = cast(tuple[DirichletBC, ...], model.dirichlet_bcs)
        partition = ConstraintHandler(model.mesh, dof_handler, dirichlet_bcs).partition()
        field_components = {spec.name: spec.components for spec in model.field_specs()}

        case_names = self.options.load_cases or model.load_case_names
        results: dict[str, StaticResult] = {}
        for name in case_names:
            # Model.load_case returns the narrow LoadCaseLike protocol (name only); every
            # registered case in this phase is a real LoadCase, which has .entries.
            case = cast(LoadCase, model.load_case(name))
            providers: list[ContributionProvider] = []
            for entry in case.entries:
                if entry.time_function is not None:
                    raise ModelError(
                        f"load case '{name}': LinearStaticAnalysis does not support time "
                        "functions (static analysis has no time axis)"
                    )
                if isinstance(entry.load, NodalLoad):
                    providers.append(
                        NodalLoadProvider(
                            entry.load,
                            model.mesh,
                            dof_handler,
                            field_components[entry.load.field],
                            factor=entry.factor,
                        )
                    )
                elif isinstance(entry.load, TractionLoad):
                    providers.append(
                        TractionLoadProvider(
                            entry.load,
                            model.mesh,
                            dof_handler,
                            field_components[entry.load.field],
                            factor=entry.factor,
                        )
                    )
                elif isinstance(entry.load, NonlocalAxialLoad):
                    providers.append(
                        NonlocalAxialLoadProvider(
                            entry.load,
                            model.mesh,
                            dof_handler,
                            field_components[entry.load.field],
                            factor=entry.factor,
                        )
                    )
                elif isinstance(entry.load, NonlocalTransverseLoad):
                    providers.append(
                        NonlocalTransverseLoadProvider(
                            entry.load, model.mesh, dof_handler, factor=entry.factor
                        )
                    )
                else:
                    raise ModelError(
                        f"load case '{name}': LinearStaticAnalysis supports NodalLoad/"
                        f"TractionLoad/NonlocalAxialLoad/NonlocalTransverseLoad only, got "
                        f"{type(entry.load).__name__}"
                    )
            force = cast("NDArray[np.float64]", assembler.assemble(providers, OperatorRole.FORCE))
            system = GlobalSystem(
                dof_handler.num_dofs,
                {OperatorRole.STIFFNESS: stiffness},
                {OperatorRole.FORCE: force},
            )
            reduced = ReducedSystem.from_global(
                system,
                partition.free_dofs,
                partition.constrained_dofs,
                partition.prescribed_values,
            )
            u_f = SparseDirectSolver().solve(reduced.k_ff, reduced.f_f)
            results[name] = StaticResult(
                displacements=reduced.recover(u_f),
                reactions=reduced.reactions(u_f),
                dof_handler=dof_handler,
            )
        return results
