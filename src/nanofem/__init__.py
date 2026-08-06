"""nanofem -- a research-grade finite element framework for nanoscale mechanics.

Phase 1: the complete object model (data structures, declarations,
validation, serialization). No finite element mathematics is implemented.
See docs/design/ for the architecture (v2) and the SDS.

Responsibilities
----------------
- Curated public API re-exports (the object-model surface below)
- Package version metadata

TODO
----
- TODO(ADR-000): confirm final package name availability before first release
- TODO(phase-2): extend the public surface as numerics land
"""

from nanofem.analysis.buckling import BucklingOptions, LinearBucklingAnalysis
from nanofem.analysis.modal import ModalAnalysis, ModalOptions
from nanofem.analysis.static import LinearStaticAnalysis, StaticOptions
from nanofem.analysis.transient import TransientAnalysis, TransientOptions
from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import (
    BodyForce,
    LineLoad,
    NodalLoad,
    NonlocalAxialLoad,
    NonlocalTransverseLoad,
    TractionLoad,
)
from nanofem.constraints.mpc import MultiPointConstraint
from nanofem.constraints.neumann import NeumannBC
from nanofem.constraints.robin import RobinBC
from nanofem.core.dof_handler import Dof, DofHandler
from nanofem.core.fields import FieldSpec, VariableType
from nanofem.core.model import DomainDefinition, Model
from nanofem.materials.material import Material
from nanofem.mesh.mesh import Cell, CellBlock, Mesh
from nanofem.mesh.node import Node
from nanofem.mesh.region import Region
from nanofem.physics.base import DeclaredTheory, Locality, TheoryDeclaration
from nanofem.state.layout import StateLayout
from nanofem.state.model_state import ModelState

__version__ = "0.25.0"

__all__ = [
    "BodyForce",
    "BucklingOptions",
    "Cell",
    "CellBlock",
    "DeclaredTheory",
    "DirichletBC",
    "Dof",
    "DofHandler",
    "DomainDefinition",
    "FieldSpec",
    "LineLoad",
    "LinearBucklingAnalysis",
    "LinearStaticAnalysis",
    "LoadCase",
    "Locality",
    "Material",
    "Mesh",
    "ModalAnalysis",
    "ModalOptions",
    "Model",
    "ModelState",
    "MultiPointConstraint",
    "NeumannBC",
    "Node",
    "NodalLoad",
    "NonlocalAxialLoad",
    "NonlocalTransverseLoad",
    "Region",
    "RobinBC",
    "StateLayout",
    "StaticOptions",
    "TheoryDeclaration",
    "TractionLoad",
    "TransientAnalysis",
    "TransientOptions",
    "VariableType",
    "__version__",
]
