"""Complete NanoFEM exception hierarchy (SDS module failure conditions).

Every module raises from this tree; users can catch ``NanoFEMError`` alone.
Errors MUST carry mechanics context (element/node/region ids), never bare
internal indices. No other module defines exception types.
"""

from __future__ import annotations


class NanoFEMError(Exception):
    """Base class for every error raised by NanoFEM."""


class InputValidationError(NanoFEMError):
    """Bad user input, caught at object construction (fail fast, SDS 2.x)."""


class MeshError(NanoFEMError):
    """Mesh data integrity failures (SDS 2.1)."""


class MeshImportError(MeshError):
    """A mesh file could not be read or converted."""


class DegenerateCellError(MeshError):
    """Non-positive Jacobian or zero-measure cell (SDS 2.6)."""


class ModelError(NanoFEMError):
    """Incomplete or inconsistent model definition (SDS 2.18)."""


class MissingMaterialError(ModelError):
    """An element set has no material assigned."""


class MissingSectionError(ModelError):
    """A structural element has no cross-section assigned (SDS 2.2)."""


class MissingPropertyError(ModelError):
    """A constitutive model requires a property the material does not define (SDS Section 5)."""


class ConstraintConflictError(ModelError):
    """Contradictory Dirichlet values or MPC cycles (SDS 2.14)."""


class PhysicsError(NanoFEMError):
    """Theory or constitutive contract violations (SDS Sections 4-5)."""


class IncompatibleContinuityError(PhysicsError):
    """Theory continuity requirement exceeds interpolation capability (SDS E-9)."""


class AssemblyError(NanoFEMError):
    """Contribution scatter failures (SDS 2.13)."""


class DofMappingError(AssemblyError):
    """A provider emitted a DOF index outside the global range (SDS 2.12)."""


class SolverError(NanoFEMError):
    """Numerical linear algebra failures (SDS 2.15-2.17)."""


class SingularMatrixError(SolverError):
    """Singular factorization; carries suspected under-constrained DOFs (SDS 2.15)."""


class IllConditionedError(SolverError):
    """Escalated conditioning warning under strict mode (SDS Section 13)."""


class ConvergenceError(SolverError):
    """Iterative, eigen, or (future) Newton non-convergence; carries residual history."""


class StateError(NanoFEMError):
    """State lifecycle or layout violations (SDS Section 7)."""


class PostProcessError(NanoFEMError):
    """Recovery, sampling, or export failures (SDS 2.19)."""
