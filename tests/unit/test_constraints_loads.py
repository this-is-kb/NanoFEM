"""Unit tests for boundary conditions, loads, load cases, time functions (13-14)."""

from __future__ import annotations

import numpy as np
import pytest

from nanofem.constraints.dirichlet import DirichletBC
from nanofem.constraints.load_case import LoadCase
from nanofem.constraints.loads import BodyForce, LineLoad, NodalLoad, TractionLoad
from nanofem.constraints.mpc import MultiPointConstraint
from nanofem.constraints.neumann import NeumannBC
from nanofem.constraints.robin import RobinBC
from nanofem.constraints.time_functions import ConstantTF, HarmonicTF, RampTF
from nanofem.utils.exceptions import ConstraintConflictError, InputValidationError


def test_dirichlet_neumann_robin_are_data_with_validation() -> None:
    """Descriptions only: validated fields, no application machinery."""
    d = DirichletBC("left", "u", ("x", "y"), 0.0)
    n = NeumannBC("right", "u", np.array([1.0, 0.0]))
    r = RobinBC("bottom", "T", coefficient=5.0, ambient_value=293.0)
    assert d.components == ("x", "y") and n.flux.shape == (2,) and r.coefficient == 5.0
    with pytest.raises(InputValidationError):
        DirichletBC("", "u", ("x",), 0.0)
    with pytest.raises(InputValidationError):
        NeumannBC("right", "u", np.array([np.nan]))
    with pytest.raises(InputValidationError):
        RobinBC("b", "T", coefficient=-1.0, ambient_value=0.0)


def test_loads_validate_vectors() -> None:
    """Point/line/surface/body loads carry finite 1-D component vectors."""
    NodalLoad("tip", "u", np.array([0.0, -1.0]))
    LineLoad("beam", "u", np.array([0.0, -5.0]))
    TractionLoad("edge", "u", np.array([1.0, 0.0]))
    BodyForce("body", "u", np.array([0.0, -9.81]))
    with pytest.raises(InputValidationError):
        NodalLoad("tip", "u", np.array([[1.0]]))  # not 1-D


def test_mpc_validation() -> None:
    """Masters/coefficients must align; empty masters is a conflict."""
    MultiPointConstraint((5, "u", "x"), ((1, "u", "x"),), (1.0,))
    with pytest.raises(ConstraintConflictError):
        MultiPointConstraint((5, "u", "x"), (), ())
    with pytest.raises(ConstraintConflictError):
        MultiPointConstraint((5, "u", "x"), ((1, "u", "x"),), (1.0, 2.0))


def test_load_case_entries_and_time_functions() -> None:
    """LoadCase aggregates (load, factor, f(t)); only ConstantTF evaluates now."""
    case = LoadCase("service")
    case.add(
        NodalLoad("tip", "u", np.array([0.0, -1.0])), factor=1.5, time_function=ConstantTF(2.0)
    )
    (entry,) = case.entries
    assert entry.factor == 1.5
    assert entry.time_function is not None and entry.time_function(0.3) == 2.0
    with pytest.raises(NotImplementedError):
        RampTF(rise_time=1.0)(0.5)
    with pytest.raises(NotImplementedError):
        HarmonicTF(amplitude=1.0, omega=2.0)(0.5)
