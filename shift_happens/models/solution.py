"""Pydantic models for solver output: assignments, metrics, and solution status."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SolverStatus(str, Enum):
    """Result status from the CP-SAT solver."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    ERROR = "error"


class Assignment(BaseModel):
    """A single shift assignment: one employee on one day in one shift."""

    employee_id: str
    employee_name: str
    day: int = Field(..., ge=0)
    shift_type: str


class EmployeeMetrics(BaseModel):
    """Per-employee summary statistics."""

    employee_id: str
    employee_name: str
    total_shifts: int = Field(..., ge=0)
    shifts_by_type: dict[str, int] = Field(default_factory=dict)
    days_off: int = Field(..., ge=0)


class ScheduleMetrics(BaseModel):
    """Aggregate metrics for the whole schedule."""

    total_assignments: int = Field(..., ge=0)
    coverage_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Percentage of shift slots filled"
    )
    employee_metrics: list[EmployeeMetrics] = Field(default_factory=list)
    soft_penalty_total: int = Field(default=0, ge=0)
    utilization_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Average employee utilization as % of available days",
    )


class ScheduleSolution(BaseModel):
    """Complete solver result."""

    status: SolverStatus
    assignments: list[Assignment] = Field(default_factory=list)
    objective_value: float | None = None
    metrics: ScheduleMetrics | None = None
    solver_wall_time_s: float | None = None
    infeasibility_reason: str | None = None
