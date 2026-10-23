"""What-if simulation engine.

Applies structured changes to a scheduling scenario, re-solves, and
compares the results against a baseline.

Pipeline:
  Original Scenario + Baseline Solution
    → validate change references
    → apply changes to a deep copy (original unchanged)
    → validate modified Scenario
    → solve modified Scenario (OR-Tools CP-SAT)
    → compare baseline vs. modified
    → SimulationResult

No LLM call is needed — changes are typed Pydantic objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field, ValidationError

from shift_happens.models.scenario import (
    ConstraintPriority,
    EmployeeAvailability,
    Scenario,
    ShiftRequirement,
)
from shift_happens.models.solution import (
    Assignment,
    ScheduleMetrics,
    ScheduleSolution,
    SolverStatus,
)
from shift_happens.solver.scheduler import solve
from shift_happens.validation.validator import validate_scenario


# ==================================================================
# Change types — typed what-if modifications
# ==================================================================


class SetEmployeeUnavailable(BaseModel):
    """Make an employee unavailable on a specific day."""

    employee_id: str = Field(..., min_length=1)
    day: int = Field(..., ge=0)
    priority: ConstraintPriority = ConstraintPriority.HARD
    penalty: int = Field(default=0, ge=0)


class ChangeStaffingRequirement(BaseModel):
    """Change the staffing requirement for a (day, shift_type) pair.

    If no requirement exists for this (day, shift_type), a new one is created.
    """

    day: int = Field(..., ge=0)
    shift_type: str = Field(..., min_length=1)
    min_employees: Optional[int] = Field(default=None, ge=0)
    max_employees: Optional[int] = Field(default=None, ge=0)


class ChangeWorkloadLimit(BaseModel):
    """Change one or more workload limit fields.

    Only fields set to a non-None value are updated.
    Use 0 for min fields or a large number for max fields to effectively remove a limit.
    """

    max_shifts_per_employee: Optional[int] = Field(default=None, ge=1)
    min_shifts_per_employee: Optional[int] = Field(default=None, ge=0)
    max_consecutive_days: Optional[int] = Field(default=None, ge=1)
    min_days_off: Optional[int] = Field(default=None, ge=0)


# Union of all supported change types
ScenarioChange = Union[
    SetEmployeeUnavailable,
    ChangeStaffingRequirement,
    ChangeWorkloadLimit,
]


# ==================================================================
# Simulation result types
# ==================================================================


class SimulationStatus(str, Enum):
    """Outcome of a what-if simulation."""

    SUCCESS = "success"              # Modified scenario solved (optimal/feasible)
    INFEASIBLE = "infeasible"        # Modified scenario is valid but infeasible
    INVALID_CHANGE = "invalid_change"  # Change references invalid entities
    VALIDATION_FAILED = "validation_failed"  # Modified scenario fails validation
    SOLVER_ERROR = "solver_error"    # Solver crashed


@dataclass
class AssignmentDiff:
    """Differences between baseline and modified assignment sets."""

    added: list[Assignment] = field(default_factory=list)
    removed: list[Assignment] = field(default_factory=list)
    unchanged: int = 0
    affected_employees: set[str] = field(default_factory=set)
    affected_days: set[int] = field(default_factory=set)


@dataclass
class MetricsDiff:
    """Numeric differences between baseline and modified metrics.

    Each field is (baseline_value, modified_value, delta).
    None means the metric was unavailable in one or both solutions.
    """

    total_assignments: tuple[int, int, int] | None = None
    coverage_pct: tuple[float, float, float] | None = None
    utilization_pct: tuple[float, float, float] | None = None
    soft_penalty_total: tuple[int, int, int] | None = None
    objective_value: tuple[float, float, float] | None = None


@dataclass
class SimulationResult:
    """Complete outcome of a what-if simulation.

    Inspect ``status`` first:
      - SUCCESS → ``modified_solution`` and ``diff`` are populated.
      - INFEASIBLE → ``modified_solution`` has status=infeasible.
      - INVALID_CHANGE → ``errors`` explains what went wrong.
      - VALIDATION_FAILED → ``errors`` explains why the modified scenario is invalid.
      - SOLVER_ERROR → ``errors`` explains the solver failure.
    """

    status: SimulationStatus
    baseline_scenario: Scenario | None = None
    baseline_solution: ScheduleSolution | None = None
    modified_scenario: Scenario | None = None
    modified_solution: ScheduleSolution | None = None
    changes_applied: list[ScenarioChange] = field(default_factory=list)
    assignment_diff: AssignmentDiff | None = None
    metrics_diff: MetricsDiff | None = None
    errors: list[str] = field(default_factory=list)


# ==================================================================
# Core simulation function
# ==================================================================


def simulate(
    scenario: Scenario,
    baseline_solution: ScheduleSolution,
    changes: list[ScenarioChange],
    *,
    solver_timeout_s: float = 60.0,
) -> SimulationResult:
    """Run a what-if simulation.

    Args:
        scenario: The original validated Scenario.
        baseline_solution: The solver result for *scenario*.
        changes: One or more typed modifications to apply.
        solver_timeout_s: Solver timeout for the modified scenario.

    Returns:
        A SimulationResult with the comparison.
    """
    if not changes:
        return SimulationResult(
            status=SimulationStatus.INVALID_CHANGE,
            baseline_scenario=scenario,
            baseline_solution=baseline_solution,
            errors=["No changes provided."],
        )

    # ---- Step 1: Validate change references ----
    employee_ids = {e.id for e in scenario.employees}
    shift_names = {s.name for s in scenario.shift_types}

    for change in changes:
        validation_errors = _validate_change(change, scenario, employee_ids, shift_names)
        if validation_errors:
            return SimulationResult(
                status=SimulationStatus.INVALID_CHANGE,
                baseline_scenario=scenario,
                baseline_solution=baseline_solution,
                changes_applied=[],
                errors=validation_errors,
            )

    # ---- Step 2: Apply changes to a deep copy ----
    try:
        modified = _apply_changes(scenario, changes)
    except (ValidationError, ValueError) as exc:
        return SimulationResult(
            status=SimulationStatus.INVALID_CHANGE,
            baseline_scenario=scenario,
            baseline_solution=baseline_solution,
            errors=[f"Failed to construct modified scenario: {exc}"],
        )

    # ---- Step 3: Semantic validation ----
    validation = validate_scenario(modified)
    if not validation.valid:
        return SimulationResult(
            status=SimulationStatus.VALIDATION_FAILED,
            baseline_scenario=scenario,
            baseline_solution=baseline_solution,
            modified_scenario=modified,
            changes_applied=list(changes),
            errors=validation.errors,
        )

    # ---- Step 4: Solve the modified scenario ----
    try:
        modified_solution = solve(modified, timeout_s=solver_timeout_s)
    except Exception as exc:
        return SimulationResult(
            status=SimulationStatus.SOLVER_ERROR,
            baseline_scenario=scenario,
            baseline_solution=baseline_solution,
            modified_scenario=modified,
            changes_applied=list(changes),
            errors=[f"Solver error: {type(exc).__name__}: {exc}"],
        )

    # ---- Step 5: Build comparison ----
    assignment_diff = _compare_assignments(
        baseline_solution.assignments, modified_solution.assignments
    )
    metrics_diff = _compare_metrics(
        baseline_solution, modified_solution
    )

    if modified_solution.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
        status = SimulationStatus.SUCCESS
    elif modified_solution.status == SolverStatus.INFEASIBLE:
        status = SimulationStatus.INFEASIBLE
    else:
        status = SimulationStatus.SOLVER_ERROR

    errors: list[str] = []
    if status == SimulationStatus.INFEASIBLE:
        if modified_solution.infeasibility_reason:
            errors.append(modified_solution.infeasibility_reason)
        else:
            errors.append(
                "The modified scenario is infeasible — no valid schedule exists "
                "for the changed constraints."
            )

    return SimulationResult(
        status=status,
        baseline_scenario=scenario,
        baseline_solution=baseline_solution,
        modified_scenario=modified,
        modified_solution=modified_solution,
        changes_applied=list(changes),
        assignment_diff=assignment_diff,
        metrics_diff=metrics_diff,
        errors=errors,
    )


# ==================================================================
# Internal helpers
# ==================================================================


def _validate_change(
    change: ScenarioChange,
    scenario: Scenario,
    employee_ids: set[str],
    shift_names: set[str],
) -> list[str]:
    """Validate that a change references entities that exist in the scenario."""
    errors: list[str] = []

    if isinstance(change, SetEmployeeUnavailable):
        if change.employee_id not in employee_ids:
            errors.append(
                f"Unknown employee_id '{change.employee_id}'. "
                f"Valid IDs: {sorted(employee_ids)}"
            )
        if change.day >= scenario.num_days:
            errors.append(
                f"Day {change.day} is out of range "
                f"(scenario has {scenario.num_days} days, valid: 0-{scenario.num_days - 1})."
            )

    elif isinstance(change, ChangeStaffingRequirement):
        if change.shift_type not in shift_names:
            errors.append(
                f"Unknown shift_type '{change.shift_type}'. "
                f"Valid types: {sorted(shift_names)}"
            )
        if change.day >= scenario.num_days:
            errors.append(
                f"Day {change.day} is out of range "
                f"(scenario has {scenario.num_days} days, valid: 0-{scenario.num_days - 1})."
            )
        if (
            change.min_employees is not None
            and change.max_employees is not None
            and change.max_employees < change.min_employees
        ):
            errors.append(
                f"max_employees ({change.max_employees}) cannot be less than "
                f"min_employees ({change.min_employees})."
            )

    # ChangeWorkloadLimit has no entity references to validate

    return errors


def _apply_changes(
    scenario: Scenario,
    changes: list[ScenarioChange],
) -> Scenario:
    """Apply changes to a deep copy of the scenario. Returns a new Scenario."""
    # Deep-copy via Pydantic serialization — guarantees no mutation
    data = scenario.model_dump()

    for change in changes:
        if isinstance(change, SetEmployeeUnavailable):
            avail_entry = {
                "employee_id": change.employee_id,
                "day": change.day,
                "priority": change.priority.value,
                "penalty": change.penalty,
            }
            data.setdefault("employee_availability", []).append(avail_entry)

        elif isinstance(change, ChangeStaffingRequirement):
            reqs = data.get("shift_requirements", [])
            # Find existing requirement for this (day, shift_type)
            found = False
            for req in reqs:
                if req["day"] == change.day and req["shift_type"] == change.shift_type:
                    if change.min_employees is not None:
                        req["min_employees"] = change.min_employees
                    if change.max_employees is not None:
                        req["max_employees"] = change.max_employees
                    found = True
                    break
            if not found:
                # Create new requirement
                new_req: dict = {
                    "day": change.day,
                    "shift_type": change.shift_type,
                    "min_employees": change.min_employees if change.min_employees is not None else 0,
                }
                if change.max_employees is not None:
                    new_req["max_employees"] = change.max_employees
                reqs.append(new_req)
            data["shift_requirements"] = reqs

        elif isinstance(change, ChangeWorkloadLimit):
            wl = data.get("workload_limits", {})
            if change.max_shifts_per_employee is not None:
                wl["max_shifts_per_employee"] = change.max_shifts_per_employee
            if change.min_shifts_per_employee is not None:
                wl["min_shifts_per_employee"] = change.min_shifts_per_employee
            if change.max_consecutive_days is not None:
                wl["max_consecutive_days"] = change.max_consecutive_days
            if change.min_days_off is not None:
                wl["min_days_off"] = change.min_days_off
            data["workload_limits"] = wl

    return Scenario.model_validate(data)


def _compare_assignments(
    baseline: list[Assignment],
    modified: list[Assignment],
) -> AssignmentDiff:
    """Compare two assignment lists and identify differences."""
    # Use (employee_id, day, shift_type) as the assignment key
    def _key(a: Assignment) -> tuple[str, int, str]:
        return (a.employee_id, a.day, a.shift_type)

    baseline_set = {_key(a) for a in baseline}
    modified_set = {_key(a) for a in modified}

    baseline_by_key = {_key(a): a for a in baseline}
    modified_by_key = {_key(a): a for a in modified}

    added_keys = modified_set - baseline_set
    removed_keys = baseline_set - modified_set
    unchanged_count = len(baseline_set & modified_set)

    added = [modified_by_key[k] for k in sorted(added_keys)]
    removed = [baseline_by_key[k] for k in sorted(removed_keys)]

    affected_employees: set[str] = set()
    affected_days: set[int] = set()
    for k in added_keys | removed_keys:
        affected_employees.add(k[0])
        affected_days.add(k[1])

    return AssignmentDiff(
        added=added,
        removed=removed,
        unchanged=unchanged_count,
        affected_employees=affected_employees,
        affected_days=affected_days,
    )


def _compare_metrics(
    baseline_solution: ScheduleSolution,
    modified_solution: ScheduleSolution,
) -> MetricsDiff:
    """Compare numeric metrics between baseline and modified solutions."""
    diff = MetricsDiff()

    bm = baseline_solution.metrics
    mm = modified_solution.metrics

    if bm is not None and mm is not None:
        diff.total_assignments = (
            bm.total_assignments,
            mm.total_assignments,
            mm.total_assignments - bm.total_assignments,
        )
        diff.coverage_pct = (
            bm.coverage_pct,
            mm.coverage_pct,
            round(mm.coverage_pct - bm.coverage_pct, 2),
        )
        diff.utilization_pct = (
            bm.utilization_pct,
            mm.utilization_pct,
            round(mm.utilization_pct - bm.utilization_pct, 2),
        )
        diff.soft_penalty_total = (
            bm.soft_penalty_total,
            mm.soft_penalty_total,
            mm.soft_penalty_total - bm.soft_penalty_total,
        )

    bo = baseline_solution.objective_value
    mo = modified_solution.objective_value
    if bo is not None and mo is not None:
        diff.objective_value = (bo, mo, round(mo - bo, 4))

    return diff
