"""Validate a Scenario before it reaches the solver.

The Pydantic model already enforces schema-level rules (types, ranges,
referential integrity).  This module adds *semantic* validation:
checks that are about the meaning of the scheduling problem, not its shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shift_happens.models.scenario import Scenario


@dataclass
class ValidationResult:
    """Outcome of semantic validation."""

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_scenario(scenario: Scenario) -> ValidationResult:
    """Run all semantic checks on *scenario* and return a ValidationResult.

    This does NOT duplicate the Pydantic model_validator checks (those already
    ran at construction time).  Instead it looks for problems the solver is
    likely to hit, such as obviously infeasible staffing requirements.
    """
    result = ValidationResult()

    _check_staffing_feasibility(scenario, result)
    _check_days_off_feasibility(scenario, result)
    _check_workload_limits_consistency(scenario, result)
    _check_shift_requirement_coverage(scenario, result)

    result.valid = len(result.errors) == 0
    return result


# ------------------------------------------------------------------
# Individual checks
# ------------------------------------------------------------------


def _check_staffing_feasibility(scenario: Scenario, result: ValidationResult) -> None:
    """Warn if total workforce may be too small for staffing requirements."""
    num_employees = len(scenario.employees)
    for req in scenario.shift_requirements:
        if req.min_employees > num_employees:
            result.errors.append(
                f"Day {req.day}, shift '{req.shift_type}': requires at least "
                f"{req.min_employees} employees, but only {num_employees} exist."
            )


def _check_days_off_feasibility(scenario: Scenario, result: ValidationResult) -> None:
    """Error if required days off exceed total days."""
    wl = scenario.workload_limits
    if wl.min_days_off is not None and wl.min_days_off > scenario.num_days:
        result.errors.append(
            f"min_days_off ({wl.min_days_off}) exceeds the schedule length "
            f"({scenario.num_days} days)."
        )

    if wl.min_days_off is not None and wl.max_shifts_per_employee is not None:
        max_possible = scenario.num_days - wl.min_days_off
        if wl.max_shifts_per_employee < max_possible:
            # Not an error — just fewer shifts allowed.
            pass
        if wl.min_shifts_per_employee is not None:
            if wl.min_shifts_per_employee > max_possible:
                result.errors.append(
                    f"min_shifts_per_employee ({wl.min_shifts_per_employee}) is "
                    f"impossible given min_days_off ({wl.min_days_off}) over "
                    f"{scenario.num_days} days (max workable = {max_possible})."
                )


def _check_workload_limits_consistency(
    scenario: Scenario, result: ValidationResult
) -> None:
    """Warn about workload limits that are obviously contradictory."""
    wl = scenario.workload_limits
    if (
        wl.min_shifts_per_employee is not None
        and wl.max_shifts_per_employee is not None
        and wl.min_shifts_per_employee > wl.max_shifts_per_employee
    ):
        result.errors.append(
            f"min_shifts_per_employee ({wl.min_shifts_per_employee}) > "
            f"max_shifts_per_employee ({wl.max_shifts_per_employee})."
        )


def _check_shift_requirement_coverage(
    scenario: Scenario, result: ValidationResult
) -> None:
    """Warn if some (day, shift) pairs have no requirement defined."""
    shift_names = {s.name for s in scenario.shift_types}
    covered = {(r.day, r.shift_type) for r in scenario.shift_requirements}
    for day in range(scenario.num_days):
        for shift in shift_names:
            if (day, shift) not in covered:
                result.warnings.append(
                    f"No staffing requirement for day {day}, shift '{shift}'. "
                    f"Solver will allow 0 employees."
                )
