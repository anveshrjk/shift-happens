"""Independent verification of hard constraints in a schedule solution.

This module checks solver output against the original scenario constraints
WITHOUT relying on the solver's own reporting.  Used by the benchmark to
confirm solution correctness.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from shift_happens.models.scenario import ConstraintPriority, Scenario
from shift_happens.models.solution import Assignment, ScheduleSolution, SolverStatus


@dataclass
class Violation:
    """A single hard-constraint violation."""
    constraint: str
    detail: str


@dataclass
class VerificationResult:
    """Outcome of independent constraint verification."""
    passed: bool = True
    violations: list[Violation] = field(default_factory=list)
    checks_performed: int = 0


def verify_solution(scenario: Scenario, solution: ScheduleSolution) -> VerificationResult:
    """Independently verify that *solution* satisfies all hard constraints.

    Only checks solutions where the solver reported OPTIMAL or FEASIBLE.
    Returns immediately for INFEASIBLE/ERROR statuses.
    """
    result = VerificationResult()

    if solution.status not in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
        # Nothing to verify — solver didn't claim a valid schedule
        return result

    _check_one_shift_per_day(scenario, solution, result)
    _check_staffing_requirements(scenario, solution, result)
    _check_hard_unavailability(scenario, solution, result)
    _check_hard_shift_restrictions(scenario, solution, result)
    _check_max_shifts_per_employee(scenario, solution, result)
    _check_min_days_off(scenario, solution, result)
    _check_max_consecutive_days(scenario, solution, result)
    _check_max_employees_cap(scenario, solution, result)

    result.passed = len(result.violations) == 0
    return result


# ------------------------------------------------------------------
# Individual constraint checks
# ------------------------------------------------------------------


def _check_one_shift_per_day(
    scenario: Scenario, solution: ScheduleSolution, result: VerificationResult
) -> None:
    """Each employee works at most one shift per day."""
    result.checks_performed += 1
    counter: Counter[tuple[str, int]] = Counter()
    for a in solution.assignments:
        key = (a.employee_id, a.day)
        counter[key] += 1
    for (emp_id, day), count in counter.items():
        if count > 1:
            result.violations.append(Violation(
                constraint="one_shift_per_day",
                detail=f"{emp_id} assigned {count} shifts on day {day}",
            ))


def _check_staffing_requirements(
    scenario: Scenario, solution: ScheduleSolution, result: VerificationResult
) -> None:
    """Min staffing requirements are met."""
    result.checks_performed += 1
    for req in scenario.shift_requirements:
        count = sum(
            1 for a in solution.assignments
            if a.day == req.day and a.shift_type == req.shift_type
        )
        if count < req.min_employees:
            result.violations.append(Violation(
                constraint="min_staffing",
                detail=(
                    f"Day {req.day} shift '{req.shift_type}': "
                    f"need {req.min_employees}, got {count}"
                ),
            ))


def _check_hard_unavailability(
    scenario: Scenario, solution: ScheduleSolution, result: VerificationResult
) -> None:
    """Hard unavailability constraints are respected."""
    result.checks_performed += 1
    hard_unavail = {
        (av.employee_id, av.day)
        for av in scenario.employee_availability
        if av.priority == ConstraintPriority.HARD
    }
    for a in solution.assignments:
        if (a.employee_id, a.day) in hard_unavail:
            result.violations.append(Violation(
                constraint="hard_unavailability",
                detail=f"{a.employee_id} assigned on day {a.day} but marked unavailable (hard)",
            ))


def _check_hard_shift_restrictions(
    scenario: Scenario, solution: ScheduleSolution, result: VerificationResult
) -> None:
    """Hard shift restrictions are respected."""
    result.checks_performed += 1
    hard_restrict = {
        (r.employee_id, r.shift_type)
        for r in scenario.employee_shift_restrictions
        if r.priority == ConstraintPriority.HARD
    }
    for a in solution.assignments:
        if (a.employee_id, a.shift_type) in hard_restrict:
            result.violations.append(Violation(
                constraint="hard_shift_restriction",
                detail=f"{a.employee_id} assigned to '{a.shift_type}' but restricted (hard)",
            ))


def _check_max_shifts_per_employee(
    scenario: Scenario, solution: ScheduleSolution, result: VerificationResult
) -> None:
    """Max shifts per employee is respected."""
    result.checks_performed += 1
    wl = scenario.workload_limits
    if wl.max_shifts_per_employee is None:
        return
    counts = Counter(a.employee_id for a in solution.assignments)
    for emp_id, count in counts.items():
        if count > wl.max_shifts_per_employee:
            result.violations.append(Violation(
                constraint="max_shifts_per_employee",
                detail=f"{emp_id} has {count} shifts, max is {wl.max_shifts_per_employee}",
            ))


def _check_min_days_off(
    scenario: Scenario, solution: ScheduleSolution, result: VerificationResult
) -> None:
    """Min days off per employee is respected."""
    result.checks_performed += 1
    wl = scenario.workload_limits
    if wl.min_days_off is None:
        return
    emp_ids = {e.id for e in scenario.employees}
    working_days: dict[str, set[int]] = {eid: set() for eid in emp_ids}
    for a in solution.assignments:
        working_days[a.employee_id].add(a.day)
    for emp_id in emp_ids:
        days_off = scenario.num_days - len(working_days[emp_id])
        if days_off < wl.min_days_off:
            result.violations.append(Violation(
                constraint="min_days_off",
                detail=f"{emp_id} has {days_off} days off, min is {wl.min_days_off}",
            ))


def _check_max_consecutive_days(
    scenario: Scenario, solution: ScheduleSolution, result: VerificationResult
) -> None:
    """Max consecutive working days is respected."""
    result.checks_performed += 1
    wl = scenario.workload_limits
    if wl.max_consecutive_days is None:
        return
    emp_ids = {e.id for e in scenario.employees}
    working_days: dict[str, set[int]] = {eid: set() for eid in emp_ids}
    for a in solution.assignments:
        working_days[a.employee_id].add(a.day)
    for emp_id in emp_ids:
        days = sorted(working_days[emp_id])
        if not days:
            continue
        streak = 1
        for i in range(1, len(days)):
            if days[i] == days[i - 1] + 1:
                streak += 1
            else:
                streak = 1
            if streak > wl.max_consecutive_days:
                result.violations.append(Violation(
                    constraint="max_consecutive_days",
                    detail=(
                        f"{emp_id} has {streak} consecutive working days "
                        f"(max {wl.max_consecutive_days})"
                    ),
                ))
                break


def _check_max_employees_cap(
    scenario: Scenario, solution: ScheduleSolution, result: VerificationResult
) -> None:
    """Max employees cap on shifts is respected."""
    result.checks_performed += 1
    for req in scenario.shift_requirements:
        if req.max_employees is None:
            continue
        count = sum(
            1 for a in solution.assignments
            if a.day == req.day and a.shift_type == req.shift_type
        )
        if count > req.max_employees:
            result.violations.append(Violation(
                constraint="max_employees_cap",
                detail=(
                    f"Day {req.day} shift '{req.shift_type}': "
                    f"max {req.max_employees}, got {count}"
                ),
            ))
