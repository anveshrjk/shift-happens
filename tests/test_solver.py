"""Tests for the OR-Tools CP-SAT scheduler.

All tests are deterministic — no LLM dependency.
"""

import pytest

from shift_happens.models.scenario import (
    ConstraintPriority,
    Employee,
    EmployeeAvailability,
    EmployeeShiftRestriction,
    Scenario,
    ShiftRequirement,
    ShiftType,
    WorkloadLimits,
)
from shift_happens.models.solution import SolverStatus
from shift_happens.solver.scheduler import solve


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_employees(n: int) -> list[Employee]:
    names = [
        "Alice", "Bob", "Charlie", "Diana", "Eve",
        "Frank", "Grace", "Hank", "Ivy", "Jack",
    ]
    return [Employee(id=names[i].lower(), name=names[i]) for i in range(n)]


def _make_shifts(*names: str) -> list[ShiftType]:
    return [ShiftType(name=n) for n in names]


def _uniform_requirements(
    num_days: int, shift_names: list[str], min_emp: int
) -> list[ShiftRequirement]:
    return [
        ShiftRequirement(day=d, shift_type=s, min_employees=min_emp)
        for d in range(num_days)
        for s in shift_names
    ]


# ------------------------------------------------------------------
# Feasibility tests
# ------------------------------------------------------------------


class TestBasicFeasibility:
    """Simple scenarios that must be feasible."""

    def test_trivial_1emp_1shift_1day(self):
        scenario = Scenario(
            num_days=1,
            employees=_make_employees(1),
            shift_types=_make_shifts("morning"),
            shift_requirements=[
                ShiftRequirement(day=0, shift_type="morning", min_employees=1)
            ],
        )
        sol = solve(scenario)
        assert sol.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert len(sol.assignments) == 1
        assert sol.assignments[0].employee_name == "Alice"
        assert sol.assignments[0].shift_type == "morning"

    def test_2emp_2shift_3day(self):
        scenario = Scenario(
            num_days=3,
            employees=_make_employees(2),
            shift_types=_make_shifts("morning", "night"),
            shift_requirements=_uniform_requirements(3, ["morning", "night"], 1),
        )
        sol = solve(scenario)
        assert sol.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert len(sol.assignments) == 6  # 3 days × 2 shifts × 1 each

    def test_5emp_3shift_7day(self):
        """A moderately sized problem."""
        scenario = Scenario(
            num_days=7,
            employees=_make_employees(5),
            shift_types=_make_shifts("morning", "afternoon", "night"),
            shift_requirements=_uniform_requirements(
                7, ["morning", "afternoon", "night"], 1
            ),
        )
        sol = solve(scenario)
        assert sol.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert len(sol.assignments) == 21  # 7 × 3 × 1

    def test_no_requirements_is_feasible(self):
        """With no staffing requirements, any assignment (including none) is fine."""
        scenario = Scenario(
            num_days=3,
            employees=_make_employees(2),
            shift_types=_make_shifts("morning"),
        )
        sol = solve(scenario)
        assert sol.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)


# ------------------------------------------------------------------
# Infeasibility tests
# ------------------------------------------------------------------


class TestInfeasibility:
    """Scenarios that cannot have a valid schedule."""

    def test_more_required_than_available(self):
        """2 employees but each shift needs 3."""
        scenario = Scenario(
            num_days=1,
            employees=_make_employees(2),
            shift_types=_make_shifts("morning"),
            shift_requirements=[
                ShiftRequirement(day=0, shift_type="morning", min_employees=3)
            ],
        )
        sol = solve(scenario)
        assert sol.status == SolverStatus.INFEASIBLE
        assert sol.assignments == []
        assert sol.infeasibility_reason is not None

    def test_all_employees_unavailable(self):
        """Every employee is unavailable on the only day."""
        employees = _make_employees(3)
        scenario = Scenario(
            num_days=1,
            employees=employees,
            shift_types=_make_shifts("morning"),
            shift_requirements=[
                ShiftRequirement(day=0, shift_type="morning", min_employees=1)
            ],
            employee_availability=[
                EmployeeAvailability(employee_id=e.id, day=0)
                for e in employees
            ],
        )
        sol = solve(scenario)
        assert sol.status == SolverStatus.INFEASIBLE


# ------------------------------------------------------------------
# Constraint satisfaction tests
# ------------------------------------------------------------------


class TestConstraints:
    """Verify that specific constraints are respected."""

    def test_one_shift_per_employee_per_day(self):
        scenario = Scenario(
            num_days=1,
            employees=_make_employees(3),
            shift_types=_make_shifts("morning", "night"),
            shift_requirements=[
                ShiftRequirement(day=0, shift_type="morning", min_employees=1),
                ShiftRequirement(day=0, shift_type="night", min_employees=1),
            ],
        )
        sol = solve(scenario)
        assert sol.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        # No employee should appear twice on day 0
        day0_emps = [a.employee_id for a in sol.assignments if a.day == 0]
        assert len(day0_emps) == len(set(day0_emps))

    def test_employee_shift_restriction_hard(self):
        """Alice cannot work night shifts."""
        scenario = Scenario(
            num_days=2,
            employees=_make_employees(3),
            shift_types=_make_shifts("morning", "night"),
            shift_requirements=_uniform_requirements(2, ["morning", "night"], 1),
            employee_shift_restrictions=[
                EmployeeShiftRestriction(
                    employee_id="alice",
                    shift_type="night",
                    priority=ConstraintPriority.HARD,
                )
            ],
        )
        sol = solve(scenario)
        assert sol.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        alice_nights = [
            a for a in sol.assignments
            if a.employee_id == "alice" and a.shift_type == "night"
        ]
        assert len(alice_nights) == 0

    def test_employee_unavailability_hard(self):
        """Bob is unavailable on day 0."""
        scenario = Scenario(
            num_days=2,
            employees=_make_employees(3),
            shift_types=_make_shifts("morning"),
            shift_requirements=_uniform_requirements(2, ["morning"], 1),
            employee_availability=[
                EmployeeAvailability(
                    employee_id="bob", day=0, priority=ConstraintPriority.HARD
                )
            ],
        )
        sol = solve(scenario)
        assert sol.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        bob_day0 = [
            a for a in sol.assignments
            if a.employee_id == "bob" and a.day == 0
        ]
        assert len(bob_day0) == 0

    def test_max_shifts_per_employee(self):
        """Each employee works at most 2 shifts over 5 days."""
        scenario = Scenario(
            num_days=5,
            employees=_make_employees(5),
            shift_types=_make_shifts("morning"),
            shift_requirements=_uniform_requirements(5, ["morning"], 1),
            workload_limits=WorkloadLimits(max_shifts_per_employee=2),
        )
        sol = solve(scenario)
        assert sol.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        from collections import Counter
        counts = Counter(a.employee_id for a in sol.assignments)
        for emp_id, count in counts.items():
            assert count <= 2, f"{emp_id} has {count} shifts, max is 2"

    def test_min_days_off(self):
        """Each employee gets at least 2 days off in a 5-day schedule."""
        scenario = Scenario(
            num_days=5,
            employees=_make_employees(5),
            shift_types=_make_shifts("morning"),
            shift_requirements=_uniform_requirements(5, ["morning"], 1),
            workload_limits=WorkloadLimits(min_days_off=2),
        )
        sol = solve(scenario)
        assert sol.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        from collections import Counter
        counts = Counter(a.employee_id for a in sol.assignments)
        for emp in scenario.employees:
            worked = counts.get(emp.id, 0)
            days_off = scenario.num_days - worked
            assert days_off >= 2, f"{emp.name} has {days_off} days off, needs 2"


# ------------------------------------------------------------------
# Metrics tests
# ------------------------------------------------------------------


class TestMetrics:
    """Verify that metrics are computed correctly."""

    def test_coverage_100pct(self):
        scenario = Scenario(
            num_days=2,
            employees=_make_employees(3),
            shift_types=_make_shifts("morning"),
            shift_requirements=_uniform_requirements(2, ["morning"], 1),
        )
        sol = solve(scenario)
        assert sol.metrics is not None
        assert sol.metrics.coverage_pct == 100.0

    def test_metrics_employee_count(self):
        scenario = Scenario(
            num_days=2,
            employees=_make_employees(3),
            shift_types=_make_shifts("morning"),
            shift_requirements=_uniform_requirements(2, ["morning"], 1),
        )
        sol = solve(scenario)
        assert sol.metrics is not None
        assert len(sol.metrics.employee_metrics) == 3

    def test_solver_wall_time_reported(self):
        scenario = Scenario(
            num_days=1,
            employees=_make_employees(1),
            shift_types=_make_shifts("morning"),
            shift_requirements=[
                ShiftRequirement(day=0, shift_type="morning", min_employees=1)
            ],
        )
        sol = solve(scenario)
        assert sol.solver_wall_time_s is not None
        assert sol.solver_wall_time_s >= 0


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    """Boundary and unusual scenarios."""

    def test_single_employee_single_day(self):
        scenario = Scenario(
            num_days=1,
            employees=_make_employees(1),
            shift_types=_make_shifts("morning"),
            shift_requirements=[
                ShiftRequirement(day=0, shift_type="morning", min_employees=1)
            ],
        )
        sol = solve(scenario)
        assert sol.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)

    def test_many_employees_few_slots(self):
        """10 employees but only 1 slot per day — should still be feasible."""
        scenario = Scenario(
            num_days=3,
            employees=_make_employees(10),
            shift_types=_make_shifts("morning"),
            shift_requirements=_uniform_requirements(3, ["morning"], 1),
        )
        sol = solve(scenario)
        assert sol.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)

    def test_max_employees_cap(self):
        """Shift has max_employees = 1, so at most 1 employee per slot."""
        scenario = Scenario(
            num_days=1,
            employees=_make_employees(5),
            shift_types=_make_shifts("morning"),
            shift_requirements=[
                ShiftRequirement(
                    day=0, shift_type="morning",
                    min_employees=1, max_employees=1,
                )
            ],
        )
        sol = solve(scenario)
        assert sol.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        day0_count = sum(1 for a in sol.assignments if a.day == 0)
        assert day0_count == 1
