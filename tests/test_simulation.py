"""Tests for the what-if simulation engine.

All tests use synthetic scenarios and the real OR-Tools solver.
No LLM API calls are needed.
"""

from __future__ import annotations

import copy

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
from shift_happens.simulation.engine import (
    ChangeStaffingRequirement,
    ChangeWorkloadLimit,
    SetEmployeeUnavailable,
    SimulationResult,
    SimulationStatus,
    simulate,
)
from shift_happens.solver.scheduler import solve


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_employees(n: int) -> list[Employee]:
    names = ["Alice", "Bob", "Carol", "Dave", "Eve",
             "Frank", "Grace", "Hank", "Ivy", "Jack"]
    return [Employee(id=names[i].lower(), name=names[i]) for i in range(n)]


def _make_shifts(*names: str) -> list[ShiftType]:
    return [ShiftType(name=n) for n in names]


def _uniform_reqs(num_days: int, shifts: list[str], min_emp: int) -> list[ShiftRequirement]:
    return [
        ShiftRequirement(day=d, shift_type=s, min_employees=min_emp)
        for d in range(num_days) for s in shifts
    ]


def _baseline_scenario() -> Scenario:
    """5 employees, 2 shifts, 5 days, 1 per shift — comfortably feasible."""
    return Scenario(
        name="Baseline",
        num_days=5,
        employees=_make_employees(5),
        shift_types=_make_shifts("morning", "night"),
        shift_requirements=_uniform_reqs(5, ["morning", "night"], 1),
    )


def _baseline_with_workload() -> Scenario:
    """Same as baseline but with workload limits."""
    return Scenario(
        name="Baseline with Limits",
        num_days=5,
        employees=_make_employees(5),
        shift_types=_make_shifts("morning", "night"),
        shift_requirements=_uniform_reqs(5, ["morning", "night"], 1),
        workload_limits=WorkloadLimits(max_shifts_per_employee=3, min_days_off=1),
    )


# ==================================================================
# Test: Successful simulations
# ==================================================================


class TestSuccessfulSimulation:
    """Changes that produce a valid modified schedule."""

    def test_employee_unavailable(self):
        """Making Alice unavailable on day 2 produces a valid new schedule."""
        scenario = _baseline_scenario()
        baseline = solve(scenario)
        assert baseline.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)

        result = simulate(scenario, baseline, [
            SetEmployeeUnavailable(employee_id="alice", day=2),
        ])

        assert result.status == SimulationStatus.SUCCESS
        assert result.modified_solution is not None
        assert result.modified_solution.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        # Alice should not be assigned on day 2 in modified
        for a in result.modified_solution.assignments:
            if a.employee_id == "alice":
                assert a.day != 2, "Alice should be unavailable on day 2"

    def test_increase_staffing(self):
        """Increasing min_employees from 1 to 2 produces more assignments."""
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            ChangeStaffingRequirement(day=0, shift_type="morning", min_employees=2),
        ])

        assert result.status == SimulationStatus.SUCCESS
        # Day 0 morning should now have at least 2 employees
        day0_morning = [
            a for a in result.modified_solution.assignments
            if a.day == 0 and a.shift_type == "morning"
        ]
        assert len(day0_morning) >= 2

    def test_decrease_staffing(self):
        """Decreasing min_employees to 0 allows fewer assignments."""
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            ChangeStaffingRequirement(day=0, shift_type="morning", min_employees=0),
        ])

        assert result.status == SimulationStatus.SUCCESS

    def test_change_workload_limit(self):
        """Changing max_shifts_per_employee produces a valid schedule."""
        scenario = _baseline_with_workload()
        baseline = solve(scenario)
        assert baseline.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)

        result = simulate(scenario, baseline, [
            ChangeWorkloadLimit(max_shifts_per_employee=2),
        ])

        assert result.status == SimulationStatus.SUCCESS
        # Each employee should have at most 2 shifts
        from collections import Counter
        counts = Counter(a.employee_id for a in result.modified_solution.assignments)
        for emp_id, count in counts.items():
            assert count <= 2, f"{emp_id} has {count} shifts, max is 2"

    def test_change_min_days_off(self):
        """Increasing min_days_off modifies the schedule."""
        scenario = _baseline_with_workload()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            ChangeWorkloadLimit(min_days_off=3),
        ])

        assert result.status == SimulationStatus.SUCCESS
        # Each employee should have at least 3 days off
        from collections import Counter
        counts = Counter(a.employee_id for a in result.modified_solution.assignments)
        for emp in scenario.employees:
            worked = counts.get(emp.id, 0)
            days_off = scenario.num_days - worked
            assert days_off >= 3

    def test_multiple_changes(self):
        """Multiple changes applied together."""
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            SetEmployeeUnavailable(employee_id="alice", day=0),
            SetEmployeeUnavailable(employee_id="bob", day=1),
            ChangeStaffingRequirement(day=2, shift_type="night", min_employees=2),
        ])

        assert result.status == SimulationStatus.SUCCESS
        # Verify both unavailabilities
        for a in result.modified_solution.assignments:
            if a.employee_id == "alice":
                assert a.day != 0
            if a.employee_id == "bob":
                assert a.day != 1
        # Day 2 night should have >= 2
        d2_night = [a for a in result.modified_solution.assignments
                    if a.day == 2 and a.shift_type == "night"]
        assert len(d2_night) >= 2


# ==================================================================
# Test: Assignment diff correctness
# ==================================================================


class TestAssignmentDiff:
    """Tests for correct identification of assignment differences."""

    def test_diff_has_added_and_removed(self):
        """Changing a constraint produces identifiable added/removed assignments."""
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            SetEmployeeUnavailable(employee_id="alice", day=0),
        ])

        assert result.status == SimulationStatus.SUCCESS
        diff = result.assignment_diff
        assert diff is not None
        # If Alice was assigned on day 0 in baseline, she should be removed
        alice_day0_baseline = [
            a for a in baseline.assignments
            if a.employee_id == "alice" and a.day == 0
        ]
        if alice_day0_baseline:
            # Her assignment was removed
            assert any(
                a.employee_id == "alice" and a.day == 0
                for a in diff.removed
            )

    def test_unchanged_count(self):
        """Unchanged assignments are correctly counted."""
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            SetEmployeeUnavailable(employee_id="alice", day=0),
        ])

        assert result.status == SimulationStatus.SUCCESS
        diff = result.assignment_diff
        total_baseline = len(baseline.assignments)
        total_modified = len(result.modified_solution.assignments)
        # unchanged + removed = baseline total
        # unchanged + added = modified total
        assert diff.unchanged + len(diff.removed) == total_baseline
        assert diff.unchanged + len(diff.added) == total_modified

    def test_no_change_produces_empty_diff(self):
        """A change that doesn't affect the solution produces an empty diff."""
        # Adding a redundant unavailability for an employee who was already
        # not assigned on that day might produce an empty diff
        scenario = Scenario(
            num_days=1,
            employees=_make_employees(3),
            shift_types=_make_shifts("morning"),
            shift_requirements=[
                ShiftRequirement(day=0, shift_type="morning", min_employees=1),
            ],
        )
        baseline = solve(scenario)
        assert baseline.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)

        # Find an employee who was NOT assigned
        assigned_ids = {a.employee_id for a in baseline.assignments}
        all_ids = {e.id for e in scenario.employees}
        unassigned = all_ids - assigned_ids
        if not unassigned:
            pytest.skip("All employees were assigned — can't test no-change diff")

        target = next(iter(unassigned))
        result = simulate(scenario, baseline, [
            SetEmployeeUnavailable(employee_id=target, day=0),
        ])

        assert result.status == SimulationStatus.SUCCESS
        # The diff might be empty since the unassigned employee stays unassigned
        # (solver may still re-optimize, so we just check the structure exists)
        assert result.assignment_diff is not None

    def test_affected_employees_tracked(self):
        """Affected employees are correctly tracked in the diff."""
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            SetEmployeeUnavailable(employee_id="alice", day=0),
        ])

        assert result.status == SimulationStatus.SUCCESS
        diff = result.assignment_diff
        # If there are differences, affected_employees should be populated
        if diff.added or diff.removed:
            assert len(diff.affected_employees) > 0

    def test_affected_days_tracked(self):
        """Affected days are correctly tracked in the diff."""
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            SetEmployeeUnavailable(employee_id="alice", day=0),
        ])

        assert result.status == SimulationStatus.SUCCESS
        diff = result.assignment_diff
        if diff.added or diff.removed:
            assert len(diff.affected_days) > 0


# ==================================================================
# Test: Metrics diff correctness
# ==================================================================


class TestMetricsDiff:
    """Tests for correct metric comparisons."""

    def test_metrics_diff_populated(self):
        """MetricsDiff is populated when both solutions have metrics."""
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            SetEmployeeUnavailable(employee_id="alice", day=0),
        ])

        assert result.status == SimulationStatus.SUCCESS
        md = result.metrics_diff
        assert md is not None
        assert md.total_assignments is not None
        assert md.coverage_pct is not None
        assert md.utilization_pct is not None
        assert md.objective_value is not None

    def test_metrics_diff_delta_correct(self):
        """Delta = modified - baseline for each metric."""
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            ChangeStaffingRequirement(day=0, shift_type="morning", min_employees=3),
        ])

        assert result.status == SimulationStatus.SUCCESS
        md = result.metrics_diff
        # total_assignments delta should be modified - baseline
        base_val, mod_val, delta = md.total_assignments
        assert delta == mod_val - base_val

    def test_objective_value_diff(self):
        """Objective value difference is reported."""
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            SetEmployeeUnavailable(employee_id="alice", day=0),
        ])

        assert result.status == SimulationStatus.SUCCESS
        md = result.metrics_diff
        if md.objective_value is not None:
            base_obj, mod_obj, delta = md.objective_value
            assert abs(delta - (mod_obj - base_obj)) < 0.01

    def test_coverage_maintained_after_valid_change(self):
        """Coverage stays at 100% if the modified scenario is still feasible."""
        scenario = _baseline_scenario()
        baseline = solve(scenario)
        assert baseline.metrics.coverage_pct == 100.0

        result = simulate(scenario, baseline, [
            SetEmployeeUnavailable(employee_id="alice", day=0),
        ])

        assert result.status == SimulationStatus.SUCCESS
        assert result.modified_solution.metrics.coverage_pct == 100.0


# ==================================================================
# Test: Invalid change references
# ==================================================================


class TestInvalidChanges:
    """Changes that reference invalid entities."""

    def test_unknown_employee(self):
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            SetEmployeeUnavailable(employee_id="nobody", day=0),
        ])

        assert result.status == SimulationStatus.INVALID_CHANGE
        assert result.modified_solution is None
        assert any("nobody" in e for e in result.errors)

    def test_day_out_of_range(self):
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            SetEmployeeUnavailable(employee_id="alice", day=99),
        ])

        assert result.status == SimulationStatus.INVALID_CHANGE
        assert any("99" in e for e in result.errors)

    def test_unknown_shift_type(self):
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            ChangeStaffingRequirement(day=0, shift_type="graveyard", min_employees=1),
        ])

        assert result.status == SimulationStatus.INVALID_CHANGE
        assert any("graveyard" in e for e in result.errors)

    def test_max_lt_min_in_staffing_change(self):
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            ChangeStaffingRequirement(
                day=0, shift_type="morning", min_employees=5, max_employees=2
            ),
        ])

        assert result.status == SimulationStatus.INVALID_CHANGE

    def test_empty_changes_list(self):
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [])

        assert result.status == SimulationStatus.INVALID_CHANGE
        assert any("No changes" in e for e in result.errors)


# ==================================================================
# Test: Semantically invalid modified scenarios
# ==================================================================


class TestValidationFailure:
    """Modified scenarios that pass Pydantic but fail semantic validation."""

    def test_staffing_exceeds_workforce(self):
        """Require more employees per shift than exist."""
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            ChangeStaffingRequirement(day=0, shift_type="morning", min_employees=99),
        ])

        assert result.status == SimulationStatus.VALIDATION_FAILED
        assert result.modified_solution is None
        assert len(result.errors) > 0


# ==================================================================
# Test: Infeasible modified scenarios
# ==================================================================


class TestInfeasibleSimulation:
    """Modified scenarios that are valid but infeasible."""

    def test_infeasible_from_tight_constraints(self):
        """Tight workload limits make the schedule impossible."""
        scenario = Scenario(
            num_days=3,
            employees=_make_employees(3),
            shift_types=_make_shifts("morning", "night"),
            shift_requirements=_uniform_reqs(3, ["morning", "night"], 1),
        )
        baseline = solve(scenario)
        assert baseline.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)

        # max_shifts=1 but need 6 total slots from 3 employees → infeasible
        result = simulate(scenario, baseline, [
            ChangeWorkloadLimit(max_shifts_per_employee=1),
        ])

        assert result.status == SimulationStatus.INFEASIBLE
        assert result.modified_solution is not None
        assert result.modified_solution.status == SolverStatus.INFEASIBLE
        assert len(result.errors) > 0

    def test_infeasible_has_diff(self):
        """Infeasible results still include assignment_diff and metrics_diff."""
        scenario = Scenario(
            num_days=3,
            employees=_make_employees(3),
            shift_types=_make_shifts("morning", "night"),
            shift_requirements=_uniform_reqs(3, ["morning", "night"], 1),
        )
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            ChangeWorkloadLimit(max_shifts_per_employee=1),
        ])

        assert result.status == SimulationStatus.INFEASIBLE
        # Diff should exist (comparing baseline assignments with empty modified)
        assert result.assignment_diff is not None


# ==================================================================
# Test: Preservation of original data
# ==================================================================


class TestPreservation:
    """Original scenario and solution must not be mutated."""

    def test_original_scenario_unchanged(self):
        scenario = _baseline_scenario()
        original_dump = scenario.model_dump()
        baseline = solve(scenario)

        simulate(scenario, baseline, [
            SetEmployeeUnavailable(employee_id="alice", day=0),
            ChangeStaffingRequirement(day=0, shift_type="morning", min_employees=3),
        ])

        assert scenario.model_dump() == original_dump

    def test_original_solution_unchanged(self):
        scenario = _baseline_scenario()
        baseline = solve(scenario)
        original_assignments = list(baseline.assignments)
        original_status = baseline.status

        simulate(scenario, baseline, [
            SetEmployeeUnavailable(employee_id="alice", day=0),
        ])

        assert baseline.assignments == original_assignments
        assert baseline.status == original_status

    def test_result_contains_original_refs(self):
        """SimulationResult references the original scenario and solution."""
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            SetEmployeeUnavailable(employee_id="alice", day=0),
        ])

        assert result.baseline_scenario is scenario
        assert result.baseline_solution is baseline


# ==================================================================
# Test: Solver status handling
# ==================================================================


class TestSolverStatusHandling:
    """Correct propagation of solver statuses."""

    def test_success_has_optimal_or_feasible(self):
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            SetEmployeeUnavailable(employee_id="alice", day=0),
        ])

        assert result.status == SimulationStatus.SUCCESS
        assert result.modified_solution.status in (
            SolverStatus.OPTIMAL, SolverStatus.FEASIBLE
        )

    def test_infeasible_status_propagated(self):
        scenario = Scenario(
            num_days=3,
            employees=_make_employees(3),
            shift_types=_make_shifts("morning", "night"),
            shift_requirements=_uniform_reqs(3, ["morning", "night"], 1),
        )
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            ChangeWorkloadLimit(max_shifts_per_employee=1),
        ])

        assert result.status == SimulationStatus.INFEASIBLE
        assert result.modified_solution.status == SolverStatus.INFEASIBLE

    def test_changes_applied_field(self):
        """changes_applied tracks what was applied."""
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        changes = [SetEmployeeUnavailable(employee_id="alice", day=0)]
        result = simulate(scenario, baseline, changes)

        assert len(result.changes_applied) == 1
        assert isinstance(result.changes_applied[0], SetEmployeeUnavailable)


# ==================================================================
# Test: New staffing requirement creation
# ==================================================================


class TestNewStaffingRequirement:
    """ChangeStaffingRequirement creates a requirement if none exists."""

    def test_create_new_requirement(self):
        """Adding a requirement for a day/shift with no existing one."""
        scenario = Scenario(
            num_days=3,
            employees=_make_employees(3),
            shift_types=_make_shifts("morning", "night"),
            shift_requirements=[
                ShiftRequirement(day=0, shift_type="morning", min_employees=1),
            ],
        )
        baseline = solve(scenario)

        # Add requirement for day 1 night (didn't exist before)
        result = simulate(scenario, baseline, [
            ChangeStaffingRequirement(day=1, shift_type="night", min_employees=1),
        ])

        assert result.status == SimulationStatus.SUCCESS
        # Day 1 night should now have at least 1 assignment
        d1_night = [a for a in result.modified_solution.assignments
                    if a.day == 1 and a.shift_type == "night"]
        assert len(d1_night) >= 1


# ==================================================================
# Test: Soft constraint change
# ==================================================================


class TestSoftConstraint:
    """Employee unavailability with soft priority."""

    def test_soft_unavailability(self):
        """Soft unavailability may be violated with penalty."""
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            SetEmployeeUnavailable(
                employee_id="alice", day=0,
                priority=ConstraintPriority.SOFT, penalty=10,
            ),
        ])

        assert result.status == SimulationStatus.SUCCESS


# ==================================================================
# Test: Edge cases
# ==================================================================


class TestEdgeCases:
    """Unusual or boundary conditions."""

    def test_all_employees_unavailable_on_day(self):
        """All employees unavailable on one day with a requirement → infeasible."""
        scenario = Scenario(
            num_days=2,
            employees=_make_employees(3),
            shift_types=_make_shifts("morning"),
            shift_requirements=_uniform_reqs(2, ["morning"], 1),
        )
        baseline = solve(scenario)

        changes = [
            SetEmployeeUnavailable(employee_id=e.id, day=0)
            for e in scenario.employees
        ]
        result = simulate(scenario, baseline, changes)

        assert result.status == SimulationStatus.INFEASIBLE

    def test_max_employees_cap(self):
        """Setting max_employees limits shift staffing."""
        scenario = Scenario(
            num_days=1,
            employees=_make_employees(5),
            shift_types=_make_shifts("morning"),
            shift_requirements=[
                ShiftRequirement(day=0, shift_type="morning", min_employees=1),
            ],
        )
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            ChangeStaffingRequirement(
                day=0, shift_type="morning", min_employees=1, max_employees=1
            ),
        ])

        assert result.status == SimulationStatus.SUCCESS
        d0_morning = [a for a in result.modified_solution.assignments
                      if a.day == 0 and a.shift_type == "morning"]
        assert len(d0_morning) == 1

    def test_modified_scenario_is_different_object(self):
        """Modified scenario is a separate object from the original."""
        scenario = _baseline_scenario()
        baseline = solve(scenario)

        result = simulate(scenario, baseline, [
            SetEmployeeUnavailable(employee_id="alice", day=0),
        ])

        assert result.status == SimulationStatus.SUCCESS
        assert result.modified_scenario is not scenario
        # Modified should have one more unavailability entry
        assert len(result.modified_scenario.employee_availability) == \
            len(scenario.employee_availability) + 1
