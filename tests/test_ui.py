"""Tests for the explainer module and UI helper logic.

Tests the grounded explanation generator and the data transformation
functions used by the UI.  No LLM calls, no Streamlit server.
"""

from __future__ import annotations

import pytest

from shift_happens.explanation.explainer import (
    describe_change,
    explain_schedule,
    explain_simulation,
)
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
from shift_happens.models.solution import (
    Assignment,
    EmployeeMetrics,
    ScheduleMetrics,
    ScheduleSolution,
    SolverStatus,
)
from shift_happens.simulation.engine import (
    ChangeStaffingRequirement,
    ChangeWorkloadLimit,
    SetEmployeeUnavailable,
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


def _simple_scenario() -> Scenario:
    return Scenario(
        name="Test Scenario",
        num_days=3,
        employees=_make_employees(4),
        shift_types=_make_shifts("morning", "night"),
        shift_requirements=_uniform_reqs(3, ["morning", "night"], 1),
    )


def _solved_pair():
    """Return (scenario, solution) for a feasible schedule."""
    scenario = _simple_scenario()
    solution = solve(scenario)
    assert solution.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
    return scenario, solution


# ==================================================================
# Tests: explain_schedule
# ==================================================================


class TestExplainSchedule:
    """Tests for grounded schedule explanation."""

    def test_returns_string(self):
        scenario, solution = _solved_pair()
        result = explain_schedule(scenario, solution)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_scenario_name(self):
        scenario, solution = _solved_pair()
        result = explain_schedule(scenario, solution)
        assert "Test Scenario" in result

    def test_contains_status(self):
        scenario, solution = _solved_pair()
        result = explain_schedule(scenario, solution)
        assert "Optimal" in result or "Feasible" in result

    def test_contains_employee_count(self):
        scenario, solution = _solved_pair()
        result = explain_schedule(scenario, solution)
        assert "4" in result  # 4 employees

    def test_contains_day_count(self):
        scenario, solution = _solved_pair()
        result = explain_schedule(scenario, solution)
        assert "3" in result  # 3 days

    def test_contains_metrics(self):
        scenario, solution = _solved_pair()
        result = explain_schedule(scenario, solution)
        assert "assignments" in result.lower() or "Assignments" in result
        assert "Coverage" in result or "coverage" in result

    def test_contains_workload_info(self):
        scenario, solution = _solved_pair()
        result = explain_schedule(scenario, solution)
        # Should mention workload distribution
        assert "Workload" in result or "workload" in result

    def test_infeasible_explanation(self):
        """Infeasible schedules should mention infeasibility."""
        scenario = Scenario(
            num_days=1,
            employees=_make_employees(1),
            shift_types=_make_shifts("morning", "night"),
            shift_requirements=[
                ShiftRequirement(day=0, shift_type="morning", min_employees=1),
                ShiftRequirement(day=0, shift_type="night", min_employees=1),
            ],
        )
        solution = solve(scenario)
        result = explain_schedule(scenario, solution)
        assert "infeasible" in result.lower() or "Infeasible" in result or "No feasible" in result

    def test_with_workload_limits(self):
        """Explanation mentions workload limits when configured."""
        scenario = Scenario(
            name="Limited",
            num_days=5,
            employees=_make_employees(5),
            shift_types=_make_shifts("morning"),
            shift_requirements=_uniform_reqs(5, ["morning"], 1),
            workload_limits=WorkloadLimits(max_shifts_per_employee=3, min_days_off=1),
        )
        solution = solve(scenario)
        result = explain_schedule(scenario, solution)
        assert "Max shifts" in result or "max_shifts" in result or "3" in result

    def test_with_restrictions(self):
        """Explanation notes shift restrictions."""
        scenario = Scenario(
            num_days=3,
            employees=_make_employees(3),
            shift_types=_make_shifts("morning", "night"),
            shift_requirements=_uniform_reqs(3, ["morning", "night"], 1),
            employee_shift_restrictions=[
                EmployeeShiftRestriction(employee_id="alice", shift_type="night"),
            ],
        )
        solution = solve(scenario)
        result = explain_schedule(scenario, solution)
        assert "restriction" in result.lower()


# ==================================================================
# Tests: explain_simulation
# ==================================================================


class TestExplainSimulation:
    """Tests for simulation comparison explanations."""

    def test_returns_string(self):
        scenario, baseline = _solved_pair()
        # Solve a modified scenario
        modified_scenario = Scenario(
            name="Modified",
            num_days=3,
            employees=_make_employees(4),
            shift_types=_make_shifts("morning", "night"),
            shift_requirements=_uniform_reqs(3, ["morning", "night"], 1),
            employee_availability=[
                EmployeeAvailability(employee_id="alice", day=0),
            ],
        )
        modified_solution = solve(modified_scenario)
        result = explain_simulation(
            scenario, baseline,
            modified_scenario, modified_solution,
            ["Make alice unavailable on day 0"],
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_changes(self):
        scenario, baseline = _solved_pair()
        modified_solution = solve(scenario)
        result = explain_simulation(
            scenario, baseline,
            scenario, modified_solution,
            ["Change A", "Change B"],
        )
        assert "Change A" in result
        assert "Change B" in result

    def test_contains_metrics_comparison(self):
        scenario, baseline = _solved_pair()
        modified_solution = solve(scenario)
        result = explain_simulation(
            scenario, baseline,
            scenario, modified_solution,
            ["Test change"],
        )
        assert "Assignments" in result
        assert "Coverage" in result


# ==================================================================
# Tests: describe_change
# ==================================================================


class TestDescribeChange:
    """Tests for human-readable change descriptions."""

    def test_employee_unavailable(self):
        change = SetEmployeeUnavailable(employee_id="alice", day=2)
        result = describe_change(change)
        assert "alice" in result
        assert "day 2" in result
        assert "hard" in result

    def test_employee_unavailable_soft(self):
        change = SetEmployeeUnavailable(
            employee_id="bob", day=1,
            priority=ConstraintPriority.SOFT, penalty=10,
        )
        result = describe_change(change)
        assert "bob" in result
        assert "soft" in result

    def test_staffing_change(self):
        change = ChangeStaffingRequirement(
            day=0, shift_type="morning", min_employees=3,
        )
        result = describe_change(change)
        assert "morning" in result
        assert "day 0" in result
        assert "min=3" in result

    def test_staffing_change_with_max(self):
        change = ChangeStaffingRequirement(
            day=1, shift_type="night", min_employees=1, max_employees=4,
        )
        result = describe_change(change)
        assert "max=4" in result

    def test_workload_change(self):
        change = ChangeWorkloadLimit(max_shifts_per_employee=3, min_days_off=2)
        result = describe_change(change)
        assert "max_shifts=3" in result
        assert "min_days_off=2" in result


# ==================================================================
# Tests: UI data transformations
# ==================================================================


class TestUIDataTransformations:
    """Test the helper functions used by the UI (imported separately)."""

    def test_assignments_to_df(self):
        """The pivot function produces a proper grid."""
        from shift_happens.ui.app import _assignments_to_df

        scenario, solution = _solved_pair()
        df = _assignments_to_df(scenario, solution)

        # Should have num_days rows
        assert len(df) == scenario.num_days
        # Should have columns for each shift type
        for st_name in [s.name for s in scenario.shift_types]:
            assert st_name in df.columns

    def test_assignments_to_df_empty(self):
        """An empty assignment list produces a grid with dashes."""
        from shift_happens.ui.app import _assignments_to_df

        scenario = _simple_scenario()
        solution = ScheduleSolution(
            status=SolverStatus.INFEASIBLE,
            assignments=[],
        )
        df = _assignments_to_df(scenario, solution)
        assert len(df) == scenario.num_days
        # All cells should be "—"
        for col in df.columns:
            for val in df[col]:
                assert val == "—"


# ==================================================================
# Tests: Streamlit entry point
# ==================================================================


class TestStreamlitEntryPoint:
    """Verify the app module can be imported and main() exists."""

    def test_import(self):
        import shift_happens.ui.app as app_module
        assert hasattr(app_module, "main")
        assert callable(app_module.main)

    def test_init_state_function_exists(self):
        from shift_happens.ui.app import _init_state
        assert callable(_init_state)


# ==================================================================
# Tests: Integration — explainer with real solver
# ==================================================================


class TestExplainerIntegration:
    """End-to-end: build scenario → solve → explain."""

    def test_full_pipeline_explanation(self):
        """Explain a fully solved scenario."""
        scenario = Scenario(
            name="Integration Test",
            num_days=5,
            employees=_make_employees(5),
            shift_types=_make_shifts("morning", "afternoon", "night"),
            shift_requirements=_uniform_reqs(5, ["morning", "afternoon", "night"], 1),
            workload_limits=WorkloadLimits(max_shifts_per_employee=4, min_days_off=1),
        )
        solution = solve(scenario)
        assert solution.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)

        explanation = explain_schedule(scenario, solution)
        assert "Integration Test" in explanation
        assert "5" in explanation  # 5 employees or 5 days
        assert "3" in explanation  # 3 shift types
        assert "Coverage" in explanation or "coverage" in explanation

    def test_simulation_explanation_integration(self):
        """Explain a simulated what-if comparison."""
        from shift_happens.simulation.engine import simulate, SimulationStatus

        scenario = _simple_scenario()
        baseline = solve(scenario)
        assert baseline.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)

        changes = [SetEmployeeUnavailable(employee_id="alice", day=0)]
        sim_result = simulate(scenario, baseline, changes)
        assert sim_result.status == SimulationStatus.SUCCESS

        explanation = explain_simulation(
            scenario, baseline,
            sim_result.modified_scenario, sim_result.modified_solution,
            [describe_change(c) for c in changes],
        )
        assert "alice" in explanation
        assert "Comparison" in explanation or "comparison" in explanation or "Baseline" in explanation
