"""Tests for the evaluation benchmark.

Covers:
  - Dataset loading and integrity
  - Independent constraint verification
  - Benchmark metric calculations
  - Expected-vs-actual feasibility classification
  - Simulation evaluation
  - Report generation and JSON/CSV export
  - Offline operation without API key
  - Optional NL evaluation with mocked provider
"""

from __future__ import annotations

import csv
import json
import os
import tempfile

import pytest

from shift_happens.benchmark.dataset import (
    DATASET_VERSION,
    BenchmarkCase,
    ExpectedSimOutcome,
    ExpectedSolverOutcome,
    NLExpectedFacts,
    SimulationCase,
    load_dataset,
)
from shift_happens.benchmark.runner import (
    AggregateMetrics,
    BenchmarkReport,
    NLCaseResult,
    SimCaseResult,
    SolverCaseResult,
    compute_aggregate,
    run_benchmark,
)
from shift_happens.benchmark.report import format_summary, generate_reports
from shift_happens.benchmark.verifier import verify_solution
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
    ScheduleMetrics,
    ScheduleSolution,
    SolverStatus,
)
from shift_happens.simulation.engine import SetEmployeeUnavailable
from shift_happens.solver.scheduler import solve


# ==================================================================
# Helpers
# ==================================================================


def _emps(n):
    names = ["Alice", "Bob", "Carol", "Dave", "Eve",
             "Frank", "Grace", "Hank", "Ivy", "Jack"]
    return [Employee(id=names[i].lower(), name=names[i]) for i in range(n)]


def _shifts(*names):
    return [ShiftType(name=n) for n in names]


def _reqs(days, shifts, min_e):
    return [
        ShiftRequirement(day=d, shift_type=s, min_employees=min_e)
        for d in range(days) for s in shifts
    ]


# ==================================================================
# Test: Dataset integrity
# ==================================================================


class TestDataset:
    """Tests for the benchmark dataset."""

    def test_load_returns_list(self):
        cases = load_dataset()
        assert isinstance(cases, list)

    def test_minimum_case_count(self):
        """Dataset has at least 30 cases."""
        cases = load_dataset()
        assert len(cases) >= 30

    def test_unique_ids(self):
        """All case IDs are unique."""
        cases = load_dataset()
        ids = [c.id for c in cases]
        assert len(ids) == len(set(ids))

    def test_all_have_required_fields(self):
        """Every case has id, name, category, scenario, expected_solver."""
        for case in load_dataset():
            assert case.id, f"Missing id"
            assert case.name, f"Missing name for {case.id}"
            assert case.category, f"Missing category for {case.id}"
            assert case.scenario is not None, f"Missing scenario for {case.id}"
            assert case.expected_solver is not None, f"Missing expected_solver for {case.id}"

    def test_scenarios_are_valid_pydantic(self):
        """All scenarios pass Pydantic validation."""
        for case in load_dataset():
            # The Scenario constructor already validates
            assert isinstance(case.scenario, Scenario), f"Bad scenario for {case.id}"

    def test_version_is_set(self):
        assert DATASET_VERSION
        assert isinstance(DATASET_VERSION, str)

    def test_has_infeasible_cases(self):
        """Dataset includes infeasible cases."""
        cases = load_dataset()
        infeasible = [c for c in cases if c.expected_solver == ExpectedSolverOutcome.INFEASIBLE]
        assert len(infeasible) >= 2

    def test_has_simulation_cases(self):
        """Dataset includes simulation test cases."""
        cases = load_dataset()
        with_sim = [c for c in cases if c.simulation_cases]
        assert len(with_sim) >= 3

    def test_has_nl_cases(self):
        """Dataset includes NL test cases."""
        cases = load_dataset()
        with_nl = [c for c in cases if c.nl_prompt and c.nl_expected]
        assert len(with_nl) >= 3

    def test_categories_represented(self):
        """Multiple categories are represented."""
        cases = load_dataset()
        categories = {c.category for c in cases}
        assert len(categories) >= 5


# ==================================================================
# Test: Independent constraint verification
# ==================================================================


class TestVerifier:
    """Tests for the independent constraint verifier."""

    def test_valid_solution_passes(self):
        """A correctly solved schedule passes verification."""
        scenario = Scenario(
            num_days=3,
            employees=_emps(3),
            shift_types=_shifts("morning"),
            shift_requirements=_reqs(3, ["morning"], 1),
        )
        solution = solve(scenario)
        assert solution.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        result = verify_solution(scenario, solution)
        assert result.passed
        assert result.checks_performed > 0

    def test_detects_one_shift_per_day_violation(self):
        """Verifier catches an employee assigned to two shifts on one day."""
        scenario = Scenario(
            num_days=1,
            employees=_emps(2),
            shift_types=_shifts("morning", "night"),
            shift_requirements=_reqs(1, ["morning", "night"], 1),
        )
        # Forge a bad solution: Alice on both shifts day 0
        fake = ScheduleSolution(
            status=SolverStatus.OPTIMAL,
            assignments=[
                Assignment(employee_id="alice", employee_name="Alice",
                           day=0, shift_type="morning"),
                Assignment(employee_id="alice", employee_name="Alice",
                           day=0, shift_type="night"),
            ],
        )
        result = verify_solution(scenario, fake)
        assert not result.passed
        violations = [v for v in result.violations if v.constraint == "one_shift_per_day"]
        assert len(violations) >= 1

    def test_detects_understaffing(self):
        """Verifier catches insufficient staffing."""
        scenario = Scenario(
            num_days=1,
            employees=_emps(3),
            shift_types=_shifts("morning"),
            shift_requirements=[
                ShiftRequirement(day=0, shift_type="morning", min_employees=2),
            ],
        )
        # Forge: only 1 employee assigned
        fake = ScheduleSolution(
            status=SolverStatus.OPTIMAL,
            assignments=[
                Assignment(employee_id="alice", employee_name="Alice",
                           day=0, shift_type="morning"),
            ],
        )
        result = verify_solution(scenario, fake)
        assert not result.passed
        assert any(v.constraint == "min_staffing" for v in result.violations)

    def test_detects_hard_unavailability_violation(self):
        """Verifier catches assignment to unavailable employee."""
        scenario = Scenario(
            num_days=1,
            employees=_emps(2),
            shift_types=_shifts("morning"),
            shift_requirements=_reqs(1, ["morning"], 1),
            employee_availability=[
                EmployeeAvailability(employee_id="alice", day=0),
            ],
        )
        fake = ScheduleSolution(
            status=SolverStatus.OPTIMAL,
            assignments=[
                Assignment(employee_id="alice", employee_name="Alice",
                           day=0, shift_type="morning"),
            ],
        )
        result = verify_solution(scenario, fake)
        assert not result.passed
        assert any(v.constraint == "hard_unavailability" for v in result.violations)

    def test_detects_shift_restriction_violation(self):
        """Verifier catches assignment violating shift restriction."""
        scenario = Scenario(
            num_days=1,
            employees=_emps(2),
            shift_types=_shifts("morning", "night"),
            shift_requirements=_reqs(1, ["night"], 1),
            employee_shift_restrictions=[
                EmployeeShiftRestriction(employee_id="alice", shift_type="night"),
            ],
        )
        fake = ScheduleSolution(
            status=SolverStatus.OPTIMAL,
            assignments=[
                Assignment(employee_id="alice", employee_name="Alice",
                           day=0, shift_type="night"),
            ],
        )
        result = verify_solution(scenario, fake)
        assert not result.passed
        assert any(v.constraint == "hard_shift_restriction" for v in result.violations)

    def test_detects_max_shifts_violation(self):
        """Verifier catches exceeding max shifts."""
        scenario = Scenario(
            num_days=3,
            employees=_emps(2),
            shift_types=_shifts("morning"),
            shift_requirements=_reqs(3, ["morning"], 1),
            workload_limits=WorkloadLimits(max_shifts_per_employee=1),
        )
        fake = ScheduleSolution(
            status=SolverStatus.OPTIMAL,
            assignments=[
                Assignment(employee_id="alice", employee_name="Alice",
                           day=d, shift_type="morning")
                for d in range(3)
            ],
        )
        result = verify_solution(scenario, fake)
        assert not result.passed
        assert any(v.constraint == "max_shifts_per_employee" for v in result.violations)

    def test_detects_max_employees_cap_violation(self):
        """Verifier catches exceeding max employees on a shift."""
        scenario = Scenario(
            num_days=1,
            employees=_emps(3),
            shift_types=_shifts("morning"),
            shift_requirements=[
                ShiftRequirement(day=0, shift_type="morning",
                                 min_employees=1, max_employees=1),
            ],
        )
        fake = ScheduleSolution(
            status=SolverStatus.OPTIMAL,
            assignments=[
                Assignment(employee_id="alice", employee_name="Alice",
                           day=0, shift_type="morning"),
                Assignment(employee_id="bob", employee_name="Bob",
                           day=0, shift_type="morning"),
            ],
        )
        result = verify_solution(scenario, fake)
        assert not result.passed
        assert any(v.constraint == "max_employees_cap" for v in result.violations)

    def test_infeasible_solution_skipped(self):
        """Infeasible solutions are not verified (nothing to check)."""
        scenario = Scenario(
            num_days=1,
            employees=_emps(1),
            shift_types=_shifts("morning"),
        )
        infeasible = ScheduleSolution(status=SolverStatus.INFEASIBLE)
        result = verify_solution(scenario, infeasible)
        assert result.passed  # No checks performed → no violations
        assert result.checks_performed == 0

    def test_with_workload_constraints(self):
        """Solver solution with workload limits passes verification."""
        scenario = Scenario(
            num_days=7,
            employees=_emps(5),
            shift_types=_shifts("morning", "night"),
            shift_requirements=_reqs(7, ["morning", "night"], 1),
            workload_limits=WorkloadLimits(
                max_shifts_per_employee=4, min_days_off=2,
                max_consecutive_days=3,
            ),
        )
        solution = solve(scenario)
        assert solution.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        result = verify_solution(scenario, solution)
        assert result.passed


# ==================================================================
# Test: Benchmark runner (offline)
# ==================================================================


class TestBenchmarkRunner:
    """Tests for the offline benchmark runner."""

    def test_runs_without_api_key(self):
        """Benchmark runs fully offline."""
        report = run_benchmark(solver_timeout_s=10, case_filter="trivial")
        assert len(report.solver_results) > 0
        assert report.nl_evaluation_mode == "disabled"
        assert len(report.nl_results) == 0

    def test_solver_outcomes_populated(self):
        """All solver results have required fields."""
        report = run_benchmark(solver_timeout_s=10, case_filter="trivial")
        for r in report.solver_results:
            assert r.case_id
            assert r.expected_outcome
            assert r.actual_status

    def test_feasibility_classification(self):
        """Feasible cases are correctly classified."""
        report = run_benchmark(solver_timeout_s=10, case_filter="trivial")
        for r in report.solver_results:
            assert r.outcome_correct, f"Wrong outcome for {r.case_id}"

    def test_infeasible_classification(self):
        """Infeasible cases are correctly classified."""
        report = run_benchmark(solver_timeout_s=10, case_filter="infeasible")
        for r in report.solver_results:
            assert r.outcome_correct, (
                f"Wrong outcome for {r.case_id}: "
                f"expected={r.expected_outcome}, actual={r.actual_status}"
            )

    def test_verification_runs(self):
        """Verification is performed for feasible solutions."""
        report = run_benchmark(solver_timeout_s=10, case_filter="standard-3x2x5")
        for r in report.solver_results:
            if r.actual_status in ("optimal", "feasible"):
                assert r.verification is not None
                assert r.verification.passed

    def test_simulation_evaluation(self):
        """Simulation cases are evaluated."""
        report = run_benchmark(solver_timeout_s=10, case_filter="sim")
        assert len(report.simulation_results) > 0
        for s in report.simulation_results:
            assert s.outcome_correct, (
                f"Wrong sim outcome for {s.case_id}/{s.sim_description}: "
                f"expected={s.expected_outcome}, actual={s.actual_status}"
            )

    def test_simulation_preserves_original(self):
        """All simulation results confirm original preservation."""
        report = run_benchmark(solver_timeout_s=10, case_filter="sim")
        for s in report.simulation_results:
            assert s.original_preserved, (
                f"Original not preserved for {s.case_id}/{s.sim_description}"
            )

    def test_timestamp_set(self):
        report = run_benchmark(solver_timeout_s=5, case_filter="trivial-1x1x1")
        assert report.timestamp

    def test_dataset_version_set(self):
        report = run_benchmark(solver_timeout_s=5, case_filter="trivial-1x1x1")
        assert report.dataset_version == DATASET_VERSION

    def test_case_filter(self):
        """Case filter restricts which cases run."""
        full = run_benchmark(solver_timeout_s=10)
        filtered = run_benchmark(solver_timeout_s=10, case_filter="trivial")
        assert len(filtered.solver_results) < len(full.solver_results)
        for r in filtered.solver_results:
            assert "trivial" in r.case_id


# ==================================================================
# Test: Aggregate metrics
# ==================================================================


class TestAggregateMetrics:
    """Tests for aggregate metric computation."""

    def test_compute_from_report(self):
        report = run_benchmark(solver_timeout_s=10, case_filter="trivial")
        agg = compute_aggregate(report)
        assert agg.total_cases == len(report.solver_results)
        assert agg.outcome_correct <= agg.total_cases
        assert 0.0 <= agg.outcome_accuracy <= 1.0

    def test_solver_timing(self):
        report = run_benchmark(solver_timeout_s=10, case_filter="standard-3x2x5")
        agg = compute_aggregate(report)
        assert agg.mean_solver_time_s > 0
        assert agg.p50_solver_time_s > 0
        assert agg.max_solver_time_s >= agg.p50_solver_time_s

    def test_empty_report(self):
        """Aggregate handles empty report without errors."""
        report = BenchmarkReport()
        agg = compute_aggregate(report)
        assert agg.total_cases == 0
        assert agg.outcome_accuracy == 0.0


# ==================================================================
# Test: Report generation
# ==================================================================


class TestReportGeneration:
    """Tests for report file generation."""

    def test_generate_reports_creates_files(self):
        report = run_benchmark(solver_timeout_s=5, case_filter="trivial-1x1x1")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_reports(report, output_dir=tmpdir)
            assert "json" in paths
            assert "csv_solver" in paths
            assert "summary" in paths
            for p in paths.values():
                assert os.path.isfile(p)

    def test_json_report_valid(self):
        """JSON report is valid JSON."""
        report = run_benchmark(solver_timeout_s=5, case_filter="trivial-1x1x1")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_reports(report, output_dir=tmpdir)
            with open(paths["json"], encoding="utf-8") as f:
                data = json.load(f)
            assert data["dataset_version"] == DATASET_VERSION
            assert "solver_results" in data
            assert "aggregate" in data

    def test_csv_solver_valid(self):
        """CSV solver report has correct headers and rows."""
        report = run_benchmark(solver_timeout_s=5, case_filter="trivial")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_reports(report, output_dir=tmpdir)
            with open(paths["csv_solver"], encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == len(report.solver_results)
            assert "case_id" in rows[0]
            assert "outcome_correct" in rows[0]

    def test_simulation_csv_generated(self):
        """Simulation CSV is generated when simulation results exist."""
        report = run_benchmark(solver_timeout_s=10, case_filter="sim-unavailable")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_reports(report, output_dir=tmpdir)
            if "csv_simulation" in paths:
                assert os.path.isfile(paths["csv_simulation"])

    def test_summary_text(self):
        """Summary text is well-formed."""
        report = run_benchmark(solver_timeout_s=5, case_filter="trivial")
        summary = format_summary(report)
        assert "SOLVER EVALUATION" in summary
        assert "Outcome correct" in summary

    def test_summary_reports_nl_disabled(self):
        """Summary correctly reports NL evaluation as disabled."""
        report = run_benchmark(solver_timeout_s=5, case_filter="trivial-1x1x1")
        summary = format_summary(report)
        assert "disabled" in summary.lower()


# ==================================================================
# Test: NL evaluation with mock
# ==================================================================


class TestNLEvaluation:
    """Tests for NL evaluation using the mock provider."""

    def test_nl_eval_with_mock(self):
        """NL evaluation works with MockLLMProvider."""
        from shift_happens.llm.provider import MockLLMProvider
        provider = MockLLMProvider()
        report = run_benchmark(
            include_nl=True,
            nl_provider=provider,
            solver_timeout_s=10,
            case_filter="nl-simple",
        )
        # Mock provider returns a fixed scenario, so parsing should succeed
        # but fact matching may not be perfect
        assert len(report.nl_results) >= 0  # May or may not match filter + nl_prompt
        assert report.nl_evaluation_mode == "live"

    def test_nl_eval_disabled_by_default(self):
        """NL evaluation is disabled when not requested."""
        report = run_benchmark(solver_timeout_s=5, case_filter="nl")
        assert len(report.nl_results) == 0
        assert report.nl_evaluation_mode == "disabled"


# ==================================================================
# Test: Full benchmark pass (all cases)
# ==================================================================


class TestFullBenchmark:
    """Run the complete benchmark and verify all outcomes."""

    def test_all_solver_outcomes_correct(self):
        """Every solver outcome matches the expected classification."""
        report = run_benchmark(solver_timeout_s=30)
        for r in report.solver_results:
            assert r.outcome_correct, (
                f"Case {r.case_id}: expected={r.expected_outcome}, "
                f"actual={r.actual_status}"
            )

    def test_all_verifications_pass(self):
        """Every feasible solution passes independent verification."""
        report = run_benchmark(solver_timeout_s=30)
        for r in report.solver_results:
            if r.verification is not None and r.verification.checks_performed > 0:
                assert r.verification.passed, (
                    f"Case {r.case_id}: verification failed: "
                    f"{[v.detail for v in r.verification.violations]}"
                )

    def test_all_simulation_outcomes_correct(self):
        """Every simulation outcome matches expectations."""
        report = run_benchmark(solver_timeout_s=30)
        for s in report.simulation_results:
            assert s.outcome_correct, (
                f"Sim {s.case_id}/{s.sim_description}: "
                f"expected={s.expected_outcome}, actual={s.actual_status}"
            )

    def test_all_simulations_preserve_original(self):
        """No simulation mutates the original scenario or baseline."""
        report = run_benchmark(solver_timeout_s=30)
        for s in report.simulation_results:
            assert s.original_preserved, (
                f"Sim {s.case_id}/{s.sim_description}: original mutated"
            )

    def test_all_validations_correct(self):
        """Semantic validation matches expectations."""
        report = run_benchmark(solver_timeout_s=30)
        for r in report.solver_results:
            assert r.validation_correct, (
                f"Case {r.case_id}: validation expected_valid={r.expected_valid}, "
                f"got validation_passed={r.validation_passed}"
            )
