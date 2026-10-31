"""Benchmark runner — evaluates solver, simulation, and optionally NL parsing.

The default mode is fully offline: no API key, no network access.
Live LLM evaluation must be explicitly enabled.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from shift_happens.benchmark.dataset import (
    DATASET_VERSION,
    BenchmarkCase,
    ExpectedSimOutcome,
    ExpectedSolverOutcome,
    NLExpectedFacts,
    SimulationCase,
    load_dataset,
)
from shift_happens.benchmark.verifier import VerificationResult, verify_solution
from shift_happens.models.scenario import Scenario
from shift_happens.models.solution import ScheduleSolution, SolverStatus
from shift_happens.simulation.engine import SimulationStatus, simulate
from shift_happens.solver.scheduler import solve
from shift_happens.validation.validator import validate_scenario


# ==================================================================
# Result types
# ==================================================================


@dataclass
class SolverCaseResult:
    """Result of evaluating one benchmark case against the solver."""
    case_id: str
    case_name: str
    category: str
    expected_outcome: str
    actual_status: str
    outcome_correct: bool
    validation_passed: bool
    expected_valid: bool
    validation_correct: bool
    solver_skipped: bool = False  # True when validation failed → solver not called
    verification: VerificationResult | None = None
    objective_value: float | None = None
    solver_time_s: float | None = None
    total_assignments: int | None = None
    coverage_pct: float | None = None
    utilization_pct: float | None = None
    soft_penalty: int | None = None
    workload_spread: int | None = None  # max_shifts - min_shifts across employees
    error: str | None = None


@dataclass
class SimCaseResult:
    """Result of evaluating one simulation sub-case."""
    case_id: str
    sim_description: str
    expected_outcome: str
    actual_status: str
    outcome_correct: bool
    original_preserved: bool = True
    has_assignment_diff: bool = False
    has_metrics_diff: bool = False
    error: str | None = None


@dataclass
class NLCaseResult:
    """Result of evaluating one NL parsing case."""
    case_id: str
    nl_prompt: str
    parse_success: bool
    scenario_valid: bool | None = None
    facts_checked: int = 0
    facts_matched: int = 0
    fact_details: dict[str, bool] = field(default_factory=dict)
    error: str | None = None


@dataclass
class BenchmarkReport:
    """Complete benchmark evaluation report."""
    dataset_version: str = DATASET_VERSION
    timestamp: str = ""
    solver_results: list[SolverCaseResult] = field(default_factory=list)
    simulation_results: list[SimCaseResult] = field(default_factory=list)
    nl_results: list[NLCaseResult] = field(default_factory=list)
    nl_evaluation_mode: str = "disabled"  # "disabled", "live", "mock"


# ==================================================================
# Runner
# ==================================================================


def run_benchmark(
    *,
    include_nl: bool = False,
    nl_provider=None,
    solver_timeout_s: float = 30.0,
    case_filter: str | None = None,
) -> BenchmarkReport:
    """Run the full benchmark evaluation.

    Args:
        include_nl: Enable NL parsing evaluation.
        nl_provider: An LLMProvider instance (real or mock) for NL eval.
        solver_timeout_s: Solver timeout per case.
        case_filter: If set, only run cases whose ID contains this string.

    Returns:
        A BenchmarkReport with all results.
    """
    dataset = load_dataset()
    if case_filter:
        dataset = [c for c in dataset if case_filter in c.id]

    report = BenchmarkReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    if include_nl and nl_provider is not None:
        report.nl_evaluation_mode = "live"
    elif include_nl:
        report.nl_evaluation_mode = "mock_required"

    # --- Solver evaluation ---
    for case in dataset:
        report.solver_results.append(
            _evaluate_solver(case, solver_timeout_s)
        )

    # --- Simulation evaluation ---
    for case in dataset:
        if case.simulation_cases:
            for sim_case in case.simulation_cases:
                report.simulation_results.append(
                    _evaluate_simulation(case, sim_case, solver_timeout_s)
                )

    # --- NL evaluation (optional) ---
    if include_nl and nl_provider is not None:
        for case in dataset:
            if case.nl_prompt and case.nl_expected:
                report.nl_results.append(
                    _evaluate_nl(case, nl_provider)
                )

    return report


# ==================================================================
# Solver evaluation
# ==================================================================


def _evaluate_solver(case: BenchmarkCase, timeout: float) -> SolverCaseResult:
    """Solve one benchmark case and compare against expectations."""
    # Semantic validation
    validation = validate_scenario(case.scenario)
    validation_passed = validation.valid
    validation_correct = validation_passed == case.expected_valid

    # If validation fails and we expected it to fail, don't solve
    if not validation_passed and not case.expected_valid:
        return SolverCaseResult(
            case_id=case.id,
            case_name=case.name,
            category=case.category,
            expected_outcome=case.expected_solver.value,
            actual_status="validation_rejected",
            outcome_correct=True,  # End-to-end: correctly identified as invalid
            validation_passed=False,
            expected_valid=case.expected_valid,
            validation_correct=True,
            solver_skipped=True,
        )

    # If validation fails unexpectedly, report but don't solve
    if not validation_passed and case.expected_valid:
        return SolverCaseResult(
            case_id=case.id,
            case_name=case.name,
            category=case.category,
            expected_outcome=case.expected_solver.value,
            actual_status="validation_rejected",
            outcome_correct=False,
            validation_passed=False,
            expected_valid=case.expected_valid,
            validation_correct=False,
            solver_skipped=True,
            error="Validation failed unexpectedly: " + "; ".join(validation.errors),
        )

    # Solve
    try:
        solution = solve(case.scenario, timeout_s=timeout)
    except Exception as exc:
        return SolverCaseResult(
            case_id=case.id,
            case_name=case.name,
            category=case.category,
            expected_outcome=case.expected_solver.value,
            actual_status="error",
            outcome_correct=False,
            validation_passed=validation_passed,
            expected_valid=case.expected_valid,
            validation_correct=validation_correct,
            error=f"{type(exc).__name__}: {exc}",
        )

    # Classify outcome
    actual = solution.status.value
    if case.expected_solver == ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE:
        outcome_correct = solution.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
    elif case.expected_solver == ExpectedSolverOutcome.INFEASIBLE:
        outcome_correct = solution.status == SolverStatus.INFEASIBLE
    else:
        outcome_correct = False

    # Independent verification
    verification = verify_solution(case.scenario, solution)

    # Metrics
    total_assignments = None
    coverage_pct = None
    utilization_pct = None
    soft_penalty = None
    workload_spread = None

    if solution.metrics:
        total_assignments = solution.metrics.total_assignments
        coverage_pct = solution.metrics.coverage_pct
        utilization_pct = solution.metrics.utilization_pct
        soft_penalty = solution.metrics.soft_penalty_total
        if solution.metrics.employee_metrics:
            shifts = [e.total_shifts for e in solution.metrics.employee_metrics]
            workload_spread = max(shifts) - min(shifts) if shifts else None

    return SolverCaseResult(
        case_id=case.id,
        case_name=case.name,
        category=case.category,
        expected_outcome=case.expected_solver.value,
        actual_status=actual,
        outcome_correct=outcome_correct,
        validation_passed=validation_passed,
        expected_valid=case.expected_valid,
        validation_correct=validation_correct,
        verification=verification,
        objective_value=solution.objective_value,
        solver_time_s=solution.solver_wall_time_s,
        total_assignments=total_assignments,
        coverage_pct=coverage_pct,
        utilization_pct=utilization_pct,
        soft_penalty=soft_penalty,
        workload_spread=workload_spread,
    )


# ==================================================================
# Simulation evaluation
# ==================================================================


def _evaluate_simulation(
    case: BenchmarkCase, sim_case: SimulationCase, timeout: float
) -> SimCaseResult:
    """Evaluate one simulation sub-case."""
    # First, solve the baseline
    try:
        baseline_solution = solve(case.scenario, timeout_s=timeout)
    except Exception as exc:
        return SimCaseResult(
            case_id=case.id,
            sim_description=sim_case.description,
            expected_outcome=sim_case.expected_outcome.value,
            actual_status="solver_error",
            outcome_correct=False,
            error=f"Baseline solve failed: {exc}",
        )

    if baseline_solution.status not in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
        return SimCaseResult(
            case_id=case.id,
            sim_description=sim_case.description,
            expected_outcome=sim_case.expected_outcome.value,
            actual_status="baseline_infeasible",
            outcome_correct=False,
            error="Cannot simulate — baseline is infeasible",
        )

    # Preserve check: save original dump
    original_dump = case.scenario.model_dump()
    original_assignments = list(baseline_solution.assignments)

    # Run simulation
    try:
        sim_result = simulate(
            case.scenario, baseline_solution, sim_case.changes,
            solver_timeout_s=timeout,
        )
    except Exception as exc:
        return SimCaseResult(
            case_id=case.id,
            sim_description=sim_case.description,
            expected_outcome=sim_case.expected_outcome.value,
            actual_status="error",
            outcome_correct=False,
            error=f"Simulation error: {exc}",
        )

    # Check preservation
    preserved = (
        case.scenario.model_dump() == original_dump
        and baseline_solution.assignments == original_assignments
    )

    # Map simulation status to expected outcome
    status_map = {
        SimulationStatus.SUCCESS: ExpectedSimOutcome.SUCCESS,
        SimulationStatus.INFEASIBLE: ExpectedSimOutcome.INFEASIBLE,
        SimulationStatus.INVALID_CHANGE: ExpectedSimOutcome.INVALID_CHANGE,
        SimulationStatus.VALIDATION_FAILED: ExpectedSimOutcome.VALIDATION_FAILED,
    }
    actual_mapped = status_map.get(sim_result.status)
    outcome_correct = actual_mapped == sim_case.expected_outcome

    return SimCaseResult(
        case_id=case.id,
        sim_description=sim_case.description,
        expected_outcome=sim_case.expected_outcome.value,
        actual_status=sim_result.status.value,
        outcome_correct=outcome_correct,
        original_preserved=preserved,
        has_assignment_diff=sim_result.assignment_diff is not None,
        has_metrics_diff=sim_result.metrics_diff is not None,
    )


# ==================================================================
# NL evaluation (optional, requires provider)
# ==================================================================


def _evaluate_nl(case: BenchmarkCase, provider) -> NLCaseResult:
    """Evaluate NL parsing for one case."""
    from shift_happens.parser.nl_parser import parse_natural_language

    prompt = case.nl_prompt
    expected = case.nl_expected

    try:
        parse_result = parse_natural_language(prompt, provider)
    except Exception as exc:
        return NLCaseResult(
            case_id=case.id,
            nl_prompt=prompt,
            parse_success=False,
            error=f"Parse exception: {exc}",
        )

    if not parse_result.success:
        return NLCaseResult(
            case_id=case.id,
            nl_prompt=prompt,
            parse_success=False,
            error=f"Parse failed: {', '.join(parse_result.errors)}",
        )

    scenario = parse_result.scenario
    validation = validate_scenario(scenario)

    # Check expected facts
    facts_checked = 0
    facts_matched = 0
    detail: dict[str, bool] = {}

    if expected.employee_count is not None:
        facts_checked += 1
        match = len(scenario.employees) == expected.employee_count
        detail["employee_count"] = match
        facts_matched += int(match)

    if expected.num_days is not None:
        facts_checked += 1
        match = scenario.num_days == expected.num_days
        detail["num_days"] = match
        facts_matched += int(match)

    if expected.shift_types is not None:
        facts_checked += 1
        parsed_shifts = sorted(s.name.lower() for s in scenario.shift_types)
        expected_shifts = sorted(s.lower() for s in expected.shift_types)
        match = parsed_shifts == expected_shifts
        detail["shift_types"] = match
        facts_matched += int(match)

    if expected.has_availability_constraints:
        facts_checked += 1
        match = len(scenario.employee_availability) > 0
        detail["has_availability"] = match
        facts_matched += int(match)

    if expected.has_shift_restrictions:
        facts_checked += 1
        match = len(scenario.employee_shift_restrictions) > 0
        detail["has_restrictions"] = match
        facts_matched += int(match)

    if expected.has_workload_limits:
        facts_checked += 1
        wl = scenario.workload_limits
        match = any([
            wl.max_shifts_per_employee is not None,
            wl.min_shifts_per_employee is not None,
            wl.max_consecutive_days is not None,
            wl.min_days_off is not None,
        ])
        detail["has_workload_limits"] = match
        facts_matched += int(match)

    return NLCaseResult(
        case_id=case.id,
        nl_prompt=prompt,
        parse_success=True,
        scenario_valid=validation.valid,
        facts_checked=facts_checked,
        facts_matched=facts_matched,
        fact_details=detail,
    )


# ==================================================================
# Aggregate metrics
# ==================================================================


@dataclass
class AggregateMetrics:
    """Summary statistics computed from benchmark results."""
    # Overall
    total_cases: int = 0
    outcome_correct: int = 0          # End-to-end correctness (all cases)
    outcome_accuracy: float = 0.0
    # Validation
    validation_correct: int = 0
    validation_accuracy: float = 0.0
    validation_rejected: int = 0      # Cases rejected before solver
    # Solver (only cases actually submitted to solver)
    solver_submitted: int = 0
    solver_outcome_correct: int = 0
    solver_outcome_accuracy: float = 0.0
    feasible_count: int = 0
    infeasible_count: int = 0
    solver_error_count: int = 0
    # Independent verification (only feasible solutions with checks performed)
    verification_passed: int = 0
    verification_total: int = 0
    verification_rate: float = 0.0
    # Timing (solver cases only)
    mean_solver_time_s: float = 0.0
    p50_solver_time_s: float = 0.0
    p95_solver_time_s: float = 0.0
    max_solver_time_s: float = 0.0
    # Quality (feasible cases only)
    mean_coverage_pct: float = 0.0
    mean_utilization_pct: float = 0.0
    # Simulation
    sim_total: int = 0
    sim_correct: int = 0
    sim_accuracy: float = 0.0
    sim_preserved: int = 0
    # NL
    nl_total: int = 0
    nl_parse_success: int = 0
    nl_parse_rate: float = 0.0
    nl_facts_checked: int = 0
    nl_facts_matched: int = 0
    nl_fact_accuracy: float = 0.0
    nl_mode: str = "disabled"


def compute_aggregate(report: BenchmarkReport) -> AggregateMetrics:
    """Compute summary metrics from a benchmark report."""
    m = AggregateMetrics()

    # Overall
    sr = report.solver_results
    m.total_cases = len(sr)
    m.outcome_correct = sum(1 for r in sr if r.outcome_correct)
    m.outcome_accuracy = m.outcome_correct / m.total_cases if m.total_cases else 0.0

    # Validation
    m.validation_correct = sum(1 for r in sr if r.validation_correct)
    m.validation_accuracy = m.validation_correct / m.total_cases if m.total_cases else 0.0
    m.validation_rejected = sum(1 for r in sr if r.solver_skipped)

    # Solver accuracy — only for cases that were actually submitted to the solver
    solver_cases = [r for r in sr if not r.solver_skipped]
    m.solver_submitted = len(solver_cases)
    m.solver_outcome_correct = sum(1 for r in solver_cases if r.outcome_correct)
    m.solver_outcome_accuracy = (
        m.solver_outcome_correct / m.solver_submitted if m.solver_submitted else 0.0
    )

    m.feasible_count = sum(
        1 for r in solver_cases if r.actual_status in ("optimal", "feasible")
    )
    m.infeasible_count = sum(1 for r in solver_cases if r.actual_status == "infeasible")
    m.solver_error_count = sum(
        1 for r in solver_cases if r.actual_status == "error"
    )

    # Independent verification — only count cases with checks actually performed
    verified = [
        r for r in solver_cases
        if r.verification is not None and r.verification.checks_performed > 0
    ]
    m.verification_total = len(verified)
    m.verification_passed = sum(1 for r in verified if r.verification.passed)
    m.verification_rate = (
        m.verification_passed / m.verification_total if m.verification_total else 0.0
    )

    times = [r.solver_time_s for r in solver_cases if r.solver_time_s is not None]
    if times:
        m.mean_solver_time_s = sum(times) / len(times)
        sorted_t = sorted(times)
        m.p50_solver_time_s = sorted_t[len(sorted_t) // 2]
        p95_idx = min(int(len(sorted_t) * 0.95), len(sorted_t) - 1)
        m.p95_solver_time_s = sorted_t[p95_idx]
        m.max_solver_time_s = sorted_t[-1]

    coverages = [r.coverage_pct for r in solver_cases if r.coverage_pct is not None]
    m.mean_coverage_pct = sum(coverages) / len(coverages) if coverages else 0.0

    utils = [r.utilization_pct for r in solver_cases if r.utilization_pct is not None]
    m.mean_utilization_pct = sum(utils) / len(utils) if utils else 0.0

    # Simulation metrics
    sim = report.simulation_results
    m.sim_total = len(sim)
    m.sim_correct = sum(1 for s in sim if s.outcome_correct)
    m.sim_accuracy = m.sim_correct / m.sim_total if m.sim_total else 0.0
    m.sim_preserved = sum(1 for s in sim if s.original_preserved)

    # NL metrics
    nl = report.nl_results
    m.nl_total = len(nl)
    m.nl_parse_success = sum(1 for n in nl if n.parse_success)
    m.nl_parse_rate = m.nl_parse_success / m.nl_total if m.nl_total else 0.0
    m.nl_facts_checked = sum(n.facts_checked for n in nl)
    m.nl_facts_matched = sum(n.facts_matched for n in nl)
    m.nl_fact_accuracy = (
        m.nl_facts_matched / m.nl_facts_checked if m.nl_facts_checked else 0.0
    )
    m.nl_mode = report.nl_evaluation_mode

    return m
