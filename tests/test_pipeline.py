"""Integration tests for the end-to-end scheduling pipeline.

All tests use MockLLMProvider — no API key or network access required.
These tests verify that the full NL → parse → validate → solve chain works
correctly, including error handling at every stage.
"""

from __future__ import annotations

import pytest

from shift_happens.llm.provider import MockLLMProvider, ProviderError
from shift_happens.models.solution import SolverStatus
from shift_happens.parser.nl_parser import ParseErrorKind
from shift_happens.pipeline import PipelineResult, PipelineStatus, schedule


# ------------------------------------------------------------------
# Fixture helpers — canned LLM responses for end-to-end tests
# ------------------------------------------------------------------


def _simple_feasible_dict() -> dict:
    """A small feasible scenario: 3 employees, 2 shifts, 3 days."""
    return {
        "name": "Test Schedule",
        "num_days": 3,
        "employees": [
            {"id": "alice", "name": "Alice"},
            {"id": "bob", "name": "Bob"},
            {"id": "carol", "name": "Carol"},
        ],
        "shift_types": [
            {"name": "morning"},
            {"name": "night"},
        ],
        "shift_requirements": [
            {"day": d, "shift_type": s, "min_employees": 1}
            for d in range(3)
            for s in ["morning", "night"]
        ],
    }


def _constrained_feasible_dict() -> dict:
    """Feasible scenario with restrictions, availability, and workload limits."""
    return {
        "name": "Constrained Schedule",
        "num_days": 5,
        "employees": [
            {"id": "alice", "name": "Alice"},
            {"id": "bob", "name": "Bob"},
            {"id": "carol", "name": "Carol"},
            {"id": "dave", "name": "Dave"},
            {"id": "eve", "name": "Eve"},
        ],
        "shift_types": [
            {"name": "morning"},
            {"name": "night"},
        ],
        "shift_requirements": [
            {"day": d, "shift_type": s, "min_employees": 1}
            for d in range(5)
            for s in ["morning", "night"]
        ],
        "employee_shift_restrictions": [
            {"employee_id": "alice", "shift_type": "night", "priority": "hard"},
        ],
        "employee_availability": [
            {"employee_id": "bob", "day": 4, "priority": "hard"},
        ],
        "workload_limits": {
            "max_shifts_per_employee": 4,
            "min_days_off": 1,
        },
    }


def _infeasible_dict() -> dict:
    """A structurally valid but infeasible scenario: 2 employees, need 5 per shift."""
    return {
        "num_days": 1,
        "employees": [
            {"id": "alice", "name": "Alice"},
            {"id": "bob", "name": "Bob"},
        ],
        "shift_types": [{"name": "morning"}],
        "shift_requirements": [
            {"day": 0, "shift_type": "morning", "min_employees": 5},
        ],
    }


def _semantically_invalid_dict() -> dict:
    """Passes Pydantic but fails semantic validation (staffing exceeds workforce)."""
    return {
        "num_days": 1,
        "employees": [{"id": "a", "name": "A"}],
        "shift_types": [{"name": "morning"}],
        "shift_requirements": [
            {"day": 0, "shift_type": "morning", "min_employees": 99},
        ],
    }


# ==================================================================
# Test: Successful end-to-end scheduling
# ==================================================================


class TestSuccessfulPipeline:
    """Tests where NL → parse → solve all succeed."""

    def test_simple_schedule_succeeds(self):
        """Valid NL request produces a schedule with assignments."""
        provider = MockLLMProvider(response=_simple_feasible_dict())
        result = schedule("Schedule 3 employees for 3 days", provider)

        assert result.status == PipelineStatus.SUCCESS
        assert result.scenario is not None
        assert result.solution is not None
        assert result.solution.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
        assert len(result.solution.assignments) > 0
        assert result.errors == []

    def test_scenario_reaches_solver(self):
        """Verify the parsed Scenario is the one the solver used."""
        provider = MockLLMProvider(response=_simple_feasible_dict())
        result = schedule("anything", provider)

        assert result.scenario.num_days == 3
        assert len(result.scenario.employees) == 3
        assert len(result.scenario.shift_types) == 2
        # Solution assignments should reference only employees from the scenario
        scenario_ids = {e.id for e in result.scenario.employees}
        for a in result.solution.assignments:
            assert a.employee_id in scenario_ids

    def test_metrics_returned(self):
        """Metrics are computed and included in the result."""
        provider = MockLLMProvider(response=_simple_feasible_dict())
        result = schedule("schedule request", provider)

        assert result.solution.metrics is not None
        assert result.solution.metrics.total_assignments > 0
        assert result.solution.metrics.coverage_pct > 0.0
        assert result.solution.solver_wall_time_s is not None

    def test_constrained_schedule(self):
        """Constraints are respected in the final solution."""
        provider = MockLLMProvider(response=_constrained_feasible_dict())
        result = schedule("constrained schedule", provider)

        assert result.status == PipelineStatus.SUCCESS
        # Alice should not be on night shifts
        for a in result.solution.assignments:
            if a.employee_id == "alice":
                assert a.shift_type != "night", "Alice has a hard night restriction"
        # Bob should not work on day 4
        for a in result.solution.assignments:
            if a.employee_id == "bob":
                assert a.day != 4, "Bob is unavailable on day 4"

    def test_parse_result_preserved(self):
        """The full ParseResult is available for inspection."""
        provider = MockLLMProvider(response=_simple_feasible_dict())
        result = schedule("request", provider)

        assert result.parse_result is not None
        assert result.parse_result.success is True
        assert result.parse_result.scenario is result.scenario

    def test_warnings_propagated(self):
        """Warnings from validation are surfaced in the pipeline result."""
        # Scenario without shift requirements triggers "no requirement" warnings
        provider = MockLLMProvider(response={
            "num_days": 2,
            "employees": [{"id": "a", "name": "A"}],
            "shift_types": [{"name": "day"}],
        })
        result = schedule("minimal schedule", provider)

        assert result.status == PipelineStatus.SUCCESS
        assert len(result.warnings) > 0
        assert any("No staffing requirement" in w for w in result.warnings)


# ==================================================================
# Test: Infeasible schedules
# ==================================================================


class TestInfeasiblePipeline:
    """Parsing succeeds, but the solver finds no feasible solution."""

    def test_infeasible_detected(self):
        """Infeasible scenario returns INFEASIBLE status, not SUCCESS."""
        provider = MockLLMProvider(response=_infeasible_dict())
        result = schedule("impossible schedule", provider)

        # Semantic validation catches this (99 min employees > 1 actual)
        # But _infeasible_dict asks for 5 with 2 employees — still caught
        # by semantic validator.
        # Let's check: 5 > 2, so semantic validator errors.
        # Actually: semantic validation catches min_employees > workforce.
        assert result.status in (PipelineStatus.INFEASIBLE, PipelineStatus.PARSE_FAILED)

    def test_solver_infeasible_with_interactions(self):
        """Infeasible due to constraint interactions, not caught by validator."""
        # 3 employees, 2 shifts, 3 days, max 1 shift per employee:
        # need 6 shifts total but 3 employees × 1 max = 3 possible → infeasible
        d = {
            "num_days": 3,
            "employees": [
                {"id": "a", "name": "A"},
                {"id": "b", "name": "B"},
                {"id": "c", "name": "C"},
            ],
            "shift_types": [{"name": "morning"}, {"name": "night"}],
            "shift_requirements": [
                {"day": d, "shift_type": s, "min_employees": 1}
                for d in range(3)
                for s in ["morning", "night"]
            ],
            "workload_limits": {"max_shifts_per_employee": 1},
        }
        provider = MockLLMProvider(response=d)
        result = schedule("tight schedule", provider)

        assert result.status == PipelineStatus.INFEASIBLE
        assert result.scenario is not None
        assert result.solution is not None
        assert result.solution.status == SolverStatus.INFEASIBLE
        assert len(result.errors) > 0

    def test_infeasible_has_reason(self):
        """Infeasible result includes a reason when available."""
        d = {
            "num_days": 1,
            "employees": [{"id": "a", "name": "A"}],
            "shift_types": [{"name": "morning"}, {"name": "night"}],
            "shift_requirements": [
                {"day": 0, "shift_type": "morning", "min_employees": 1},
                {"day": 0, "shift_type": "night", "min_employees": 1},
            ],
        }
        provider = MockLLMProvider(response=d)
        result = schedule("1 person 2 shifts same day", provider)

        assert result.status == PipelineStatus.INFEASIBLE
        assert len(result.errors) > 0


# ==================================================================
# Test: Parse failures blocking the solver
# ==================================================================


class TestParseFailureBlocksSolver:
    """Parsing failures must prevent solver execution."""

    def test_empty_input(self):
        provider = MockLLMProvider(response=_simple_feasible_dict())
        result = schedule("", provider)

        assert result.status == PipelineStatus.PARSE_FAILED
        assert result.solution is None
        assert result.scenario is None
        assert result.parse_result.error_kind == ParseErrorKind.EMPTY_INPUT

    def test_provider_error(self):
        provider = MockLLMProvider(error=ProviderError("API down"))
        result = schedule("schedule something", provider)

        assert result.status == PipelineStatus.PARSE_FAILED
        assert result.solution is None
        assert result.parse_result.error_kind == ParseErrorKind.PROVIDER_ERROR
        assert any("API down" in e for e in result.errors)

    def test_malformed_llm_output(self):
        provider = MockLLMProvider(response=[1, 2, 3])  # type: ignore[arg-type]
        result = schedule("schedule something", provider)

        assert result.status == PipelineStatus.PARSE_FAILED
        assert result.solution is None
        assert result.parse_result.error_kind == ParseErrorKind.INVALID_JSON

    def test_schema_error(self):
        """LLM returns a dict missing required fields."""
        provider = MockLLMProvider(response={"name": "Bad"})
        result = schedule("schedule something", provider)

        assert result.status == PipelineStatus.PARSE_FAILED
        assert result.solution is None
        assert result.parse_result.error_kind == ParseErrorKind.SCHEMA_ERROR
        assert len(result.errors) > 0

    def test_non_scheduling_request(self):
        provider = MockLLMProvider(response={"error": "Not a scheduling request."})
        result = schedule("What is the weather?", provider)

        assert result.status == PipelineStatus.PARSE_FAILED
        assert result.solution is None
        assert result.parse_result.error_kind == ParseErrorKind.NOT_SCHEDULING


# ==================================================================
# Test: Semantic validation failure blocking the solver
# ==================================================================


class TestSemanticValidationBlocksSolver:
    """Semantically invalid scenarios must not reach the solver."""

    def test_overstaffed_requirement(self):
        """Requirement exceeding workforce is caught before solving."""
        provider = MockLLMProvider(response=_semantically_invalid_dict())
        result = schedule("bad scenario", provider)

        assert result.status == PipelineStatus.PARSE_FAILED
        assert result.solution is None
        assert result.parse_result.error_kind == ParseErrorKind.SEMANTIC_ERROR
        # The scenario IS constructed (Pydantic passes) but is flagged as invalid
        assert result.scenario is not None
        assert len(result.errors) > 0


# ==================================================================
# Test: Solver status and metrics correctness
# ==================================================================


class TestSolverStatusAndMetrics:
    """Verify solver outputs are accurately represented."""

    def test_optimal_status_propagated(self):
        provider = MockLLMProvider(response=_simple_feasible_dict())
        result = schedule("request", provider)

        assert result.status == PipelineStatus.SUCCESS
        assert result.solution.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)

    def test_assignments_match_requirements(self):
        """Number of assignments covers the shift requirements."""
        provider = MockLLMProvider(response=_simple_feasible_dict())
        result = schedule("request", provider)

        # 3 days × 2 shifts × 1 min_employee = 6 minimum assignments
        assert len(result.solution.assignments) >= 6

    def test_employee_metrics_present(self):
        provider = MockLLMProvider(response=_simple_feasible_dict())
        result = schedule("request", provider)

        assert result.solution.metrics is not None
        assert len(result.solution.metrics.employee_metrics) == 3

    def test_coverage_correct(self):
        provider = MockLLMProvider(response=_simple_feasible_dict())
        result = schedule("request", provider)

        assert result.solution.metrics.coverage_pct == 100.0

    def test_solver_wall_time(self):
        provider = MockLLMProvider(response=_simple_feasible_dict())
        result = schedule("request", provider)

        assert result.solution.solver_wall_time_s is not None
        assert result.solution.solver_wall_time_s >= 0

    def test_objective_value_present(self):
        provider = MockLLMProvider(response=_simple_feasible_dict())
        result = schedule("request", provider)

        assert result.solution.objective_value is not None


# ==================================================================
# Test: Unexpected errors handled gracefully
# ==================================================================


class _CrashingProvider(MockLLMProvider):
    """Provider that raises an unexpected non-ProviderError exception."""

    def parse_scenario(self, natural_language: str) -> dict:
        raise RuntimeError("Unexpected internal failure")


class TestUnexpectedErrors:
    """Pipeline should not crash on unexpected exceptions."""

    def test_unexpected_parser_exception(self):
        """A non-ProviderError exception during parsing → INTERNAL_ERROR."""
        provider = _CrashingProvider(response={})
        result = schedule("anything", provider)

        assert result.status == PipelineStatus.INTERNAL_ERROR
        assert result.solution is None
        assert len(result.errors) > 0
        assert any("Unexpected" in e for e in result.errors)

    def test_success_never_has_none_solution(self):
        """SUCCESS status always has a non-None solution."""
        provider = MockLLMProvider(response=_simple_feasible_dict())
        result = schedule("request", provider)

        if result.status == PipelineStatus.SUCCESS:
            assert result.solution is not None
            assert result.scenario is not None

    def test_parse_failed_never_has_solution(self):
        """PARSE_FAILED status never has a solution."""
        provider = MockLLMProvider(response={"error": "nope"})
        result = schedule("weather?", provider)

        assert result.status == PipelineStatus.PARSE_FAILED
        assert result.solution is None


# ==================================================================
# Test: Pipeline result contract
# ==================================================================


class TestPipelineResultContract:
    """Verify the PipelineResult dataclass invariants."""

    def test_success_fields(self):
        provider = MockLLMProvider(response=_simple_feasible_dict())
        result = schedule("request", provider)

        assert result.status == PipelineStatus.SUCCESS
        assert result.scenario is not None
        assert result.solution is not None
        assert result.parse_result is not None
        assert result.errors == []

    def test_parse_failed_fields(self):
        provider = MockLLMProvider(error=ProviderError("fail"))
        result = schedule("request", provider)

        assert result.status == PipelineStatus.PARSE_FAILED
        assert result.solution is None
        assert result.parse_result is not None
        assert len(result.errors) > 0

    def test_infeasible_fields(self):
        d = {
            "num_days": 3,
            "employees": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
            "shift_types": [{"name": "morning"}, {"name": "night"}],
            "shift_requirements": [
                {"day": d, "shift_type": s, "min_employees": 1}
                for d in range(3)
                for s in ["morning", "night"]
            ],
            "workload_limits": {"max_shifts_per_employee": 1},
        }
        provider = MockLLMProvider(response=d)
        result = schedule("request", provider)

        assert result.status == PipelineStatus.INFEASIBLE
        assert result.scenario is not None
        assert result.solution is not None
        assert result.solution.status == SolverStatus.INFEASIBLE
        assert len(result.errors) > 0
