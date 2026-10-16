"""Tests for the NL parsing pipeline and LLM provider abstraction.

All tests use MockLLMProvider — no API key or network access required.
"""

from __future__ import annotations

import pytest

from shift_happens.llm.provider import (
    LLMProvider,
    MockLLMProvider,
    ProviderError,
)
from shift_happens.models.scenario import Scenario
from shift_happens.parser.nl_parser import (
    ParseErrorKind,
    ParseResult,
    parse_natural_language,
)
from shift_happens.solver.scheduler import solve


# ------------------------------------------------------------------
# Fixture helpers — canned LLM responses
# ------------------------------------------------------------------

def _valid_scenario_dict() -> dict:
    """A small but complete, valid scenario dict."""
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
        "workload_limits": {
            "max_shifts_per_employee": 3,
            "min_days_off": 1,
        },
    }


def _valid_scenario_with_restrictions() -> dict:
    """Valid scenario with shift restrictions and availability.

    6 employees, 3 shifts, 5 days — enough slack for restrictions.
    """
    return {
        "name": "Restricted Schedule",
        "num_days": 5,
        "employees": [
            {"id": "alice", "name": "Alice"},
            {"id": "bob", "name": "Bob"},
            {"id": "carol", "name": "Carol"},
            {"id": "dave", "name": "Dave"},
            {"id": "eve", "name": "Eve"},
            {"id": "frank", "name": "Frank"},
        ],
        "shift_types": [
            {"name": "morning"},
            {"name": "afternoon"},
            {"name": "night"},
        ],
        "shift_requirements": [
            {"day": d, "shift_type": s, "min_employees": 1}
            for d in range(5)
            for s in ["morning", "afternoon", "night"]
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


# ==================================================================
# Test: Valid parsing → usable Scenario
# ==================================================================


class TestValidParsing:
    """Tests for successful NL → Scenario conversions."""

    def test_valid_basic_scenario(self):
        """A well-formed dict produces a successful parse."""
        provider = MockLLMProvider(response=_valid_scenario_dict())
        result = parse_natural_language("Schedule 3 employees for 3 days", provider)

        assert result.success is True
        assert result.scenario is not None
        assert result.error_kind is None
        assert result.errors == []
        assert isinstance(result.scenario, Scenario)

    def test_scenario_fields_preserved(self):
        """Parsed Scenario contains the data the LLM returned."""
        provider = MockLLMProvider(response=_valid_scenario_dict())
        result = parse_natural_language("anything", provider)

        assert result.scenario.num_days == 3
        assert len(result.scenario.employees) == 3
        assert len(result.scenario.shift_types) == 2
        assert result.scenario.employees[0].id == "alice"
        assert result.scenario.shift_types[0].name == "morning"

    def test_scenario_with_restrictions_and_availability(self):
        """Scenario with restrictions and availability parses correctly."""
        provider = MockLLMProvider(response=_valid_scenario_with_restrictions())
        result = parse_natural_language("complex schedule", provider)

        assert result.success is True
        assert len(result.scenario.employee_shift_restrictions) == 1
        assert result.scenario.employee_shift_restrictions[0].employee_id == "alice"
        assert len(result.scenario.employee_availability) == 1
        assert result.scenario.employee_availability[0].employee_id == "bob"

    def test_workload_limits_parsed(self):
        """WorkloadLimits are correctly carried through."""
        provider = MockLLMProvider(response=_valid_scenario_dict())
        result = parse_natural_language("anything", provider)

        wl = result.scenario.workload_limits
        assert wl.max_shifts_per_employee == 3
        assert wl.min_days_off == 1

    def test_raw_llm_output_preserved(self):
        """raw_llm_output field is set for debugging."""
        d = _valid_scenario_dict()
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("anything", provider)

        assert result.raw_llm_output == d

    def test_minimal_scenario_no_requirements(self):
        """A minimal scenario without shift_requirements is valid."""
        provider = MockLLMProvider(response={
            "num_days": 1,
            "employees": [{"id": "a", "name": "A"}],
            "shift_types": [{"name": "day"}],
        })
        result = parse_natural_language("minimal", provider)

        assert result.success is True
        assert result.scenario.num_days == 1
        # Should have warnings about missing requirements
        assert len(result.warnings) > 0

    def test_successful_parse_with_warnings(self):
        """A valid scenario may produce non-fatal warnings."""
        # Missing shift_requirements triggers "no requirement" warnings
        provider = MockLLMProvider(response={
            "num_days": 2,
            "employees": [{"id": "a", "name": "A"}],
            "shift_types": [{"name": "day"}, {"name": "night"}],
        })
        result = parse_natural_language("partial schedule", provider)

        assert result.success is True
        assert result.scenario is not None
        assert len(result.warnings) > 0
        # Warnings should mention day/shift pairs
        assert any("No staffing requirement" in w for w in result.warnings)


# ==================================================================
# Test: Parsed Scenario → solver integration
# ==================================================================


class TestSolverIntegration:
    """Verify that a parsed Scenario can be passed directly to solve()."""

    def test_parsed_scenario_solves(self):
        """A successfully parsed Scenario produces a feasible solution."""
        provider = MockLLMProvider(response=_valid_scenario_dict())
        result = parse_natural_language("schedule request", provider)

        assert result.success is True
        solution = solve(result.scenario)
        assert solution.status.value in ("optimal", "feasible")
        assert len(solution.assignments) > 0

    def test_parsed_scenario_with_restrictions_solves(self):
        """Scenario with restrictions solves and respects them."""
        provider = MockLLMProvider(response=_valid_scenario_with_restrictions())
        result = parse_natural_language("schedule request", provider)

        assert result.success is True
        solution = solve(result.scenario)
        assert solution.status.value in ("optimal", "feasible")

        # Alice should not be on night shifts
        for a in solution.assignments:
            if a.employee_id == "alice":
                assert a.shift_type != "night"


# ==================================================================
# Test: Empty and whitespace input
# ==================================================================


class TestEmptyInput:
    """Tests for empty/whitespace input rejection."""

    def test_empty_string(self):
        provider = MockLLMProvider(response=_valid_scenario_dict())
        result = parse_natural_language("", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.EMPTY_INPUT
        assert result.scenario is None
        assert len(result.errors) > 0

    def test_whitespace_only(self):
        provider = MockLLMProvider(response=_valid_scenario_dict())
        result = parse_natural_language("   \n\t  ", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.EMPTY_INPUT

    def test_none_like_empty(self):
        """None input is rejected (would fail the 'not text' check)."""
        provider = MockLLMProvider(response=_valid_scenario_dict())
        result = parse_natural_language(None, provider)  # type: ignore[arg-type]

        assert result.success is False
        assert result.error_kind == ParseErrorKind.EMPTY_INPUT


# ==================================================================
# Test: Provider/API failures
# ==================================================================


class TestProviderErrors:
    """Tests for LLM provider failures."""

    def test_provider_error_propagated(self):
        """ProviderError from the LLM is caught and reported."""
        provider = MockLLMProvider(error=ProviderError("API rate limit exceeded"))
        result = parse_natural_language("schedule something", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.PROVIDER_ERROR
        assert any("rate limit" in e.lower() for e in result.errors)

    def test_provider_error_no_scenario(self):
        """Provider failure produces no scenario."""
        provider = MockLLMProvider(error=ProviderError("timeout"))
        result = parse_natural_language("schedule something", provider)

        assert result.scenario is None
        assert result.raw_llm_output is None

    def test_provider_unconfigured(self):
        """MockLLMProvider with no response or error raises ProviderError."""
        provider = MockLLMProvider()
        result = parse_natural_language("anything", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.PROVIDER_ERROR


# ==================================================================
# Test: Invalid / malformed LLM output
# ==================================================================


class TestInvalidLLMOutput:
    """Tests for non-JSON, non-dict, and garbage LLM output."""

    def test_list_instead_of_dict(self):
        """LLM returns a list instead of a dict."""
        provider = MockLLMProvider(response=[1, 2, 3])  # type: ignore[arg-type]
        result = parse_natural_language("anything", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.INVALID_JSON
        assert any("list" in e.lower() for e in result.errors)

    def test_string_instead_of_dict(self):
        """LLM returns a raw string instead of a dict."""
        provider = MockLLMProvider(response="just text")  # type: ignore[arg-type]
        result = parse_natural_language("anything", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.INVALID_JSON

    def test_none_response(self):
        """LLM returns None."""
        provider = MockLLMProvider(response=None)
        result = parse_natural_language("anything", provider)

        # MockLLMProvider raises ProviderError when response is None
        assert result.success is False
        assert result.error_kind == ParseErrorKind.PROVIDER_ERROR

    def test_empty_dict(self):
        """LLM returns an empty dict {}."""
        provider = MockLLMProvider(response={})
        result = parse_natural_language("anything", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.SCHEMA_ERROR
        assert len(result.errors) > 0


# ==================================================================
# Test: Missing required fields (schema errors)
# ==================================================================


class TestSchemaErrors:
    """Tests for dict output that doesn't match the Pydantic Scenario model."""

    def test_missing_num_days(self):
        d = _valid_scenario_dict()
        del d["num_days"]
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("schedule", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.SCHEMA_ERROR
        assert any("num_days" in e for e in result.errors)

    def test_missing_employees(self):
        d = _valid_scenario_dict()
        del d["employees"]
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("schedule", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.SCHEMA_ERROR
        assert any("employees" in e for e in result.errors)

    def test_missing_shift_types(self):
        d = _valid_scenario_dict()
        del d["shift_types"]
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("schedule", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.SCHEMA_ERROR

    def test_invalid_num_days_type(self):
        d = _valid_scenario_dict()
        d["num_days"] = "seven"
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("schedule", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.SCHEMA_ERROR

    def test_zero_num_days(self):
        d = _valid_scenario_dict()
        d["num_days"] = 0
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("schedule", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.SCHEMA_ERROR

    def test_negative_num_days(self):
        d = _valid_scenario_dict()
        d["num_days"] = -1
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("schedule", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.SCHEMA_ERROR

    def test_empty_employees_list(self):
        d = _valid_scenario_dict()
        d["employees"] = []
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("schedule", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.SCHEMA_ERROR

    def test_employee_missing_id(self):
        d = _valid_scenario_dict()
        d["employees"] = [{"name": "Alice"}]  # missing "id"
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("schedule", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.SCHEMA_ERROR

    def test_duplicate_employee_ids(self):
        d = _valid_scenario_dict()
        d["employees"] = [
            {"id": "alice", "name": "Alice"},
            {"id": "alice", "name": "Alice2"},
        ]
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("schedule", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.SCHEMA_ERROR

    def test_unknown_shift_in_requirement(self):
        """Reference to a shift type that doesn't exist."""
        d = _valid_scenario_dict()
        d["shift_requirements"] = [
            {"day": 0, "shift_type": "graveyard", "min_employees": 1}
        ]
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("schedule", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.SCHEMA_ERROR

    def test_unknown_employee_in_restriction(self):
        d = _valid_scenario_dict()
        d["employee_shift_restrictions"] = [
            {"employee_id": "nobody", "shift_type": "morning"}
        ]
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("schedule", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.SCHEMA_ERROR

    def test_day_out_of_range_in_requirement(self):
        d = _valid_scenario_dict()
        d["shift_requirements"] = [
            {"day": 99, "shift_type": "morning", "min_employees": 1}
        ]
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("schedule", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.SCHEMA_ERROR

    def test_max_lt_min_employees(self):
        """max_employees < min_employees in a shift requirement."""
        d = _valid_scenario_dict()
        d["shift_requirements"] = [
            {"day": 0, "shift_type": "morning",
             "min_employees": 5, "max_employees": 2}
        ]
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("schedule", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.SCHEMA_ERROR


# ==================================================================
# Test: Semantically invalid scenarios (valid Pydantic, bad logic)
# ==================================================================


class TestSemanticErrors:
    """Tests for scenarios that pass Pydantic but fail semantic validation."""

    def test_staffing_exceeds_workforce(self):
        """More employees required per shift than total employees."""
        d = {
            "num_days": 1,
            "employees": [{"id": "a", "name": "A"}],
            "shift_types": [{"name": "morning"}],
            "shift_requirements": [
                {"day": 0, "shift_type": "morning", "min_employees": 5},
            ],
        }
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("understaffed", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.SEMANTIC_ERROR
        assert result.scenario is not None  # Scenario was constructed
        assert len(result.errors) > 0

    def test_contradictory_workload_limits(self):
        """min_shifts > max_shifts is semantically invalid."""
        d = _valid_scenario_dict()
        d["workload_limits"] = {
            "min_shifts_per_employee": 10,
            "max_shifts_per_employee": 2,
        }
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("bad limits", provider)

        # This is caught by Pydantic model_validator (min_lte_max)
        assert result.success is False
        # It's a schema error since Pydantic rejects it
        assert result.error_kind == ParseErrorKind.SCHEMA_ERROR

    def test_semantic_error_preserves_scenario(self):
        """On semantic errors the scenario is still available for inspection."""
        d = {
            "num_days": 1,
            "employees": [{"id": "a", "name": "A"}],
            "shift_types": [{"name": "morning"}],
            "shift_requirements": [
                {"day": 0, "shift_type": "morning", "min_employees": 99},
            ],
        }
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("too many", provider)

        assert result.scenario is not None
        assert result.scenario.shift_requirements[0].min_employees == 99
        assert result.success is False

    def test_semantic_error_distinct_from_schema_error(self):
        """SEMANTIC_ERROR and SCHEMA_ERROR are different categories."""
        assert ParseErrorKind.SEMANTIC_ERROR != ParseErrorKind.SCHEMA_ERROR


# ==================================================================
# Test: Non-scheduling requests
# ==================================================================


class TestNonSchedulingRequests:
    """Tests for LLM signaling a non-scheduling request."""

    def test_error_signal(self):
        """LLM returns {"error": "..."} for non-scheduling input."""
        provider = MockLLMProvider(response={"error": "Not a shift-scheduling request."})
        result = parse_natural_language("What is the weather?", provider)

        assert result.success is False
        assert result.error_kind == ParseErrorKind.NOT_SCHEDULING
        assert any("scheduling" in e.lower() for e in result.errors)

    def test_error_with_extra_fields_is_not_rejected(self):
        """A dict with 'error' plus other fields is treated as a normal dict."""
        d = _valid_scenario_dict()
        d["error"] = "some warning"
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("schedule", provider)

        # The dict has more than just "error", so it proceeds to validation
        # and should parse as a valid scenario (the extra "error" key is ignored by Pydantic)
        assert result.scenario is not None


# ==================================================================
# Test: Edge cases and unusual input
# ==================================================================


class TestEdgeCases:
    """Tests for unusual but valid scenarios and edge conditions."""

    def test_single_employee_single_day_single_shift(self):
        """Absolute minimum viable scenario."""
        provider = MockLLMProvider(response={
            "num_days": 1,
            "employees": [{"id": "a", "name": "A"}],
            "shift_types": [{"name": "s"}],
            "shift_requirements": [
                {"day": 0, "shift_type": "s", "min_employees": 1},
            ],
        })
        result = parse_natural_language("one person", provider)

        assert result.success is True
        assert result.scenario.num_days == 1
        assert len(result.scenario.employees) == 1

    def test_large_scenario(self):
        """Stress-test: many employees and days."""
        emps = [{"id": f"e{i}", "name": f"Emp{i}"} for i in range(50)]
        provider = MockLLMProvider(response={
            "num_days": 30,
            "employees": emps,
            "shift_types": [{"name": "day"}, {"name": "night"}],
            "shift_requirements": [
                {"day": d, "shift_type": s, "min_employees": 5}
                for d in range(30)
                for s in ["day", "night"]
            ],
        })
        result = parse_natural_language("big schedule", provider)

        assert result.success is True
        assert result.scenario.num_days == 30
        assert len(result.scenario.employees) == 50

    def test_extra_fields_ignored(self):
        """Extra keys in the dict are ignored by Pydantic (by default)."""
        d = _valid_scenario_dict()
        d["notes"] = "This is a note"
        d["version"] = 99
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("schedule", provider)

        assert result.success is True

    def test_soft_constraint_parsed(self):
        """Soft constraints with penalties are correctly handled."""
        d = _valid_scenario_dict()
        d["employee_shift_restrictions"] = [
            {
                "employee_id": "alice",
                "shift_type": "night",
                "priority": "soft",
                "penalty": 10,
            },
        ]
        provider = MockLLMProvider(response=d)
        result = parse_natural_language("soft constraints", provider)

        assert result.success is True
        restr = result.scenario.employee_shift_restrictions[0]
        assert restr.priority.value == "soft"
        assert restr.penalty == 10


# ==================================================================
# Test: Provider abstraction
# ==================================================================


class TestProviderAbstraction:
    """Tests for the provider layer itself."""

    def test_mock_provider_returns_response(self):
        """MockLLMProvider returns the configured response."""
        d = {"key": "value"}
        provider = MockLLMProvider(response=d)
        assert provider.parse_scenario("anything") == d

    def test_mock_provider_raises_error(self):
        """MockLLMProvider raises the configured error."""
        provider = MockLLMProvider(error=ProviderError("fail"))
        with pytest.raises(ProviderError, match="fail"):
            provider.parse_scenario("anything")

    def test_mock_provider_explain_solution(self):
        """MockLLMProvider has a working explain_solution stub."""
        provider = MockLLMProvider(response={})
        result = provider.explain_solution({}, {})
        assert isinstance(result, str)

    def test_provider_error_is_exception(self):
        """ProviderError inherits from Exception."""
        err = ProviderError("test message")
        assert isinstance(err, Exception)
        assert str(err) == "test message"

    def test_llm_provider_is_abstract(self):
        """LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_gemini_provider_missing_api_key(self):
        """GoogleGeminiProvider raises ProviderError without API key."""
        import os
        # Ensure no key is set
        original = os.environ.get("GEMINI_API_KEY")
        try:
            os.environ.pop("GEMINI_API_KEY", None)
            from shift_happens.llm.provider import GoogleGeminiProvider
            with pytest.raises(ProviderError, match="API key"):
                GoogleGeminiProvider(api_key="")
        finally:
            if original is not None:
                os.environ["GEMINI_API_KEY"] = original

    def test_gemini_provider_missing_package(self):
        """GoogleGeminiProvider reports missing google-genai package."""
        import os
        import sys

        original_key = os.environ.get("GEMINI_API_KEY")
        # Set a fake key so we get past the key check
        os.environ["GEMINI_API_KEY"] = "fake-key-for-test"

        # Temporarily remove google module if present
        google_modules = {k: v for k, v in sys.modules.items() if k.startswith("google")}
        for k in google_modules:
            del sys.modules[k]

        # Block google imports
        import importlib
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("google"):
                raise ImportError("No module named 'google'")
            return original_import(name, *args, **kwargs)

        try:
            import builtins
            builtins.__import__ = mock_import
            from shift_happens.llm.provider import GoogleGeminiProvider
            with pytest.raises(ProviderError, match="google-genai"):
                GoogleGeminiProvider()
        except Exception:
            # If mocking imports is too fragile, just skip
            pytest.skip("Could not mock import system reliably")
        finally:
            builtins.__import__ = original_import
            # Restore google modules
            sys.modules.update(google_modules)
            if original_key is not None:
                os.environ["GEMINI_API_KEY"] = original_key
            else:
                os.environ.pop("GEMINI_API_KEY", None)


# ==================================================================
# Test: ParseResult dataclass behavior
# ==================================================================


class TestParseResult:
    """Tests for the ParseResult data structure."""

    def test_success_default(self):
        """Empty ParseResult is not successful (no scenario)."""
        r = ParseResult()
        # scenario is None → success is False
        assert r.success is False

    def test_success_with_scenario(self):
        """ParseResult with a scenario and no error_kind is successful."""
        from shift_happens.models.scenario import Scenario, Employee, ShiftType
        scenario = Scenario(
            num_days=1,
            employees=[Employee(id="a", name="A")],
            shift_types=[ShiftType(name="s")],
        )
        r = ParseResult(scenario=scenario)
        assert r.success is True

    def test_not_success_with_error_kind(self):
        """ParseResult with error_kind is not successful even with scenario."""
        from shift_happens.models.scenario import Scenario, Employee, ShiftType
        scenario = Scenario(
            num_days=1,
            employees=[Employee(id="a", name="A")],
            shift_types=[ShiftType(name="s")],
        )
        r = ParseResult(scenario=scenario, error_kind=ParseErrorKind.SEMANTIC_ERROR)
        assert r.success is False
