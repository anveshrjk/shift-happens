"""Natural-language → structured Scenario parser.

Pipeline:
  1. User text  →  LLMProvider.parse_scenario()  →  raw dict
  2. Raw dict   →  Pydantic Scenario.model_validate()  →  Scenario
  3. Scenario   →  validate_scenario()  →  ValidationResult

The function returns a ParseResult that clearly distinguishes:
  - LLM / provider errors (could not get structured output)
  - Schema errors (LLM output doesn't match the Pydantic model)
  - Semantic errors (valid structure but logically infeasible)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pydantic import ValidationError

from shift_happens.llm.provider import LLMProvider, ProviderError
from shift_happens.models.scenario import Scenario
from shift_happens.validation.validator import ValidationResult, validate_scenario


class ParseErrorKind(str, Enum):
    """Category of parsing failure."""

    PROVIDER_ERROR = "provider_error"      # LLM API failure
    INVALID_JSON = "invalid_json"          # LLM returned something unparseable
    SCHEMA_ERROR = "schema_error"          # Dict doesn't match Pydantic model
    SEMANTIC_ERROR = "semantic_error"      # Valid structure but logically infeasible
    NOT_SCHEDULING = "not_scheduling"      # LLM signalled non-scheduling request
    EMPTY_INPUT = "empty_input"            # User gave empty / whitespace input


@dataclass
class ParseResult:
    """Outcome of the NL → Scenario pipeline.

    On success: ``scenario`` is set, ``errors`` is empty.
    On failure: ``scenario`` is None, ``errors`` describes what went wrong.
    ``warnings`` may be populated even on success (semantic warnings).
    """

    scenario: Scenario | None = None
    error_kind: ParseErrorKind | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_llm_output: dict | None = None  # The dict the LLM returned (for debugging)

    @property
    def success(self) -> bool:
        """True when parsing produced a valid, solver-ready Scenario."""
        return self.scenario is not None and self.error_kind is None


def parse_natural_language(
    text: str,
    provider: LLMProvider,
) -> ParseResult:
    """Parse *text* into a validated Scenario using *provider*.

    This is the main entry point for the NL parsing pipeline.

    Args:
        text: Natural-language scheduling request.
        provider: An LLMProvider instance (real or mock).

    Returns:
        A ParseResult with either a valid Scenario or structured errors.
    """
    # ---- Step 0: Reject empty input ----
    if not text or not text.strip():
        return ParseResult(
            error_kind=ParseErrorKind.EMPTY_INPUT,
            errors=["Input is empty. Please describe a scheduling problem."],
        )

    # ---- Step 1: Get structured output from the LLM ----
    try:
        raw_dict = provider.parse_scenario(text.strip())
    except ProviderError as exc:
        return ParseResult(
            error_kind=ParseErrorKind.PROVIDER_ERROR,
            errors=[f"LLM provider error: {exc}"],
        )

    if not isinstance(raw_dict, dict):
        return ParseResult(
            error_kind=ParseErrorKind.INVALID_JSON,
            errors=[
                f"LLM returned {type(raw_dict).__name__} instead of a JSON object."
            ],
        )

    # ---- Step 1b: Check if LLM signalled a non-scheduling request ----
    if "error" in raw_dict and len(raw_dict) == 1:
        return ParseResult(
            error_kind=ParseErrorKind.NOT_SCHEDULING,
            errors=[str(raw_dict["error"])],
            raw_llm_output=raw_dict,
        )

    # ---- Step 2: Validate against the Pydantic Scenario model ----
    try:
        scenario = Scenario.model_validate(raw_dict)
    except ValidationError as exc:
        # Collect human-readable error messages
        error_msgs = []
        for err in exc.errors():
            loc = " → ".join(str(part) for part in err["loc"]) if err["loc"] else "(root)"
            error_msgs.append(f"{loc}: {err['msg']}")
        return ParseResult(
            error_kind=ParseErrorKind.SCHEMA_ERROR,
            errors=error_msgs,
            raw_llm_output=raw_dict,
        )

    # ---- Step 3: Semantic validation ----
    validation = validate_scenario(scenario)

    if not validation.valid:
        # Semantic errors make the scenario unusable, but we still
        # return it so the caller can inspect / fix it.
        return ParseResult(
            scenario=scenario,
            error_kind=ParseErrorKind.SEMANTIC_ERROR,
            errors=validation.errors,
            warnings=validation.warnings,
            raw_llm_output=raw_dict,
        )

    return ParseResult(
        scenario=scenario,
        warnings=validation.warnings,
        raw_llm_output=raw_dict,
    )
