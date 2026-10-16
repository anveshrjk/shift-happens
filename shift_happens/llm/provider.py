"""LLM provider abstraction.

Keeps LLM-specific code isolated so the provider/model can be replaced
without touching the rest of the application.

Provides:
  - LLMProvider   — abstract interface
  - GoogleGeminiProvider — concrete Gemini implementation
  - MockLLMProvider — deterministic provider for testing
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from shift_happens.models.scenario import Scenario


def _scenario_schema_text() -> str:
    """Return a compact JSON-schema string for the Scenario model.

    This is embedded in every LLM prompt so the model knows
    exactly what structure to produce.
    """
    return json.dumps(Scenario.model_json_schema(), indent=2)


# ------------------------------------------------------------------
# System prompt shared across providers
# ------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a scheduling assistant. Your ONLY job is to convert a natural-language
shift-scheduling request into a structured JSON object.

RULES:
1. Output ONLY valid JSON — no markdown fences, no commentary, no explanation.
2. The JSON must conform EXACTLY to the schema below.
3. Days are 0-indexed integers. If the user says "7 days", num_days=7 and
   valid day indices are 0-6.
4. Every employee needs a unique "id" (lowercase, no spaces) and a "name".
5. shift_requirements must cover every (day, shift_type) pair that requires
   staffing. If the user says "each shift needs 3 people", generate one
   ShiftRequirement for every day × shift combination.
6. If information is missing or ambiguous, set the corresponding field to its
   default value or omit it — do NOT invent constraints the user did not state.
7. If the request is not about employee shift scheduling, return exactly:
   {"error": "Not a shift-scheduling request."}
8. constraint priorities are either "hard" or "soft".

JSON SCHEMA:
{schema}
"""


@dataclass(frozen=True)
class ProviderError(Exception):
    """Raised when the LLM API itself fails (network, auth, rate-limit)."""

    message: str

    def __str__(self) -> str:
        return self.message


# ------------------------------------------------------------------
# Abstract interface
# ------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract interface for an LLM provider."""

    @abstractmethod
    def parse_scenario(self, natural_language: str) -> dict:
        """Parse natural language into a structured scenario dict.

        The dict should conform to the Scenario Pydantic model schema.

        Raises:
            ProviderError: If the API call itself fails.
        """
        ...

    @abstractmethod
    def explain_solution(self, scenario_dict: dict, solution_dict: dict) -> str:
        """Generate a natural-language explanation of a solver result."""
        ...


# ------------------------------------------------------------------
# Google Gemini implementation
# ------------------------------------------------------------------


class GoogleGeminiProvider(LLMProvider):
    """LLM provider backed by the Google Gemini API.

    Requires the ``google-genai`` package and a ``GEMINI_API_KEY``
    environment variable (or pass *api_key* directly).
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise ProviderError(
                "No Gemini API key found. Set the GEMINI_API_KEY environment "
                "variable or pass api_key= to GoogleGeminiProvider."
            )

        # Lazy-import so the rest of the codebase works without google-genai
        try:
            from google import genai  # type: ignore[import-untyped]

            self._client = genai.Client(api_key=self._api_key)
        except ImportError as exc:
            raise ProviderError(
                "google-genai package is required for GoogleGeminiProvider. "
                "Install it with: pip install google-genai"
            ) from exc

    def parse_scenario(self, natural_language: str) -> dict:
        """Call Gemini to convert *natural_language* into a scenario dict."""
        from google import genai  # type: ignore[import-untyped]

        system = SYSTEM_PROMPT.format(schema=_scenario_schema_text())
        user_msg = (
            "Convert the following scheduling request into the JSON format "
            "described in your instructions. Output ONLY the JSON.\n\n"
            f"{natural_language}"
        )

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=user_msg,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.0,
                ),
            )
        except Exception as exc:
            raise ProviderError(
                f"Gemini API call failed ({type(exc).__name__}): {exc}"
            ) from exc

        raw = (response.text or "").strip()

        # Strip markdown code fences if the model wraps its output
        if raw.startswith("```"):
            lines = raw.splitlines()
            # Remove first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"Gemini returned invalid JSON: {exc}\nRaw output:\n{raw[:500]}"
            ) from exc

    def explain_solution(self, scenario_dict: dict, solution_dict: dict) -> str:
        """Generate a natural-language explanation of a solver result."""
        raise NotImplementedError(
            "Solution explanation will be implemented in Phase 6."
        )


# ------------------------------------------------------------------
# Mock provider for deterministic testing
# ------------------------------------------------------------------


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider that returns pre-configured responses.

    Usage::

        provider = MockLLMProvider(response={"num_days": 7, ...})
        result = provider.parse_scenario("anything")
        assert result == {"num_days": 7, ...}

    To simulate an API failure::

        provider = MockLLMProvider(error=ProviderError("timeout"))
    """

    def __init__(
        self,
        response: dict | None = None,
        error: ProviderError | None = None,
    ) -> None:
        self._response = response
        self._error = error

    def parse_scenario(self, natural_language: str) -> dict:
        if self._error is not None:
            raise self._error
        if self._response is not None:
            return self._response
        raise ProviderError("MockLLMProvider: no response configured")

    def explain_solution(self, scenario_dict: dict, solution_dict: dict) -> str:
        return "Mock explanation."
