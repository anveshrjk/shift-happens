"""LLM provider abstraction.

Keeps LLM-specific code isolated so the provider/model can be replaced
without touching the rest of the application.

Phase 1: Interface only. Implementation comes in Phase 3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract interface for an LLM provider."""

    @abstractmethod
    def parse_scenario(self, natural_language: str) -> dict:
        """Parse natural language into a structured scenario dict.

        The dict should conform to the Scenario Pydantic model schema.
        """
        ...

    @abstractmethod
    def explain_solution(self, scenario_dict: dict, solution_dict: dict) -> str:
        """Generate a natural-language explanation of a solver result."""
        ...
