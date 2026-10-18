"""End-to-end scheduling pipeline.

Connects the NL parser, validator, and OR-Tools solver into a single
application-level workflow:

  Natural-language request
    → parse_natural_language()   (LLM → Pydantic → semantic validation)
    → solve()                    (OR-Tools CP-SAT)
    → PipelineResult             (scenario + solution + status)

The LLM interprets; the solver decides. This module orchestrates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from shift_happens.llm.provider import LLMProvider
from shift_happens.models.scenario import Scenario
from shift_happens.models.solution import ScheduleSolution, SolverStatus
from shift_happens.parser.nl_parser import ParseErrorKind, ParseResult, parse_natural_language
from shift_happens.solver.scheduler import solve


class PipelineStatus(str, Enum):
    """Overall outcome of the end-to-end pipeline."""

    SUCCESS = "success"                  # Parsed + solved (optimal or feasible)
    INFEASIBLE = "infeasible"            # Parsed OK, but no feasible schedule
    PARSE_FAILED = "parse_failed"        # Could not produce a valid Scenario
    SOLVER_ERROR = "solver_error"        # Solver crashed or timed out
    INTERNAL_ERROR = "internal_error"    # Unexpected exception


@dataclass
class PipelineResult:
    """Complete outcome of a scheduling request.

    Always inspect ``status`` first:
      - SUCCESS      → ``scenario`` and ``solution`` are both set.
      - INFEASIBLE   → ``scenario`` is set; ``solution`` has status=infeasible.
      - PARSE_FAILED → ``parse_result`` has error details.
      - SOLVER_ERROR → ``scenario`` is set; ``errors`` describes the failure.
      - INTERNAL_ERROR → ``errors`` describes the unexpected failure.
    """

    status: PipelineStatus
    scenario: Scenario | None = None
    solution: ScheduleSolution | None = None
    parse_result: ParseResult | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def schedule(
    request: str,
    provider: LLMProvider,
    *,
    solver_timeout_s: float = 60.0,
) -> PipelineResult:
    """Run the full NL → parse → validate → solve pipeline.

    Args:
        request: Natural-language scheduling request.
        provider: An LLMProvider instance (real or mock).
        solver_timeout_s: Maximum wall-clock seconds for the OR-Tools solver.

    Returns:
        A PipelineResult with the outcome of every pipeline stage.
    """
    # ---- Stage 1: Parse NL → validated Scenario ----
    try:
        parse_result = parse_natural_language(request, provider)
    except Exception as exc:
        return PipelineResult(
            status=PipelineStatus.INTERNAL_ERROR,
            errors=[f"Unexpected parsing error: {type(exc).__name__}: {exc}"],
        )

    if not parse_result.success:
        return PipelineResult(
            status=PipelineStatus.PARSE_FAILED,
            # Include the scenario if it exists (semantic errors retain it)
            scenario=parse_result.scenario,
            parse_result=parse_result,
            errors=parse_result.errors,
            warnings=parse_result.warnings,
        )

    scenario = parse_result.scenario
    assert scenario is not None  # Guaranteed by parse_result.success

    # ---- Stage 2: Solve with OR-Tools CP-SAT ----
    try:
        solution = solve(scenario, timeout_s=solver_timeout_s)
    except Exception as exc:
        return PipelineResult(
            status=PipelineStatus.SOLVER_ERROR,
            scenario=scenario,
            parse_result=parse_result,
            errors=[f"Solver error: {type(exc).__name__}: {exc}"],
            warnings=parse_result.warnings,
        )

    # ---- Stage 3: Classify the result ----
    if solution.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
        return PipelineResult(
            status=PipelineStatus.SUCCESS,
            scenario=scenario,
            solution=solution,
            parse_result=parse_result,
            warnings=parse_result.warnings,
        )

    if solution.status == SolverStatus.INFEASIBLE:
        infeasibility_errors = []
        if solution.infeasibility_reason:
            infeasibility_errors.append(solution.infeasibility_reason)
        else:
            infeasibility_errors.append(
                "The solver could not find a feasible schedule for the "
                "given constraints."
            )
        return PipelineResult(
            status=PipelineStatus.INFEASIBLE,
            scenario=scenario,
            solution=solution,
            parse_result=parse_result,
            errors=infeasibility_errors,
            warnings=parse_result.warnings,
        )

    # SolverStatus.ERROR or any unexpected status
    return PipelineResult(
        status=PipelineStatus.SOLVER_ERROR,
        scenario=scenario,
        solution=solution,
        parse_result=parse_result,
        errors=[f"Solver returned status: {solution.status.value}"],
        warnings=parse_result.warnings,
    )
