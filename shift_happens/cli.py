"""Minimal CLI for the Shift Happens scheduling pipeline.

Usage:
  python -m shift_happens.cli "Schedule 5 employees across morning and night shifts for 7 days."
  python -m shift_happens.cli --file request.txt

Requires GEMINI_API_KEY environment variable for live LLM parsing.
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="shift_happens",
        description="AI-Powered Shift Scheduling — end-to-end pipeline",
    )
    p.add_argument(
        "request",
        nargs="?",
        help="Natural-language scheduling request (or use --file)",
    )
    p.add_argument(
        "--file", "-f",
        help="Read the scheduling request from a text file",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Solver timeout in seconds (default: 60)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success, 1 on pipeline failure, 2 on usage error."""
    args = _build_parser().parse_args(argv)

    # Resolve the request text
    if args.file:
        try:
            with open(args.file, encoding="utf-8") as fh:
                request_text = fh.read()
        except OSError as exc:
            print(f"Error reading file: {exc}", file=sys.stderr)
            return 2
    elif args.request:
        request_text = args.request
    else:
        print("Error: provide a scheduling request as an argument or via --file.", file=sys.stderr)
        return 2

    if not request_text.strip():
        print("Error: empty scheduling request.", file=sys.stderr)
        return 2

    # Build the provider
    from shift_happens.llm.provider import GoogleGeminiProvider, ProviderError
    try:
        provider = GoogleGeminiProvider()
    except ProviderError as exc:
        print(f"Provider setup failed: {exc}", file=sys.stderr)
        return 1

    # Run the pipeline
    from shift_happens.pipeline import PipelineStatus, schedule
    result = schedule(request_text, provider, solver_timeout_s=args.timeout)

    # Display the result
    if result.status == PipelineStatus.SUCCESS:
        solution = result.solution
        scenario = result.scenario
        print(f"✓ Schedule created ({solution.status.value})")
        print(f"  Scenario: {scenario.name}")
        print(f"  Days: {scenario.num_days}  |  Employees: {len(scenario.employees)}  |  Shifts: {len(scenario.shift_types)}")
        print(f"  Assignments: {solution.metrics.total_assignments}")
        print(f"  Coverage: {solution.metrics.coverage_pct:.1f}%")
        print(f"  Objective: {solution.objective_value}")
        print(f"  Solver time: {solution.solver_wall_time_s:.3f}s")
        print()
        # Print schedule grid by day
        for day in range(scenario.num_days):
            day_assignments = [a for a in solution.assignments if a.day == day]
            if day_assignments:
                parts = [f"{a.employee_name}→{a.shift_type}" for a in day_assignments]
                print(f"  Day {day}: {', '.join(parts)}")
        if result.warnings:
            print()
            for w in result.warnings:
                print(f"  ⚠ {w}")
        return 0

    elif result.status == PipelineStatus.INFEASIBLE:
        print("✗ No feasible schedule found.")
        for e in result.errors:
            print(f"  {e}")
        return 1

    elif result.status == PipelineStatus.PARSE_FAILED:
        kind = result.parse_result.error_kind.value if result.parse_result else "unknown"
        print(f"✗ Parsing failed ({kind}):")
        for e in result.errors:
            print(f"  - {e}")
        return 1

    else:
        print(f"✗ Pipeline error ({result.status.value}):")
        for e in result.errors:
            print(f"  - {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
