"""CLI entry point for the Shift Happens evaluation benchmark.

Usage:
  # Default offline evaluation (no API key needed):
  python -m shift_happens.benchmark

  # With live NL evaluation (requires GEMINI_API_KEY):
  python -m shift_happens.benchmark --live-nl

  # Filter to specific cases:
  python -m shift_happens.benchmark --filter sim

  # Custom output directory:
  python -m shift_happens.benchmark --output results/

  # Custom solver timeout:
  python -m shift_happens.benchmark --timeout 10
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="shift_happens.benchmark",
        description="Shift Happens — Evaluation Benchmark Runner",
    )
    p.add_argument(
        "--live-nl",
        action="store_true",
        help="Enable live NL parsing evaluation (requires GEMINI_API_KEY)",
    )
    p.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only run cases whose ID contains this substring",
    )
    p.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark_results",
        help="Output directory for reports (default: benchmark_results)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Solver timeout per case in seconds (default: 30)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and produce reports."""
    args = _build_parser().parse_args(argv)

    from shift_happens.benchmark.runner import run_benchmark
    from shift_happens.benchmark.report import format_summary, generate_reports

    # Set up NL provider if requested
    nl_provider = None
    if args.live_nl:
        try:
            from shift_happens.llm.provider import GoogleGeminiProvider
            nl_provider = GoogleGeminiProvider()
            print("Live NL evaluation enabled (Gemini).")
        except Exception as exc:
            print(f"Warning: Could not create Gemini provider: {exc}",
                  file=sys.stderr)
            print("NL evaluation will be skipped.", file=sys.stderr)

    # Run benchmark
    print(f"Running benchmark (timeout={args.timeout}s)...")
    report = run_benchmark(
        include_nl=args.live_nl,
        nl_provider=nl_provider,
        solver_timeout_s=args.timeout,
        case_filter=args.filter,
    )

    # Display summary
    summary = format_summary(report)
    print(summary)

    # Write reports
    paths = generate_reports(report, output_dir=args.output)
    print(f"\nReports written to {args.output}/:")
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")

    # Exit code: 0 if all correct, 1 if any failures
    from shift_happens.benchmark.runner import compute_aggregate
    agg = compute_aggregate(report)
    if agg.outcome_correct < agg.total_cases:
        return 1
    if agg.sim_total > 0 and agg.sim_correct < agg.sim_total:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
