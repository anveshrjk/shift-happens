"""Benchmark report generation — JSON, CSV, and text summary.

Reports are written to the ``benchmark_results/`` directory to keep
generated output separate from source code.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

from shift_happens.benchmark.runner import (
    AggregateMetrics,
    BenchmarkReport,
    NLCaseResult,
    SimCaseResult,
    SolverCaseResult,
    compute_aggregate,
)


DEFAULT_OUTPUT_DIR = "benchmark_results"


def generate_reports(
    report: BenchmarkReport,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    """Write JSON, CSV, and summary reports.  Returns paths keyed by format."""
    os.makedirs(output_dir, exist_ok=True)

    paths: dict[str, str] = {}
    paths["json"] = _write_json(report, output_dir)
    paths["csv_solver"] = _write_solver_csv(report, output_dir)
    if report.simulation_results:
        paths["csv_simulation"] = _write_simulation_csv(report, output_dir)
    if report.nl_results:
        paths["csv_nl"] = _write_nl_csv(report, output_dir)
    paths["summary"] = _write_summary(report, output_dir)

    return paths


def format_summary(report: BenchmarkReport) -> str:
    """Return a formatted text summary of the benchmark."""
    agg = compute_aggregate(report)
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  SHIFT HAPPENS — Evaluation Benchmark Report")
    lines.append("=" * 60)
    lines.append(f"  Dataset version : {report.dataset_version}")
    lines.append(f"  Timestamp       : {report.timestamp}")
    lines.append(f"  NL eval mode    : {report.nl_evaluation_mode}")
    lines.append("")

    # Solver section
    lines.append("─" * 60)
    lines.append("  VALIDATION")
    lines.append("─" * 60)
    lines.append(f"  Total cases     : {agg.total_cases}")
    lines.append(f"  Valid. correct   : {agg.validation_correct}/{agg.total_cases} "
                 f"({agg.validation_accuracy:.1%})")
    lines.append(f"  Rejected (valid.): {agg.validation_rejected} "
                 f"(solver skipped)")
    lines.append("")

    lines.append("─" * 60)
    lines.append("  SOLVER EVALUATION (cases submitted to solver)")
    lines.append("─" * 60)
    lines.append(f"  Cases submitted : {agg.solver_submitted}")
    lines.append(f"  Outcome correct : {agg.solver_outcome_correct}/{agg.solver_submitted} "
                 f"({agg.solver_outcome_accuracy:.1%})")
    lines.append(f"  Feasible        : {agg.feasible_count}")
    lines.append(f"  Infeasible      : {agg.infeasible_count}")
    lines.append(f"  Solver errors   : {agg.solver_error_count}")
    lines.append(f"  Constraint pass : {agg.verification_passed}/{agg.verification_total} "
                 f"({agg.verification_rate:.1%})")
    lines.append(f"  Solver time     : mean={agg.mean_solver_time_s:.3f}s  "
                 f"p50={agg.p50_solver_time_s:.3f}s  "
                 f"p95={agg.p95_solver_time_s:.3f}s  "
                 f"max={agg.max_solver_time_s:.3f}s")
    lines.append(f"  Mean coverage   : {agg.mean_coverage_pct:.1f}%")
    lines.append(f"  Mean utilization: {agg.mean_utilization_pct:.1f}%")
    lines.append("")
    lines.append(f"  End-to-end correct: {agg.outcome_correct}/{agg.total_cases} "
                 f"({agg.outcome_accuracy:.1%})")

    # Per-case solver detail
    lines.append("")
    lines.append("  Case Results:")
    for r in report.solver_results:
        mark = "✓" if r.outcome_correct else "✗"
        if r.solver_skipped:
            vmark = "—"
            time_s = "skipped"
        else:
            vmark = "✓" if r.verification and r.verification.passed else (
                "—" if r.verification is None else "✗"
            )
            time_s = f"{r.solver_time_s:.3f}s" if r.solver_time_s else "—"
        lines.append(
            f"    {mark} {r.case_id:<35} "
            f"expect={r.expected_outcome:<22} actual={r.actual_status:<20} "
            f"verify={vmark}  time={time_s}"
        )

    # Simulation section
    if report.simulation_results:
        lines.append("")
        lines.append("─" * 60)
        lines.append("  SIMULATION EVALUATION")
        lines.append("─" * 60)
        lines.append(f"  Cases           : {agg.sim_total}")
        lines.append(f"  Outcome correct : {agg.sim_correct}/{agg.sim_total} "
                     f"({agg.sim_accuracy:.1%})")
        lines.append(f"  Original preserved: {agg.sim_preserved}/{agg.sim_total}")
        lines.append("")
        for s in report.simulation_results:
            mark = "✓" if s.outcome_correct else "✗"
            pmark = "✓" if s.original_preserved else "✗"
            lines.append(
                f"    {mark} {s.case_id}  {s.sim_description}  "
                f"expect={s.expected_outcome}  actual={s.actual_status}  "
                f"preserved={pmark}"
            )

    # NL section
    if report.nl_results:
        lines.append("")
        lines.append("─" * 60)
        lines.append("  NL PARSING EVALUATION")
        lines.append("─" * 60)
        lines.append(f"  Mode            : {agg.nl_mode}")
        lines.append(f"  Cases           : {agg.nl_total}")
        lines.append(f"  Parse success   : {agg.nl_parse_success}/{agg.nl_total} "
                     f"({agg.nl_parse_rate:.1%})")
        lines.append(f"  Fact accuracy   : {agg.nl_facts_matched}/{agg.nl_facts_checked} "
                     f"({agg.nl_fact_accuracy:.1%})")
        lines.append("")
        for n in report.nl_results:
            mark = "✓" if n.parse_success else "✗"
            facts = (f"{n.facts_matched}/{n.facts_checked}" if n.facts_checked
                     else "—")
            lines.append(
                f"    {mark} {n.case_id}  parsed={n.parse_success}  "
                f"facts={facts}"
            )
            if n.fact_details:
                for k, v in n.fact_details.items():
                    fmark = "✓" if v else "✗"
                    lines.append(f"        {fmark} {k}")
    elif report.nl_evaluation_mode == "disabled":
        lines.append("")
        lines.append("  NL evaluation: disabled (use --live-nl to enable)")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ------------------------------------------------------------------
# File writers
# ------------------------------------------------------------------


def _write_json(report: BenchmarkReport, output_dir: str) -> str:
    """Write the full report as JSON."""
    path = os.path.join(output_dir, "benchmark_report.json")

    # Convert to serializable dict
    data = {
        "dataset_version": report.dataset_version,
        "timestamp": report.timestamp,
        "nl_evaluation_mode": report.nl_evaluation_mode,
        "aggregate": asdict(compute_aggregate(report)),
        "solver_results": [_solver_to_dict(r) for r in report.solver_results],
        "simulation_results": [asdict(s) for s in report.simulation_results],
        "nl_results": [asdict(n) for n in report.nl_results],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    return path


def _solver_to_dict(r: SolverCaseResult) -> dict:
    """Convert a SolverCaseResult to a JSON-safe dict."""
    d = {
        "case_id": r.case_id,
        "case_name": r.case_name,
        "category": r.category,
        "expected_outcome": r.expected_outcome,
        "actual_status": r.actual_status,
        "outcome_correct": r.outcome_correct,
        "validation_passed": r.validation_passed,
        "expected_valid": r.expected_valid,
        "validation_correct": r.validation_correct,
        "objective_value": r.objective_value,
        "solver_time_s": r.solver_time_s,
        "total_assignments": r.total_assignments,
        "coverage_pct": r.coverage_pct,
        "utilization_pct": r.utilization_pct,
        "soft_penalty": r.soft_penalty,
        "workload_spread": r.workload_spread,
        "error": r.error,
    }
    if r.verification is not None:
        d["verification_passed"] = r.verification.passed
        d["verification_checks"] = r.verification.checks_performed
        d["violations"] = [
            {"constraint": v.constraint, "detail": v.detail}
            for v in r.verification.violations
        ]
    return d


def _write_solver_csv(report: BenchmarkReport, output_dir: str) -> str:
    """Write solver results as CSV."""
    path = os.path.join(output_dir, "solver_results.csv")
    headers = [
        "case_id", "case_name", "category", "expected_outcome",
        "actual_status", "outcome_correct", "validation_correct",
        "solver_skipped", "verification_passed", "objective_value",
        "solver_time_s", "total_assignments", "coverage_pct",
        "utilization_pct", "soft_penalty", "workload_spread",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in report.solver_results:
            writer.writerow({
                "case_id": r.case_id,
                "case_name": r.case_name,
                "category": r.category,
                "expected_outcome": r.expected_outcome,
                "actual_status": r.actual_status,
                "outcome_correct": r.outcome_correct,
                "validation_correct": r.validation_correct,
                "solver_skipped": r.solver_skipped,
                "verification_passed": (
                    r.verification.passed if r.verification else ""
                ),
                "objective_value": r.objective_value,
                "solver_time_s": r.solver_time_s,
                "total_assignments": r.total_assignments,
                "coverage_pct": r.coverage_pct,
                "utilization_pct": r.utilization_pct,
                "soft_penalty": r.soft_penalty,
                "workload_spread": r.workload_spread,
            })
    return path


def _write_simulation_csv(report: BenchmarkReport, output_dir: str) -> str:
    """Write simulation results as CSV."""
    path = os.path.join(output_dir, "simulation_results.csv")
    headers = [
        "case_id", "sim_description", "expected_outcome",
        "actual_status", "outcome_correct", "original_preserved",
        "has_assignment_diff", "has_metrics_diff",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for s in report.simulation_results:
            writer.writerow({
                "case_id": s.case_id,
                "sim_description": s.sim_description,
                "expected_outcome": s.expected_outcome,
                "actual_status": s.actual_status,
                "outcome_correct": s.outcome_correct,
                "original_preserved": s.original_preserved,
                "has_assignment_diff": s.has_assignment_diff,
                "has_metrics_diff": s.has_metrics_diff,
            })
    return path


def _write_nl_csv(report: BenchmarkReport, output_dir: str) -> str:
    """Write NL evaluation results as CSV."""
    path = os.path.join(output_dir, "nl_results.csv")
    headers = [
        "case_id", "parse_success", "scenario_valid",
        "facts_checked", "facts_matched",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for n in report.nl_results:
            writer.writerow({
                "case_id": n.case_id,
                "parse_success": n.parse_success,
                "scenario_valid": n.scenario_valid,
                "facts_checked": n.facts_checked,
                "facts_matched": n.facts_matched,
            })
    return path


def _write_summary(report: BenchmarkReport, output_dir: str) -> str:
    """Write the text summary."""
    path = os.path.join(output_dir, "benchmark_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(format_summary(report))
    return path
