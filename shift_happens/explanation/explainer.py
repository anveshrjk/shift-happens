"""Grounded explanation of solver results.

Generates human-readable summaries from actual solver output — scenario
structure, assignment data, and computed metrics.  Every statement is
derived from data that already exists in the Scenario and ScheduleSolution
objects.  Nothing is invented, estimated, or hallucinated.

This module does NOT call an LLM.  If an LLM-based narrative layer is
added later, it should receive these grounded facts as input.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from shift_happens.models.scenario import Scenario
from shift_happens.models.solution import ScheduleSolution, SolverStatus


def explain_schedule(
    scenario: Scenario,
    solution: ScheduleSolution,
) -> str:
    """Build a multi-paragraph explanation of a solved schedule.

    Returns a plain-text summary suitable for display in the UI.
    Only includes information actually present in *scenario* and *solution*.
    """
    parts: list[str] = []

    # --- Overall status ---
    parts.append(_status_summary(scenario, solution))

    # --- Metrics ---
    if solution.metrics is not None:
        parts.append(_metrics_summary(scenario, solution))

    # --- Workload balance ---
    if solution.metrics is not None and solution.metrics.employee_metrics:
        parts.append(_workload_summary(scenario, solution))

    # --- Constraint notes ---
    constraint_notes = _constraint_notes(scenario, solution)
    if constraint_notes:
        parts.append(constraint_notes)

    return "\n\n".join(parts)


def explain_simulation(
    baseline_scenario: Scenario,
    baseline_solution: ScheduleSolution,
    modified_scenario: Scenario,
    modified_solution: ScheduleSolution,
    changes_description: list[str],
) -> str:
    """Build an explanation comparing baseline and modified schedules."""
    parts: list[str] = []

    parts.append("**What-If Comparison**")
    parts.append(f"Changes applied: {len(changes_description)}")
    for desc in changes_description:
        parts.append(f"  • {desc}")

    bm = baseline_solution.metrics
    mm = modified_solution.metrics

    if bm and mm:
        parts.append("")
        parts.append("| Metric | Baseline | Modified | Δ |")
        parts.append("|--------|----------|----------|---|")
        parts.append(
            f"| Assignments | {bm.total_assignments} "
            f"| {mm.total_assignments} "
            f"| {mm.total_assignments - bm.total_assignments:+d} |"
        )
        parts.append(
            f"| Coverage | {bm.coverage_pct:.1f}% "
            f"| {mm.coverage_pct:.1f}% "
            f"| {mm.coverage_pct - bm.coverage_pct:+.1f}% |"
        )
        parts.append(
            f"| Utilization | {bm.utilization_pct:.1f}% "
            f"| {mm.utilization_pct:.1f}% "
            f"| {mm.utilization_pct - bm.utilization_pct:+.1f}% |"
        )
        parts.append(
            f"| Soft Penalty | {bm.soft_penalty_total} "
            f"| {mm.soft_penalty_total} "
            f"| {mm.soft_penalty_total - bm.soft_penalty_total:+d} |"
        )

    bo = baseline_solution.objective_value
    mo = modified_solution.objective_value
    if bo is not None and mo is not None:
        delta = mo - bo
        direction = "improved" if delta < 0 else "worsened" if delta > 0 else "unchanged"
        parts.append(f"\nObjective value {direction}: {bo:.1f} → {mo:.1f} (Δ {delta:+.1f})")

    return "\n".join(parts)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _status_summary(scenario: Scenario, solution: ScheduleSolution) -> str:
    """One-line status and scenario overview."""
    status_label = {
        SolverStatus.OPTIMAL: "Optimal schedule found",
        SolverStatus.FEASIBLE: "Feasible schedule found (may not be optimal)",
        SolverStatus.INFEASIBLE: "No feasible schedule exists",
        SolverStatus.ERROR: "Solver encountered an error",
    }.get(solution.status, f"Status: {solution.status.value}")

    line = (
        f"**{status_label}** for '{scenario.name}' - "
        f"{len(scenario.employees)} employees, "
        f"{len(scenario.shift_types)} shift types, "
        f"{scenario.num_days} days."
    )
    if solution.solver_wall_time_s is not None:
        line += f"  Solved in {solution.solver_wall_time_s:.3f}s."
    return line


def _metrics_summary(scenario: Scenario, solution: ScheduleSolution) -> str:
    """Summary of key metrics."""
    m = solution.metrics
    lines: list[str] = []
    lines.append(f"**Metrics:** {m.total_assignments} total assignments.")
    lines.append(f"  • Coverage: {m.coverage_pct:.1f}% of required shift slots filled.")
    lines.append(f"  • Utilization: {m.utilization_pct:.1f}% average employee utilization.")
    if m.soft_penalty_total > 0:
        lines.append(f"  • Soft constraint penalty: {m.soft_penalty_total}.")
    if solution.objective_value is not None:
        lines.append(f"  • Objective value: {solution.objective_value:.1f}.")
    return "\n".join(lines)


def _workload_summary(scenario: Scenario, solution: ScheduleSolution) -> str:
    """Per-employee workload overview."""
    em = solution.metrics.employee_metrics
    shifts = [e.total_shifts for e in em]
    min_s, max_s = min(shifts), max(shifts)
    avg_s = sum(shifts) / len(shifts) if shifts else 0

    lines: list[str] = []
    lines.append(
        f"**Workload distribution:** "
        f"min {min_s}, max {max_s}, avg {avg_s:.1f} shifts/employee."
    )
    if max_s - min_s <= 1:
        lines.append("  • Workload is well-balanced (spread ≤ 1 shift).")
    elif max_s - min_s <= 2:
        lines.append("  • Workload is reasonably balanced (spread ≤ 2 shifts).")
    else:
        lines.append(
            f"  • Workload spread is {max_s - min_s} shifts — "
            f"consider adding workload limits."
        )
    return "\n".join(lines)


def _constraint_notes(scenario: Scenario, solution: ScheduleSolution) -> Optional[str]:
    """Notes about constraint configuration."""
    notes: list[str] = []
    wl = scenario.workload_limits

    if wl.max_shifts_per_employee is not None:
        notes.append(f"  • Max shifts/employee: {wl.max_shifts_per_employee}")
    if wl.min_shifts_per_employee is not None:
        notes.append(f"  • Min shifts/employee: {wl.min_shifts_per_employee}")
    if wl.max_consecutive_days is not None:
        notes.append(f"  • Max consecutive days: {wl.max_consecutive_days}")
    if wl.min_days_off is not None:
        notes.append(f"  • Min days off: {wl.min_days_off}")

    if scenario.employee_shift_restrictions:
        n = len(scenario.employee_shift_restrictions)
        notes.append(f"  • {n} shift restriction(s) applied.")
    if scenario.employee_availability:
        n = len(scenario.employee_availability)
        notes.append(f"  • {n} unavailability constraint(s) applied.")

    if solution.status == SolverStatus.INFEASIBLE and solution.infeasibility_reason:
        notes.insert(0, f"  ⚠ {solution.infeasibility_reason}")

    if not notes:
        return None
    return "**Constraints:**\n" + "\n".join(notes)


def describe_change(change) -> str:
    """Return a human-readable description of a simulation change object."""
    from shift_happens.simulation.engine import (
        ChangeStaffingRequirement,
        ChangeWorkloadLimit,
        SetEmployeeUnavailable,
    )

    if isinstance(change, SetEmployeeUnavailable):
        priority = "soft" if change.priority.value == "soft" else "hard"
        return (
            f"Make {change.employee_id} unavailable on day {change.day} "
            f"({priority})"
        )
    elif isinstance(change, ChangeStaffingRequirement):
        parts = []
        if change.min_employees is not None:
            parts.append(f"min={change.min_employees}")
        if change.max_employees is not None:
            parts.append(f"max={change.max_employees}")
        return (
            f"Change staffing for day {change.day} {change.shift_type}: "
            + ", ".join(parts)
        )
    elif isinstance(change, ChangeWorkloadLimit):
        parts = []
        if change.max_shifts_per_employee is not None:
            parts.append(f"max_shifts={change.max_shifts_per_employee}")
        if change.min_shifts_per_employee is not None:
            parts.append(f"min_shifts={change.min_shifts_per_employee}")
        if change.max_consecutive_days is not None:
            parts.append(f"max_consecutive={change.max_consecutive_days}")
        if change.min_days_off is not None:
            parts.append(f"min_days_off={change.min_days_off}")
        return "Change workload limits: " + ", ".join(parts)
    return str(change)
