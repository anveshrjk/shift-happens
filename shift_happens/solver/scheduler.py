"""OR-Tools CP-SAT based shift scheduler.

Takes a validated Scenario, builds a constraint-programming model,
solves it, and returns a ScheduleSolution with assignments and metrics.

Design principles:
  - Every hard constraint is an AddXxx call on the CP model.
  - Every soft constraint adds a penalty term to the objective.
  - The solver NEVER fabricates a solution; infeasible → reported as such.
"""

from __future__ import annotations

import time
from typing import Optional

from ortools.sat.python import cp_model

from shift_happens.models.scenario import (
    ConstraintPriority,
    Scenario,
)
from shift_happens.models.solution import (
    Assignment,
    EmployeeMetrics,
    ScheduleMetrics,
    ScheduleSolution,
    SolverStatus,
)


def solve(scenario: Scenario, timeout_s: float = 60.0) -> ScheduleSolution:
    """Build and solve the scheduling problem described by *scenario*.

    Args:
        scenario: A validated Scenario.
        timeout_s: Maximum wall-clock seconds for the solver.

    Returns:
        A ScheduleSolution indicating feasibility, assignments, and metrics.
    """
    model = cp_model.CpModel()

    employee_ids = [e.id for e in scenario.employees]
    employee_name = {e.id: e.name for e in scenario.employees}
    shift_names = [s.name for s in scenario.shift_types]
    days = list(range(scenario.num_days))

    # ------------------------------------------------------------------
    # Decision variables: x[(e, d, s)] = 1 iff employee e works shift s on day d
    # ------------------------------------------------------------------
    x: dict[tuple[str, int, str], cp_model.IntVar] = {}
    for e in employee_ids:
        for d in days:
            for s in shift_names:
                x[(e, d, s)] = model.new_bool_var(f"x_{e}_{d}_{s}")

    # Collect soft-penalty variables for the objective
    soft_penalties: list[cp_model.LinearExpr] = []

    # ------------------------------------------------------------------
    # C1: Each employee works at most one shift per day
    # ------------------------------------------------------------------
    for e in employee_ids:
        for d in days:
            model.add(sum(x[(e, d, s)] for s in shift_names) <= 1)

    # ------------------------------------------------------------------
    # C2: Shift staffing requirements
    # ------------------------------------------------------------------
    for req in scenario.shift_requirements:
        assigned = sum(x[(e, req.day, req.shift_type)] for e in employee_ids)
        model.add(assigned >= req.min_employees)
        if req.max_employees is not None:
            model.add(assigned <= req.max_employees)

    # ------------------------------------------------------------------
    # C3: Employee shift restrictions
    # ------------------------------------------------------------------
    for restr in scenario.employee_shift_restrictions:
        var = x[(restr.employee_id, 0, restr.shift_type)]  # placeholder — see below
        # Apply across ALL days (restriction is shift-type-wide)
        for d in days:
            var = x[(restr.employee_id, d, restr.shift_type)]
            if restr.priority == ConstraintPriority.HARD:
                model.add(var == 0)
            else:
                # Soft: penalize if assigned
                penalty = restr.penalty if restr.penalty > 0 else 1
                soft_penalties.append(var * penalty)

    # ------------------------------------------------------------------
    # C4: Employee unavailability
    # ------------------------------------------------------------------
    for avail in scenario.employee_availability:
        for s in shift_names:
            var = x[(avail.employee_id, avail.day, s)]
            if avail.priority == ConstraintPriority.HARD:
                model.add(var == 0)
            else:
                penalty = avail.penalty if avail.penalty > 0 else 1
                soft_penalties.append(var * penalty)

    # ------------------------------------------------------------------
    # C5: Workload limits
    # ------------------------------------------------------------------
    wl = scenario.workload_limits

    for e in employee_ids:
        total_shifts = sum(x[(e, d, s)] for d in days for s in shift_names)

        if wl.max_shifts_per_employee is not None:
            model.add(total_shifts <= wl.max_shifts_per_employee)

        if wl.min_shifts_per_employee is not None:
            model.add(total_shifts >= wl.min_shifts_per_employee)

        # Days off = days with no shift
        if wl.min_days_off is not None:
            days_worked = sum(
                model.new_bool_var(f"worked_{e}_{d}") for d in days
            )
            # We need a proper "worked on day d" indicator
            for d in days:
                worked_d = model.new_bool_var(f"worked_{e}_{d}_ind")
                shifts_on_d = sum(x[(e, d, s)] for s in shift_names)
                # worked_d == 1 iff shifts_on_d >= 1
                model.add(shifts_on_d >= 1).only_enforce_if(worked_d)
                model.add(shifts_on_d == 0).only_enforce_if(worked_d.negated())

            # Recompute using proper indicators
            worked_vars = []
            for d in days:
                wv = model.new_bool_var(f"wk_{e}_{d}")
                shifts_on_d = sum(x[(e, d, s)] for s in shift_names)
                model.add(shifts_on_d >= 1).only_enforce_if(wv)
                model.add(shifts_on_d == 0).only_enforce_if(wv.negated())
                worked_vars.append(wv)

            days_worked_total = sum(worked_vars)
            days_off_total = scenario.num_days - days_worked_total
            model.add(days_off_total >= wl.min_days_off)

        # Max consecutive working days
        if wl.max_consecutive_days is not None:
            mc = wl.max_consecutive_days
            # For every window of (mc+1) consecutive days, at least one must be off
            for d_start in range(scenario.num_days - mc):
                window = range(d_start, d_start + mc + 1)
                shifts_in_window = sum(
                    x[(e, d, s)] for d in window for s in shift_names
                )
                model.add(shifts_in_window <= mc)

    # ------------------------------------------------------------------
    # Objective: minimize soft penalties + (small) balance term
    # ------------------------------------------------------------------
    # Balance term: minimize the spread of total shifts across employees
    # (encourages even workload distribution)
    if len(employee_ids) > 1:
        totals = []
        for e in employee_ids:
            t = model.new_int_var(0, scenario.num_days, f"total_{e}")
            model.add(t == sum(x[(e, d, s)] for d in days for s in shift_names))
            totals.append(t)

        max_total = model.new_int_var(0, scenario.num_days, "max_total")
        min_total = model.new_int_var(0, scenario.num_days, "min_total")
        model.add_max_equality(max_total, totals)
        model.add_min_equality(min_total, totals)
        spread = model.new_int_var(0, scenario.num_days, "spread")
        model.add(spread == max_total - min_total)

        # Weight: 1 per unit of spread (low priority relative to explicit penalties)
        soft_penalties.append(spread)

    if soft_penalties:
        model.minimize(sum(soft_penalties))
    else:
        # No objective — any feasible solution is fine.
        pass

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_s

    t0 = time.perf_counter()
    status_code = solver.solve(model)
    wall_time = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Map OR-Tools status to our SolverStatus
    # ------------------------------------------------------------------
    if status_code == cp_model.OPTIMAL:
        status = SolverStatus.OPTIMAL
    elif status_code == cp_model.FEASIBLE:
        status = SolverStatus.FEASIBLE
    elif status_code == cp_model.INFEASIBLE:
        return ScheduleSolution(
            status=SolverStatus.INFEASIBLE,
            solver_wall_time_s=round(wall_time, 4),
            infeasibility_reason=_diagnose_infeasibility(scenario),
        )
    else:
        return ScheduleSolution(
            status=SolverStatus.ERROR,
            solver_wall_time_s=round(wall_time, 4),
            infeasibility_reason=f"Solver returned unexpected status: {status_code}",
        )

    # ------------------------------------------------------------------
    # Extract assignments
    # ------------------------------------------------------------------
    assignments: list[Assignment] = []
    for e in employee_ids:
        for d in days:
            for s in shift_names:
                if solver.value(x[(e, d, s)]) == 1:
                    assignments.append(
                        Assignment(
                            employee_id=e,
                            employee_name=employee_name[e],
                            day=d,
                            shift_type=s,
                        )
                    )

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    metrics = _compute_metrics(scenario, assignments)

    return ScheduleSolution(
        status=status,
        assignments=assignments,
        objective_value=solver.objective_value if soft_penalties else 0.0,
        metrics=metrics,
        solver_wall_time_s=round(wall_time, 4),
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _compute_metrics(
    scenario: Scenario, assignments: list[Assignment]
) -> ScheduleMetrics:
    """Derive schedule metrics from the assignment list."""
    employee_name = {e.id: e.name for e in scenario.employees}
    shift_names = {s.name for s in scenario.shift_types}

    # Per-employee stats
    emp_shifts: dict[str, list[Assignment]] = {e.id: [] for e in scenario.employees}
    for a in assignments:
        emp_shifts[a.employee_id].append(a)

    emp_metrics: list[EmployeeMetrics] = []
    for e in scenario.employees:
        mine = emp_shifts[e.id]
        by_type: dict[str, int] = {}
        worked_days: set[int] = set()
        for a in mine:
            by_type[a.shift_type] = by_type.get(a.shift_type, 0) + 1
            worked_days.add(a.day)
        emp_metrics.append(
            EmployeeMetrics(
                employee_id=e.id,
                employee_name=e.name,
                total_shifts=len(mine),
                shifts_by_type=by_type,
                days_off=scenario.num_days - len(worked_days),
            )
        )

    # Coverage: what % of required shift slots are filled?
    total_required_slots = 0
    filled_slots = 0
    for req in scenario.shift_requirements:
        total_required_slots += req.min_employees
        count = sum(
            1
            for a in assignments
            if a.day == req.day and a.shift_type == req.shift_type
        )
        filled_slots += min(count, req.min_employees)

    coverage_pct = (
        (filled_slots / total_required_slots * 100.0) if total_required_slots > 0 else 100.0
    )

    # Utilization: average across employees
    utilization_values = []
    for em in emp_metrics:
        utilization_values.append(em.total_shifts / scenario.num_days * 100.0)
    utilization_pct = (
        sum(utilization_values) / len(utilization_values) if utilization_values else 0.0
    )

    return ScheduleMetrics(
        total_assignments=len(assignments),
        coverage_pct=round(coverage_pct, 2),
        employee_metrics=emp_metrics,
        soft_penalty_total=0,  # TODO: track from solver objective breakdown
        utilization_pct=round(utilization_pct, 2),
    )


def _diagnose_infeasibility(scenario: Scenario) -> str:
    """Best-effort diagnosis of why a scenario is infeasible."""
    reasons: list[str] = []

    num_emp = len(scenario.employees)
    for req in scenario.shift_requirements:
        if req.min_employees > num_emp:
            reasons.append(
                f"Day {req.day} shift '{req.shift_type}' needs {req.min_employees} "
                f"employees but only {num_emp} exist."
            )

    # Check if total shift demand exceeds capacity
    wl = scenario.workload_limits
    if wl.max_shifts_per_employee is not None:
        total_capacity = num_emp * wl.max_shifts_per_employee
        total_demand = sum(r.min_employees for r in scenario.shift_requirements)
        if total_demand > total_capacity:
            reasons.append(
                f"Total shift demand ({total_demand}) exceeds total capacity "
                f"({total_capacity} = {num_emp} employees × "
                f"{wl.max_shifts_per_employee} max shifts each)."
            )

    if wl.min_days_off is not None:
        max_work_days = scenario.num_days - wl.min_days_off
        if max_work_days < 0:
            reasons.append(
                f"min_days_off ({wl.min_days_off}) exceeds schedule length "
                f"({scenario.num_days})."
            )

    # Hard unavailability may over-constrain
    hard_unavail_count = sum(
        1
        for a in scenario.employee_availability
        if a.priority.value == "hard"
    )
    if hard_unavail_count > 0:
        reasons.append(
            f"{hard_unavail_count} hard unavailability constraint(s) may reduce "
            f"available workforce below staffing requirements."
        )

    hard_restriction_count = sum(
        1
        for r in scenario.employee_shift_restrictions
        if r.priority.value == "hard"
    )
    if hard_restriction_count > 0:
        reasons.append(
            f"{hard_restriction_count} hard shift restriction(s) may limit "
            f"eligible employees below staffing requirements."
        )

    if not reasons:
        reasons.append(
            "The combination of constraints makes the problem infeasible. "
            "Try relaxing staffing requirements, availability restrictions, "
            "or workload limits."
        )

    return " | ".join(reasons)
