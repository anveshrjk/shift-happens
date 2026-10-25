"""Streamlit UI for Shift Happens.

Provides a web interface for:
  1. Entering natural-language scheduling requests
  2. Viewing generated schedules as tables and charts
  3. Running what-if simulations with side-by-side comparison

Requires GEMINI_API_KEY environment variable for live LLM parsing.
Falls back to a direct Scenario builder if no key is available.
"""

from __future__ import annotations

import os
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from shift_happens.models.scenario import (
    ConstraintPriority,
    Employee,
    EmployeeAvailability,
    EmployeeShiftRestriction,
    Scenario,
    ShiftRequirement,
    ShiftType,
    WorkloadLimits,
)
from shift_happens.models.solution import ScheduleSolution, SolverStatus
from shift_happens.explanation.explainer import (
    describe_change,
    explain_schedule,
    explain_simulation,
)
from shift_happens.simulation.engine import (
    ChangeStaffingRequirement,
    ChangeWorkloadLimit,
    SetEmployeeUnavailable,
    SimulationStatus,
    simulate,
)
from shift_happens.solver.scheduler import solve


# ==================================================================
# Page configuration
# ==================================================================

st.set_page_config(
    page_title="Shift Happens",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================================
# Session state helpers
# ==================================================================


def _init_state() -> None:
    """Initialize session state keys on first run."""
    defaults = {
        "scenario": None,
        "solution": None,
        "pipeline_errors": [],
        "pipeline_warnings": [],
        "sim_result": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ==================================================================
# Sidebar — Input mode selection
# ==================================================================


def _sidebar() -> None:
    st.sidebar.title("📅 Shift Happens")
    st.sidebar.caption("AI-Powered Decision & Simulation Engine")
    st.sidebar.divider()

    mode = st.sidebar.radio(
        "Input mode",
        ["Natural Language (LLM)", "Quick Demo (no API key)"],
        help="Natural Language requires a Gemini API key.",
    )

    if mode == "Natural Language (LLM)":
        _nl_input_panel()
    else:
        _demo_input_panel()


# ==================================================================
# Natural-language input
# ==================================================================


def _nl_input_panel() -> None:
    """Sidebar panel for NL scheduling requests."""
    api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        st.sidebar.warning(
            "Set the **GEMINI_API_KEY** environment variable to use live "
            "LLM parsing. Switch to **Quick Demo** mode to try the "
            "scheduler without an API key."
        )

    request_text = st.sidebar.text_area(
        "Scheduling request",
        placeholder=(
            "Schedule 5 employees across morning and night shifts for 7 days. "
            "Each shift needs 2 people. Everyone gets at least 2 days off. "
            "Alice cannot work night shifts."
        ),
        height=150,
    )

    timeout = st.sidebar.slider("Solver timeout (seconds)", 5, 120, 60)

    if st.sidebar.button("🚀 Generate Schedule", type="primary", use_container_width=True):
        if not request_text.strip():
            st.sidebar.error("Please enter a scheduling request.")
            return
        if not api_key:
            st.sidebar.error(
                "GEMINI_API_KEY is not set. Use Quick Demo mode or set the "
                "environment variable and restart the app."
            )
            return
        _run_pipeline(request_text.strip(), timeout)


def _run_pipeline(request: str, timeout: float) -> None:
    """Run the full NL → parse → validate → solve pipeline."""
    from shift_happens.llm.provider import GoogleGeminiProvider, ProviderError
    from shift_happens.pipeline import PipelineStatus, schedule

    # Clear previous results
    st.session_state.scenario = None
    st.session_state.solution = None
    st.session_state.pipeline_errors = []
    st.session_state.pipeline_warnings = []
    st.session_state.sim_result = None

    with st.spinner("Parsing request and generating schedule..."):
        try:
            provider = GoogleGeminiProvider()
        except ProviderError as exc:
            st.session_state.pipeline_errors = [f"Provider setup failed: {exc}"]
            return

        result = schedule(request, provider, solver_timeout_s=timeout)

    st.session_state.pipeline_warnings = result.warnings

    if result.status == PipelineStatus.SUCCESS:
        st.session_state.scenario = result.scenario
        st.session_state.solution = result.solution
    elif result.status == PipelineStatus.INFEASIBLE:
        st.session_state.scenario = result.scenario
        st.session_state.solution = result.solution  # has infeasible status
        st.session_state.pipeline_errors = result.errors
    else:
        st.session_state.scenario = result.scenario
        st.session_state.pipeline_errors = result.errors


# ==================================================================
# Quick demo (no API key needed)
# ==================================================================


def _demo_input_panel() -> None:
    """Sidebar panel for a built-in demo scenario."""
    st.sidebar.markdown("### Demo Settings")

    num_employees = st.sidebar.slider("Employees", 3, 10, 5)
    num_days = st.sidebar.slider("Days", 1, 14, 7)

    shift_options = st.sidebar.multiselect(
        "Shift types",
        ["morning", "afternoon", "night"],
        default=["morning", "night"],
    )
    if not shift_options:
        shift_options = ["morning"]

    min_per_shift = st.sidebar.number_input(
        "Min employees per shift", 1, num_employees, min(2, num_employees)
    )

    use_workload = st.sidebar.checkbox("Add workload limits", value=True)
    max_shifts = None
    min_days_off = None
    if use_workload:
        max_shifts = st.sidebar.number_input(
            "Max shifts per employee", 1, num_days, min(num_days, max(1, (num_days * len(shift_options) * min_per_shift) // num_employees + 1))
        )
        min_days_off = st.sidebar.number_input("Min days off", 0, num_days - 1, min(2, num_days - 1))

    timeout = st.sidebar.slider("Solver timeout (seconds)", 5, 120, 60, key="demo_timeout")

    if st.sidebar.button("🚀 Generate Schedule", type="primary", use_container_width=True):
        _run_demo(num_employees, num_days, shift_options, min_per_shift,
                  max_shifts, min_days_off, timeout)


_DEMO_NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve",
               "Frank", "Grace", "Hank", "Ivy", "Jack"]


def _run_demo(
    num_employees: int,
    num_days: int,
    shift_names: list[str],
    min_per_shift: int,
    max_shifts: int | None,
    min_days_off: int | None,
    timeout: float,
) -> None:
    """Build and solve a demo scenario."""
    # Clear previous results
    st.session_state.scenario = None
    st.session_state.solution = None
    st.session_state.pipeline_errors = []
    st.session_state.pipeline_warnings = []
    st.session_state.sim_result = None

    employees = [
        Employee(id=_DEMO_NAMES[i].lower(), name=_DEMO_NAMES[i])
        for i in range(num_employees)
    ]
    shift_types = [ShiftType(name=n) for n in shift_names]
    reqs = [
        ShiftRequirement(day=d, shift_type=s, min_employees=min_per_shift)
        for d in range(num_days) for s in shift_names
    ]

    wl = WorkloadLimits()
    if max_shifts is not None:
        wl = WorkloadLimits(
            max_shifts_per_employee=max_shifts,
            min_days_off=min_days_off,
        )

    try:
        scenario = Scenario(
            name="Demo Scenario",
            num_days=num_days,
            employees=employees,
            shift_types=shift_types,
            shift_requirements=reqs,
            workload_limits=wl,
        )
    except Exception as exc:
        st.session_state.pipeline_errors = [f"Invalid scenario: {exc}"]
        return

    # Validate semantically
    from shift_happens.validation.validator import validate_scenario
    validation = validate_scenario(scenario)
    if not validation.valid:
        st.session_state.pipeline_errors = validation.errors
        st.session_state.pipeline_warnings = validation.warnings
        return

    st.session_state.pipeline_warnings = validation.warnings

    with st.spinner("Solving schedule..."):
        try:
            solution = solve(scenario, timeout_s=timeout)
        except Exception as exc:
            st.session_state.pipeline_errors = [f"Solver error: {exc}"]
            return

    st.session_state.scenario = scenario
    st.session_state.solution = solution

    if solution.status == SolverStatus.INFEASIBLE:
        reason = solution.infeasibility_reason or "No feasible schedule for these constraints."
        st.session_state.pipeline_errors = [reason]


# ==================================================================
# Main content area
# ==================================================================


def _main_content() -> None:
    """Render the main content panels."""
    scenario: Scenario | None = st.session_state.scenario
    solution: ScheduleSolution | None = st.session_state.solution
    errors: list[str] = st.session_state.pipeline_errors
    warnings: list[str] = st.session_state.pipeline_warnings

    # Welcome screen
    if scenario is None and not errors:
        st.title("📅 Shift Happens")
        st.markdown(
            "**AI-Powered Decision & Simulation Engine** for employee shift scheduling.\n\n"
            "Use the sidebar to enter a scheduling request or try the Quick Demo."
        )
        st.info(
            "**How it works:** Describe your scheduling problem → "
            "the LLM parses it → Pydantic validates it → "
            "OR-Tools solves it → you see the result."
        )
        return

    # Errors with no valid solution
    if errors and (solution is None or solution.status not in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)):
        st.title("📅 Shift Happens")
        if solution and solution.status == SolverStatus.INFEASIBLE:
            st.error("⚠️ No feasible schedule exists for these constraints.")
        else:
            st.error("❌ Schedule generation failed.")
        for e in errors:
            st.warning(e)
        if warnings:
            for w in warnings:
                st.info(f"⚠ {w}")
        return

    # Successful schedule
    st.title(f"📅 {scenario.name}")

    # Warnings
    if warnings:
        for w in warnings:
            st.warning(f"⚠ {w}")

    # Tabs
    tabs = st.tabs(["📊 Schedule", "📈 Visualizations", "💡 Explanation", "🔄 What-If"])

    with tabs[0]:
        _schedule_tab(scenario, solution)

    with tabs[1]:
        _visualization_tab(scenario, solution)

    with tabs[2]:
        _explanation_tab(scenario, solution)

    with tabs[3]:
        _whatif_tab(scenario, solution)


# ==================================================================
# Tab: Schedule table
# ==================================================================


def _schedule_tab(scenario: Scenario, solution: ScheduleSolution) -> None:
    """Display the schedule as a pivot table and metrics summary."""
    # Metrics bar
    m = solution.metrics
    if m:
        cols = st.columns(4)
        cols[0].metric("Status", solution.status.value.title())
        cols[1].metric("Assignments", m.total_assignments)
        cols[2].metric("Coverage", f"{m.coverage_pct:.1f}%")
        cols[3].metric("Utilization", f"{m.utilization_pct:.1f}%")

    if not solution.assignments:
        st.info("No assignments to display.")
        return

    # Build a schedule DataFrame
    df = _assignments_to_df(scenario, solution)
    st.subheader("Schedule Grid")
    st.dataframe(df, use_container_width=True, hide_index=False)

    # Raw assignment list
    with st.expander("Detailed Assignment List"):
        detail_df = pd.DataFrame([
            {
                "Day": a.day,
                "Shift": a.shift_type,
                "Employee": a.employee_name,
                "Employee ID": a.employee_id,
            }
            for a in sorted(solution.assignments, key=lambda x: (x.day, x.shift_type, x.employee_name))
        ])
        st.dataframe(detail_df, use_container_width=True, hide_index=True)


def _assignments_to_df(scenario: Scenario, solution: ScheduleSolution) -> pd.DataFrame:
    """Pivot assignments into a Day × Shift grid showing employee names."""
    shift_names = [s.name for s in scenario.shift_types]

    grid: dict[str, list[str]] = {}
    for day in range(scenario.num_days):
        row_key = f"Day {day}"
        row: dict[str, str] = {}
        for shift in shift_names:
            names = sorted(
                a.employee_name
                for a in solution.assignments
                if a.day == day and a.shift_type == shift
            )
            row[shift] = ", ".join(names) if names else "—"
        grid[row_key] = row

    df = pd.DataFrame.from_dict(grid, orient="index")
    df.columns.name = "Shift"
    return df


# ==================================================================
# Tab: Visualizations
# ==================================================================


def _visualization_tab(scenario: Scenario, solution: ScheduleSolution) -> None:
    """Plotly charts for staffing and workload."""
    if not solution.assignments:
        st.info("No data to visualize.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Staffing by Day & Shift")
        _staffing_chart(scenario, solution)

    with col2:
        st.subheader("Employee Workload")
        _workload_chart(scenario, solution)

    st.subheader("Employee Schedule Heatmap")
    _heatmap_chart(scenario, solution)


def _staffing_chart(scenario: Scenario, solution: ScheduleSolution) -> None:
    """Bar chart: employee count per (day, shift)."""
    records = []
    for day in range(scenario.num_days):
        for shift in scenario.shift_types:
            count = sum(
                1 for a in solution.assignments
                if a.day == day and a.shift_type == shift.name
            )
            # Find the minimum requirement for this slot
            min_req = 0
            for req in scenario.shift_requirements:
                if req.day == day and req.shift_type == shift.name:
                    min_req = req.min_employees
                    break
            records.append({
                "Day": f"Day {day}",
                "Shift": shift.name,
                "Employees": count,
                "Min Required": min_req,
            })

    df = pd.DataFrame(records)
    fig = px.bar(
        df, x="Day", y="Employees", color="Shift",
        barmode="group",
        title="Actual Staffing vs Requirements",
    )

    # Add requirement markers
    for _, row in df.iterrows():
        if row["Min Required"] > 0:
            fig.add_trace(go.Scatter(
                x=[row["Day"]], y=[row["Min Required"]],
                mode="markers", marker=dict(symbol="line-ew-open", size=12, color="red"),
                name="Min Required",
                showlegend=False,
            ))

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def _workload_chart(scenario: Scenario, solution: ScheduleSolution) -> None:
    """Bar chart: shifts per employee."""
    if not solution.metrics or not solution.metrics.employee_metrics:
        st.info("No employee metrics available.")
        return

    em = solution.metrics.employee_metrics
    df = pd.DataFrame([
        {"Employee": e.employee_name, "Shifts": e.total_shifts, "Days Off": e.days_off}
        for e in em
    ])

    fig = px.bar(
        df, x="Employee", y=["Shifts", "Days Off"],
        barmode="stack",
        title="Shifts Worked vs Days Off",
        color_discrete_map={"Shifts": "#636EFA", "Days Off": "#EF553B"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def _heatmap_chart(scenario: Scenario, solution: ScheduleSolution) -> None:
    """Heatmap: employee × day showing shift assignments."""
    emp_names = [e.name for e in scenario.employees]
    days = list(range(scenario.num_days))

    # Build matrix: 0=off, 1..n=shift index
    shift_names = [s.name for s in scenario.shift_types]
    shift_map = {name: i + 1 for i, name in enumerate(shift_names)}

    matrix = []
    text_matrix = []
    for emp in scenario.employees:
        row = []
        text_row = []
        for day in days:
            assigned = [
                a.shift_type for a in solution.assignments
                if a.employee_id == emp.id and a.day == day
            ]
            if assigned:
                row.append(shift_map.get(assigned[0], 1))
                text_row.append(assigned[0])
            else:
                row.append(0)
                text_row.append("off")
        matrix.append(row)
        text_matrix.append(text_row)

    # Color scale: 0=light gray (off), then distinct colors for each shift
    n_shifts = len(shift_names)
    colorscale = [[0, "#f0f0f0"]]
    colors = px.colors.qualitative.Set2[:n_shifts] if n_shifts <= 8 else px.colors.qualitative.Alphabet[:n_shifts]
    for i in range(n_shifts):
        lo = (i + 1) / (n_shifts + 1)
        hi = (i + 2) / (n_shifts + 1)
        colorscale.append([lo, colors[i]])
        colorscale.append([hi, colors[i]])
    # Ensure we end at 1.0
    colorscale[-1][0] = 1.0

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"Day {d}" for d in days],
        y=emp_names,
        text=text_matrix,
        texttemplate="%{text}",
        colorscale=colorscale,
        showscale=False,
    ))
    fig.update_layout(
        height=max(200, 40 * len(emp_names) + 100),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ==================================================================
# Tab: Explanation
# ==================================================================


def _explanation_tab(scenario: Scenario, solution: ScheduleSolution) -> None:
    """Show grounded explanation of the schedule."""
    explanation = explain_schedule(scenario, solution)
    st.markdown(explanation)

    # Per-employee detail
    if solution.metrics and solution.metrics.employee_metrics:
        with st.expander("Per-Employee Details"):
            for em in solution.metrics.employee_metrics:
                parts = [f"**{em.employee_name}**: {em.total_shifts} shifts, {em.days_off} days off"]
                if em.shifts_by_type:
                    breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(em.shifts_by_type.items()))
                    parts.append(f"  ({breakdown})")
                st.markdown(" ".join(parts))


# ==================================================================
# Tab: What-If Simulation
# ==================================================================


def _whatif_tab(scenario: Scenario, solution: ScheduleSolution) -> None:
    """Interface for what-if simulations."""
    if solution.status not in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
        st.info("What-if simulation requires a feasible baseline schedule.")
        return

    st.subheader("Configure What-If Changes")

    changes = []
    emp_ids = [e.id for e in scenario.employees]
    emp_names = {e.id: e.name for e in scenario.employees}
    shift_names = [s.name for s in scenario.shift_types]

    # Change type 1: Employee unavailability
    with st.expander("👤 Make Employee Unavailable", expanded=True):
        c1_emp = st.selectbox("Employee", emp_ids, format_func=lambda x: emp_names[x], key="sim_emp")
        c1_day = st.number_input("Day", 0, scenario.num_days - 1, 0, key="sim_day")
        c1_priority = st.radio("Priority", ["Hard", "Soft"], key="sim_priority", horizontal=True)
        c1_add = st.button("➕ Add unavailability", key="add_unavail")

        if c1_add:
            priority = ConstraintPriority.HARD if c1_priority == "Hard" else ConstraintPriority.SOFT
            change = SetEmployeeUnavailable(
                employee_id=c1_emp, day=c1_day, priority=priority,
                penalty=10 if priority == ConstraintPriority.SOFT else 0,
            )
            if "sim_changes" not in st.session_state:
                st.session_state.sim_changes = []
            st.session_state.sim_changes.append(change)

    # Change type 2: Staffing requirement
    with st.expander("📋 Change Staffing Requirement"):
        c2_day = st.number_input("Day", 0, scenario.num_days - 1, 0, key="sim_staff_day")
        c2_shift = st.selectbox("Shift type", shift_names, key="sim_staff_shift")
        c2_min = st.number_input("Min employees", 0, len(scenario.employees), 1, key="sim_staff_min")
        c2_max_enabled = st.checkbox("Set max employees", key="sim_staff_max_enabled")
        c2_max = None
        if c2_max_enabled:
            c2_max = st.number_input("Max employees", c2_min, len(scenario.employees), max(c2_min, 2), key="sim_staff_max")
        c2_add = st.button("➕ Add staffing change", key="add_staffing")

        if c2_add:
            change = ChangeStaffingRequirement(
                day=c2_day, shift_type=c2_shift,
                min_employees=c2_min, max_employees=c2_max,
            )
            if "sim_changes" not in st.session_state:
                st.session_state.sim_changes = []
            st.session_state.sim_changes.append(change)

    # Change type 3: Workload limits
    with st.expander("⚖️ Change Workload Limits"):
        c3_fields: dict = {}
        c3_max_shifts = st.checkbox("Change max shifts/employee", key="sim_wl_max_enabled")
        if c3_max_shifts:
            c3_fields["max_shifts_per_employee"] = st.number_input(
                "Max shifts/employee", 1, scenario.num_days, 3, key="sim_wl_max"
            )
        c3_min_off = st.checkbox("Change min days off", key="sim_wl_off_enabled")
        if c3_min_off:
            c3_fields["min_days_off"] = st.number_input(
                "Min days off", 0, scenario.num_days - 1, 1, key="sim_wl_off"
            )
        c3_add = st.button("➕ Add workload change", key="add_workload")

        if c3_add and c3_fields:
            change = ChangeWorkloadLimit(**c3_fields)
            if "sim_changes" not in st.session_state:
                st.session_state.sim_changes = []
            st.session_state.sim_changes.append(change)

    # Show queued changes
    sim_changes = st.session_state.get("sim_changes", [])
    if sim_changes:
        st.subheader(f"Queued Changes ({len(sim_changes)})")
        for i, ch in enumerate(sim_changes):
            st.text(f"  {i + 1}. {describe_change(ch)}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Run Simulation", type="primary", use_container_width=True):
                _run_simulation(scenario, solution, sim_changes)
        with col2:
            if st.button("🗑️ Clear Changes", use_container_width=True):
                st.session_state.sim_changes = []
                st.session_state.sim_result = None
                st.rerun()

    # Show simulation result
    sim_result = st.session_state.get("sim_result")
    if sim_result is not None:
        _display_simulation_result(scenario, solution, sim_result)


def _run_simulation(
    scenario: Scenario, solution: ScheduleSolution, changes: list
) -> None:
    """Execute the simulation and store the result."""
    with st.spinner("Running what-if simulation..."):
        result = simulate(scenario, solution, changes)

    st.session_state.sim_result = result
    st.rerun()


def _display_simulation_result(
    baseline_scenario: Scenario,
    baseline_solution: ScheduleSolution,
    result,
) -> None:
    """Render the simulation comparison."""
    st.divider()
    st.subheader("Simulation Result")

    if result.status == SimulationStatus.SUCCESS:
        st.success(f"✅ Simulation succeeded — {result.modified_solution.status.value}")

        # Metrics comparison
        if baseline_solution.metrics and result.modified_solution.metrics:
            bm = baseline_solution.metrics
            mm = result.modified_solution.metrics
            cols = st.columns(4)
            cols[0].metric(
                "Assignments",
                mm.total_assignments,
                delta=mm.total_assignments - bm.total_assignments,
            )
            cols[1].metric(
                "Coverage",
                f"{mm.coverage_pct:.1f}%",
                delta=f"{mm.coverage_pct - bm.coverage_pct:+.1f}%",
            )
            cols[2].metric(
                "Utilization",
                f"{mm.utilization_pct:.1f}%",
                delta=f"{mm.utilization_pct - bm.utilization_pct:+.1f}%",
            )
            cols[3].metric(
                "Soft Penalty",
                mm.soft_penalty_total,
                delta=mm.soft_penalty_total - bm.soft_penalty_total,
                delta_color="inverse",
            )

        # Side-by-side schedules
        col_base, col_mod = st.columns(2)
        with col_base:
            st.markdown("**Baseline Schedule**")
            df_base = _assignments_to_df(baseline_scenario, baseline_solution)
            st.dataframe(df_base, use_container_width=True, hide_index=False)

        with col_mod:
            st.markdown("**Modified Schedule**")
            if result.modified_scenario and result.modified_solution:
                df_mod = _assignments_to_df(result.modified_scenario, result.modified_solution)
                st.dataframe(df_mod, use_container_width=True, hide_index=False)

        # Assignment diff
        if result.assignment_diff:
            diff = result.assignment_diff
            with st.expander(
                f"Assignment Diff: +{len(diff.added)} / -{len(diff.removed)} / "
                f"={diff.unchanged}"
            ):
                if diff.added:
                    st.markdown("**Added:**")
                    for a in diff.added:
                        st.text(f"  + Day {a.day}: {a.employee_name} → {a.shift_type}")
                if diff.removed:
                    st.markdown("**Removed:**")
                    for a in diff.removed:
                        st.text(f"  - Day {a.day}: {a.employee_name} → {a.shift_type}")
                if diff.affected_employees:
                    st.markdown(f"**Affected employees:** {', '.join(sorted(diff.affected_employees))}")
                if diff.affected_days:
                    st.markdown(f"**Affected days:** {', '.join(str(d) for d in sorted(diff.affected_days))}")

        # Grounded explanation
        if result.modified_scenario and result.modified_solution:
            with st.expander("Explanation"):
                changes_desc = [describe_change(c) for c in result.changes_applied]
                explanation = explain_simulation(
                    baseline_scenario, baseline_solution,
                    result.modified_scenario, result.modified_solution,
                    changes_desc,
                )
                st.markdown(explanation)

    elif result.status == SimulationStatus.INFEASIBLE:
        st.error("⚠️ The modified scenario is infeasible — no valid schedule exists.")
        for e in result.errors:
            st.warning(e)

    elif result.status == SimulationStatus.INVALID_CHANGE:
        st.error("❌ Invalid change(s):")
        for e in result.errors:
            st.warning(e)

    elif result.status == SimulationStatus.VALIDATION_FAILED:
        st.error("❌ Modified scenario failed validation:")
        for e in result.errors:
            st.warning(e)

    else:
        st.error(f"❌ Simulation error ({result.status.value}):")
        for e in result.errors:
            st.warning(e)


# ==================================================================
# Entry point
# ==================================================================


def main() -> None:
    """Main Streamlit application."""
    _sidebar()
    _main_content()


if __name__ == "__main__":
    main()
