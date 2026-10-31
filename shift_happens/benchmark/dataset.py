"""Fixed, versioned benchmark dataset for Shift Happens evaluation.

Each BenchmarkCase bundles a Scenario (or raw dict for invalid-schema tests),
expected solver outcome, expected validation result, an optional NL prompt
with expected facts, and optional what-if simulation changes with expected
outcomes.

The dataset is deterministic — no random generation at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

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
from shift_happens.simulation.engine import (
    ChangeStaffingRequirement,
    ChangeWorkloadLimit,
    ScenarioChange,
    SetEmployeeUnavailable,
)


DATASET_VERSION = "1.0.0"


class ExpectedSolverOutcome(str, Enum):
    """What the solver should report."""
    OPTIMAL_OR_FEASIBLE = "optimal_or_feasible"
    INFEASIBLE = "infeasible"


class ExpectedSimOutcome(str, Enum):
    """Expected simulation result status."""
    SUCCESS = "success"
    INFEASIBLE = "infeasible"
    INVALID_CHANGE = "invalid_change"
    VALIDATION_FAILED = "validation_failed"


@dataclass
class NLExpectedFacts:
    """Facts a parser should extract from an NL prompt."""
    employee_count: int | None = None
    employee_names: list[str] | None = None
    num_days: int | None = None
    shift_types: list[str] | None = None
    min_per_shift: int | None = None
    has_availability_constraints: bool = False
    has_shift_restrictions: bool = False
    has_workload_limits: bool = False


@dataclass
class SimulationCase:
    """A what-if simulation sub-case within a benchmark case."""
    description: str
    changes: list[ScenarioChange]
    expected_outcome: ExpectedSimOutcome


@dataclass
class BenchmarkCase:
    """A single benchmark scenario with expectations."""
    id: str
    name: str
    category: str
    description: str
    scenario: Scenario
    expected_solver: ExpectedSolverOutcome
    expected_valid: bool = True  # Should validate_scenario pass?
    nl_prompt: str | None = None
    nl_expected: NLExpectedFacts | None = None
    simulation_cases: list[SimulationCase] = field(default_factory=list)


# ==================================================================
# Helper builders
# ==================================================================

_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace",
    "Hank", "Ivy", "Jack", "Kate", "Leo", "Mia", "Nick", "Olivia",
]


def _emps(n: int) -> list[Employee]:
    return [Employee(id=_NAMES[i].lower(), name=_NAMES[i]) for i in range(n)]


def _shifts(*names: str) -> list[ShiftType]:
    return [ShiftType(name=n) for n in names]


def _reqs(days: int, shifts: list[str], min_e: int) -> list[ShiftRequirement]:
    return [
        ShiftRequirement(day=d, shift_type=s, min_employees=min_e)
        for d in range(days) for s in shifts
    ]


# ==================================================================
# The benchmark dataset
# ==================================================================

def load_dataset() -> list[BenchmarkCase]:
    """Return the complete fixed benchmark dataset."""
    return list(_ALL_CASES)


def _build_cases() -> list[BenchmarkCase]:
    """Construct all benchmark cases."""
    cases: list[BenchmarkCase] = []

    # ------------------------------------------------------------------
    # Category 1: Trivial feasible
    # ------------------------------------------------------------------
    cases.append(BenchmarkCase(
        id="trivial-1x1x1",
        name="Trivial 1 employee 1 shift 1 day",
        category="trivial",
        description="Minimum possible scheduling problem.",
        scenario=Scenario(
            name="Trivial",
            num_days=1,
            employees=_emps(1),
            shift_types=_shifts("morning"),
            shift_requirements=_reqs(1, ["morning"], 1),
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
    ))

    cases.append(BenchmarkCase(
        id="trivial-2x1x1",
        name="2 employees 1 shift 1 day",
        category="trivial",
        description="Two employees competing for one slot.",
        scenario=Scenario(
            name="Two for one",
            num_days=1,
            employees=_emps(2),
            shift_types=_shifts("morning"),
            shift_requirements=_reqs(1, ["morning"], 1),
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
    ))

    # ------------------------------------------------------------------
    # Category 2: Standard feasible
    # ------------------------------------------------------------------
    cases.append(BenchmarkCase(
        id="standard-3x2x5",
        name="3 employees 2 shifts 5 days",
        category="standard",
        description="Small but realistic problem.",
        scenario=Scenario(
            name="Small Office",
            num_days=5,
            employees=_emps(3),
            shift_types=_shifts("morning", "night"),
            shift_requirements=_reqs(5, ["morning", "night"], 1),
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
        nl_prompt=(
            "Schedule 3 employees (Alice, Bob, Carol) for 5 days "
            "with morning and night shifts. Each shift needs 1 person."
        ),
        nl_expected=NLExpectedFacts(
            employee_count=3,
            employee_names=["Alice", "Bob", "Carol"],
            num_days=5,
            shift_types=["morning", "night"],
            min_per_shift=1,
        ),
        simulation_cases=[
            SimulationCase(
                description="Alice unavailable day 0",
                changes=[SetEmployeeUnavailable(employee_id="alice", day=0)],
                expected_outcome=ExpectedSimOutcome.SUCCESS,
            ),
        ],
    ))

    cases.append(BenchmarkCase(
        id="standard-5x2x7",
        name="5 employees 2 shifts 7 days",
        category="standard",
        description="Week-long schedule with morning/night shifts.",
        scenario=Scenario(
            name="Weekly Roster",
            num_days=7,
            employees=_emps(5),
            shift_types=_shifts("morning", "night"),
            shift_requirements=_reqs(7, ["morning", "night"], 1),
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
        nl_prompt=(
            "Schedule 5 employees across morning and night shifts for 7 days. "
            "Each shift needs at least 1 person."
        ),
        nl_expected=NLExpectedFacts(
            employee_count=5, num_days=7,
            shift_types=["morning", "night"], min_per_shift=1,
        ),
    ))

    cases.append(BenchmarkCase(
        id="standard-5x3x7",
        name="5 employees 3 shifts 7 days",
        category="standard",
        description="Three-shift weekly schedule.",
        scenario=Scenario(
            name="Three Shift Week",
            num_days=7,
            employees=_emps(5),
            shift_types=_shifts("morning", "afternoon", "night"),
            shift_requirements=_reqs(7, ["morning", "afternoon", "night"], 1),
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
    ))

    cases.append(BenchmarkCase(
        id="standard-8x2x7",
        name="8 employees 2 shifts 7 days",
        category="standard",
        description="Larger crew, 2 per shift.",
        scenario=Scenario(
            name="Larger Crew",
            num_days=7,
            employees=_emps(8),
            shift_types=_shifts("morning", "night"),
            shift_requirements=_reqs(7, ["morning", "night"], 2),
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
        nl_prompt=(
            "Schedule 8 employees for 7 days with morning and night shifts. "
            "Each shift needs at least 2 employees."
        ),
        nl_expected=NLExpectedFacts(
            employee_count=8, num_days=7,
            shift_types=["morning", "night"], min_per_shift=2,
        ),
    ))

    cases.append(BenchmarkCase(
        id="standard-10x3x14",
        name="10 employees 3 shifts 14 days",
        category="standard",
        description="Two-week schedule with three shifts.",
        scenario=Scenario(
            name="Two Week Roster",
            num_days=14,
            employees=_emps(10),
            shift_types=_shifts("morning", "afternoon", "night"),
            shift_requirements=_reqs(14, ["morning", "afternoon", "night"], 2),
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
    ))

    # ------------------------------------------------------------------
    # Category 3: With workload limits
    # ------------------------------------------------------------------
    cases.append(BenchmarkCase(
        id="workload-max-shifts",
        name="Max shifts per employee",
        category="workload",
        description="5 emp, 7 days, max 4 shifts each.",
        scenario=Scenario(
            name="Max Shifts",
            num_days=7,
            employees=_emps(5),
            shift_types=_shifts("morning", "night"),
            shift_requirements=_reqs(7, ["morning", "night"], 1),
            workload_limits=WorkloadLimits(max_shifts_per_employee=4),
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
        simulation_cases=[
            SimulationCase(
                description="Reduce max shifts to 3",
                changes=[ChangeWorkloadLimit(max_shifts_per_employee=3)],
                expected_outcome=ExpectedSimOutcome.SUCCESS,
            ),
        ],
    ))

    cases.append(BenchmarkCase(
        id="workload-days-off",
        name="Minimum days off",
        category="workload",
        description="5 emp, 7 days, 2 days off minimum.",
        scenario=Scenario(
            name="Days Off",
            num_days=7,
            employees=_emps(5),
            shift_types=_shifts("morning", "night"),
            shift_requirements=_reqs(7, ["morning", "night"], 1),
            workload_limits=WorkloadLimits(min_days_off=2),
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
        nl_prompt=(
            "Schedule 5 employees for 7 days, morning and night shifts. "
            "Each shift needs 1 person. Everyone gets at least 2 days off."
        ),
        nl_expected=NLExpectedFacts(
            employee_count=5, num_days=7,
            shift_types=["morning", "night"], min_per_shift=1,
            has_workload_limits=True,
        ),
    ))

    cases.append(BenchmarkCase(
        id="workload-consecutive",
        name="Max consecutive days",
        category="workload",
        description="5 emp, 7 days, max 3 consecutive working days.",
        scenario=Scenario(
            name="Consecutive Limit",
            num_days=7,
            employees=_emps(5),
            shift_types=_shifts("morning"),
            shift_requirements=_reqs(7, ["morning"], 1),
            workload_limits=WorkloadLimits(max_consecutive_days=3),
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
    ))

    cases.append(BenchmarkCase(
        id="workload-combined",
        name="Combined workload limits",
        category="workload",
        description="Max shifts + min days off + consecutive limit.",
        scenario=Scenario(
            name="Combined Limits",
            num_days=7,
            employees=_emps(6),
            shift_types=_shifts("morning", "night"),
            shift_requirements=_reqs(7, ["morning", "night"], 1),
            workload_limits=WorkloadLimits(
                max_shifts_per_employee=4,
                min_days_off=2,
                max_consecutive_days=3,
            ),
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
    ))

    # ------------------------------------------------------------------
    # Category 4: Availability restrictions
    # ------------------------------------------------------------------
    cases.append(BenchmarkCase(
        id="avail-single",
        name="Single unavailability",
        category="availability",
        description="Alice cannot work day 0.",
        scenario=Scenario(
            name="Single Unavail",
            num_days=5,
            employees=_emps(4),
            shift_types=_shifts("morning", "night"),
            shift_requirements=_reqs(5, ["morning", "night"], 1),
            employee_availability=[
                EmployeeAvailability(employee_id="alice", day=0),
            ],
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
    ))

    cases.append(BenchmarkCase(
        id="avail-multiple",
        name="Multiple unavailabilities",
        category="availability",
        description="Several employees with scattered unavailabilities.",
        scenario=Scenario(
            name="Multiple Unavail",
            num_days=5,
            employees=_emps(5),
            shift_types=_shifts("morning", "night"),
            shift_requirements=_reqs(5, ["morning", "night"], 1),
            employee_availability=[
                EmployeeAvailability(employee_id="alice", day=0),
                EmployeeAvailability(employee_id="alice", day=1),
                EmployeeAvailability(employee_id="bob", day=2),
                EmployeeAvailability(employee_id="carol", day=3),
                EmployeeAvailability(employee_id="dave", day=4),
            ],
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
    ))

    cases.append(BenchmarkCase(
        id="avail-soft",
        name="Soft unavailability",
        category="availability",
        description="Alice prefers not to work day 0 (soft constraint).",
        scenario=Scenario(
            name="Soft Unavail",
            num_days=3,
            employees=_emps(3),
            shift_types=_shifts("morning", "night"),
            shift_requirements=_reqs(3, ["morning", "night"], 1),
            employee_availability=[
                EmployeeAvailability(
                    employee_id="alice", day=0,
                    priority=ConstraintPriority.SOFT, penalty=10,
                ),
            ],
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
    ))

    # ------------------------------------------------------------------
    # Category 5: Shift restrictions
    # ------------------------------------------------------------------
    cases.append(BenchmarkCase(
        id="restrict-single",
        name="Single shift restriction",
        category="restrictions",
        description="Alice cannot work night shifts.",
        scenario=Scenario(
            name="Night Restriction",
            num_days=5,
            employees=_emps(4),
            shift_types=_shifts("morning", "night"),
            shift_requirements=_reqs(5, ["morning", "night"], 1),
            employee_shift_restrictions=[
                EmployeeShiftRestriction(employee_id="alice", shift_type="night"),
            ],
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
        nl_prompt=(
            "Schedule 4 employees for 5 days with morning and night shifts. "
            "Each shift needs 1 person. Alice cannot work night shifts."
        ),
        nl_expected=NLExpectedFacts(
            employee_count=4, num_days=5,
            shift_types=["morning", "night"], min_per_shift=1,
            has_shift_restrictions=True,
        ),
    ))

    cases.append(BenchmarkCase(
        id="restrict-soft",
        name="Soft shift restriction",
        category="restrictions",
        description="Bob prefers not to work morning (soft).",
        scenario=Scenario(
            name="Soft Restriction",
            num_days=3,
            employees=_emps(3),
            shift_types=_shifts("morning", "night"),
            shift_requirements=_reqs(3, ["morning", "night"], 1),
            employee_shift_restrictions=[
                EmployeeShiftRestriction(
                    employee_id="bob", shift_type="morning",
                    priority=ConstraintPriority.SOFT, penalty=5,
                ),
            ],
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
    ))

    # ------------------------------------------------------------------
    # Category 6: Max employees cap
    # ------------------------------------------------------------------
    cases.append(BenchmarkCase(
        id="max-emp-cap",
        name="Max employees cap on shift",
        category="standard",
        description="Shift has both min and max employees.",
        scenario=Scenario(
            name="Max Cap",
            num_days=3,
            employees=_emps(5),
            shift_types=_shifts("morning"),
            shift_requirements=[
                ShiftRequirement(day=d, shift_type="morning",
                                 min_employees=1, max_employees=2)
                for d in range(3)
            ],
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
    ))

    # ------------------------------------------------------------------
    # Category 7: No requirements (all optional)
    # ------------------------------------------------------------------
    cases.append(BenchmarkCase(
        id="no-requirements",
        name="No staffing requirements",
        category="edge",
        description="No shift requirements — solver should produce a feasible result with minimal assignments.",
        scenario=Scenario(
            name="No Requirements",
            num_days=3,
            employees=_emps(3),
            shift_types=_shifts("morning"),
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
    ))

    # ------------------------------------------------------------------
    # Category 8: Infeasible scenarios
    # ------------------------------------------------------------------
    cases.append(BenchmarkCase(
        id="invalid-overstaff",
        name="More required than available (validation failure)",
        category="invalid",
        description="Need 5 per shift but only have 3 employees. Semantic validation rejects this before the solver runs.",
        scenario=Scenario(
            name="Overstaffed",
            num_days=1,
            employees=_emps(3),
            shift_types=_shifts("morning"),
            shift_requirements=[
                ShiftRequirement(day=0, shift_type="morning", min_employees=5),
            ],
        ),
        expected_solver=ExpectedSolverOutcome.INFEASIBLE,  # Would be infeasible if it reached the solver
        expected_valid=False,  # Semantic validation catches this — solver is NOT called
    ))

    cases.append(BenchmarkCase(
        id="infeasible-all-unavail",
        name="All employees unavailable",
        category="infeasible",
        description="Everyone is unavailable on day 0, but a shift is required.",
        scenario=Scenario(
            name="All Unavailable",
            num_days=2,
            employees=_emps(3),
            shift_types=_shifts("morning"),
            shift_requirements=_reqs(2, ["morning"], 1),
            employee_availability=[
                EmployeeAvailability(employee_id="alice", day=0),
                EmployeeAvailability(employee_id="bob", day=0),
                EmployeeAvailability(employee_id="carol", day=0),
            ],
        ),
        expected_solver=ExpectedSolverOutcome.INFEASIBLE,
    ))

    cases.append(BenchmarkCase(
        id="infeasible-concurrent",
        name="Concurrent shift conflict",
        category="infeasible",
        description="1 employee, 2 shifts on same day both requiring 1.",
        scenario=Scenario(
            name="Concurrent Conflict",
            num_days=1,
            employees=_emps(1),
            shift_types=_shifts("morning", "night"),
            shift_requirements=[
                ShiftRequirement(day=0, shift_type="morning", min_employees=1),
                ShiftRequirement(day=0, shift_type="night", min_employees=1),
            ],
        ),
        expected_solver=ExpectedSolverOutcome.INFEASIBLE,
    ))

    cases.append(BenchmarkCase(
        id="infeasible-workload",
        name="Infeasible workload limits",
        category="infeasible",
        description="Max 1 shift per emp but need 6 slots from 3 employees.",
        scenario=Scenario(
            name="Workload Infeasible",
            num_days=3,
            employees=_emps(3),
            shift_types=_shifts("morning", "night"),
            shift_requirements=_reqs(3, ["morning", "night"], 1),
            workload_limits=WorkloadLimits(max_shifts_per_employee=1),
        ),
        expected_solver=ExpectedSolverOutcome.INFEASIBLE,
    ))

    # ------------------------------------------------------------------
    # Category 9: Larger / stress scenarios
    # ------------------------------------------------------------------
    cases.append(BenchmarkCase(
        id="large-15x3x14",
        name="15 employees 3 shifts 14 days",
        category="large",
        description="Larger scheduling problem for performance testing.",
        scenario=Scenario(
            name="Large Roster",
            num_days=14,
            employees=_emps(15),
            shift_types=_shifts("morning", "afternoon", "night"),
            shift_requirements=_reqs(14, ["morning", "afternoon", "night"], 3),
            workload_limits=WorkloadLimits(
                max_shifts_per_employee=10,
                min_days_off=3,
            ),
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
    ))

    cases.append(BenchmarkCase(
        id="large-10x2x14-constrained",
        name="10 employees highly constrained 14 days",
        category="large",
        description="Many restrictions and availability constraints.",
        scenario=Scenario(
            name="Highly Constrained",
            num_days=14,
            employees=_emps(10),
            shift_types=_shifts("morning", "night"),
            shift_requirements=_reqs(14, ["morning", "night"], 2),
            employee_shift_restrictions=[
                EmployeeShiftRestriction(employee_id="alice", shift_type="night"),
                EmployeeShiftRestriction(employee_id="bob", shift_type="night"),
                EmployeeShiftRestriction(employee_id="carol", shift_type="morning"),
            ],
            employee_availability=[
                EmployeeAvailability(employee_id="dave", day=0),
                EmployeeAvailability(employee_id="dave", day=1),
                EmployeeAvailability(employee_id="eve", day=5),
                EmployeeAvailability(employee_id="eve", day=6),
                EmployeeAvailability(employee_id="frank", day=10),
                EmployeeAvailability(employee_id="frank", day=11),
            ],
            workload_limits=WorkloadLimits(
                max_shifts_per_employee=8,
                min_days_off=4,
                max_consecutive_days=4,
            ),
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
    ))

    # ------------------------------------------------------------------
    # Category 10: Simulation test cases
    # ------------------------------------------------------------------
    sim_base = Scenario(
        name="Simulation Base",
        num_days=5,
        employees=_emps(5),
        shift_types=_shifts("morning", "night"),
        shift_requirements=_reqs(5, ["morning", "night"], 1),
        workload_limits=WorkloadLimits(max_shifts_per_employee=4, min_days_off=1),
    )

    cases.append(BenchmarkCase(
        id="sim-unavailable",
        name="Simulation: employee unavailable",
        category="simulation",
        description="Make Alice unavailable on day 2.",
        scenario=sim_base,
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
        simulation_cases=[
            SimulationCase(
                description="Alice unavailable day 2",
                changes=[SetEmployeeUnavailable(employee_id="alice", day=2)],
                expected_outcome=ExpectedSimOutcome.SUCCESS,
            ),
        ],
    ))

    cases.append(BenchmarkCase(
        id="sim-increase-staffing",
        name="Simulation: increase staffing",
        category="simulation",
        description="Increase morning staffing on day 0 to 2.",
        scenario=sim_base,
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
        simulation_cases=[
            SimulationCase(
                description="Day 0 morning min=2",
                changes=[ChangeStaffingRequirement(
                    day=0, shift_type="morning", min_employees=2,
                )],
                expected_outcome=ExpectedSimOutcome.SUCCESS,
            ),
        ],
    ))

    cases.append(BenchmarkCase(
        id="sim-tighten-workload",
        name="Simulation: tighten workload to infeasible",
        category="simulation",
        description="Reduce max shifts to make it infeasible.",
        scenario=Scenario(
            name="Sim Tight Base",
            num_days=3,
            employees=_emps(3),
            shift_types=_shifts("morning", "night"),
            shift_requirements=_reqs(3, ["morning", "night"], 1),
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
        simulation_cases=[
            SimulationCase(
                description="Max shifts=1 makes it infeasible",
                changes=[ChangeWorkloadLimit(max_shifts_per_employee=1)],
                expected_outcome=ExpectedSimOutcome.INFEASIBLE,
            ),
        ],
    ))

    cases.append(BenchmarkCase(
        id="sim-invalid-employee",
        name="Simulation: invalid employee ref",
        category="simulation",
        description="Reference a non-existent employee.",
        scenario=sim_base,
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
        simulation_cases=[
            SimulationCase(
                description="Unknown employee 'nobody'",
                changes=[SetEmployeeUnavailable(employee_id="nobody", day=0)],
                expected_outcome=ExpectedSimOutcome.INVALID_CHANGE,
            ),
        ],
    ))

    cases.append(BenchmarkCase(
        id="sim-invalid-shift",
        name="Simulation: invalid shift ref",
        category="simulation",
        description="Reference a non-existent shift type.",
        scenario=sim_base,
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
        simulation_cases=[
            SimulationCase(
                description="Unknown shift 'graveyard'",
                changes=[ChangeStaffingRequirement(
                    day=0, shift_type="graveyard", min_employees=1,
                )],
                expected_outcome=ExpectedSimOutcome.INVALID_CHANGE,
            ),
        ],
    ))

    cases.append(BenchmarkCase(
        id="sim-multiple-changes",
        name="Simulation: multiple changes",
        category="simulation",
        description="Apply several changes at once.",
        scenario=sim_base,
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
        simulation_cases=[
            SimulationCase(
                description="Alice unavail + increase staffing",
                changes=[
                    SetEmployeeUnavailable(employee_id="alice", day=0),
                    ChangeStaffingRequirement(
                        day=1, shift_type="night", min_employees=2,
                    ),
                ],
                expected_outcome=ExpectedSimOutcome.SUCCESS,
            ),
        ],
    ))

    # ------------------------------------------------------------------
    # Category 11: Edge cases
    # ------------------------------------------------------------------
    cases.append(BenchmarkCase(
        id="edge-single-day",
        name="Single day schedule",
        category="edge",
        description="Only 1 day in the schedule.",
        scenario=Scenario(
            name="Single Day",
            num_days=1,
            employees=_emps(5),
            shift_types=_shifts("morning", "afternoon", "night"),
            shift_requirements=_reqs(1, ["morning", "afternoon", "night"], 1),
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
    ))

    cases.append(BenchmarkCase(
        id="edge-many-emp-few-slots",
        name="Many employees few slots",
        category="edge",
        description="10 employees but only 1 slot needed.",
        scenario=Scenario(
            name="Overstaffed Pool",
            num_days=1,
            employees=_emps(10),
            shift_types=_shifts("morning"),
            shift_requirements=[
                ShiftRequirement(day=0, shift_type="morning", min_employees=1),
            ],
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
    ))

    cases.append(BenchmarkCase(
        id="edge-min-eq-max",
        name="Min equals max employees",
        category="edge",
        description="Exactly 2 employees per shift required.",
        scenario=Scenario(
            name="Exact Staffing",
            num_days=3,
            employees=_emps(4),
            shift_types=_shifts("morning"),
            shift_requirements=[
                ShiftRequirement(day=d, shift_type="morning",
                                 min_employees=2, max_employees=2)
                for d in range(3)
            ],
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
    ))

    # ------------------------------------------------------------------
    # Category 12: NL-only test cases
    # ------------------------------------------------------------------
    cases.append(BenchmarkCase(
        id="nl-complex",
        name="NL: complex request",
        category="nl",
        description="Complex NL prompt with multiple constraints.",
        scenario=Scenario(
            name="Complex NL",
            num_days=7,
            employees=_emps(6),
            shift_types=_shifts("morning", "afternoon", "night"),
            shift_requirements=_reqs(7, ["morning", "afternoon", "night"], 1),
            employee_shift_restrictions=[
                EmployeeShiftRestriction(employee_id="alice", shift_type="night"),
            ],
            employee_availability=[
                EmployeeAvailability(employee_id="bob", day=0),
                EmployeeAvailability(employee_id="bob", day=6),
            ],
            workload_limits=WorkloadLimits(
                max_shifts_per_employee=5, min_days_off=2,
            ),
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
        nl_prompt=(
            "Schedule 6 employees for 7 days with morning, afternoon, and "
            "night shifts. Each shift needs 1 person. Alice cannot work "
            "night shifts. Bob is unavailable on day 0 and day 6. "
            "Max 5 shifts per person, everyone gets at least 2 days off."
        ),
        nl_expected=NLExpectedFacts(
            employee_count=6, num_days=7,
            shift_types=["morning", "afternoon", "night"],
            min_per_shift=1,
            has_shift_restrictions=True,
            has_availability_constraints=True,
            has_workload_limits=True,
        ),
    ))

    cases.append(BenchmarkCase(
        id="nl-simple",
        name="NL: simple request",
        category="nl",
        description="Straightforward NL prompt.",
        scenario=Scenario(
            name="Simple NL",
            num_days=5,
            employees=_emps(4),
            shift_types=_shifts("morning", "night"),
            shift_requirements=_reqs(5, ["morning", "night"], 1),
        ),
        expected_solver=ExpectedSolverOutcome.OPTIMAL_OR_FEASIBLE,
        nl_prompt=(
            "I need a schedule for 4 employees working morning and night "
            "shifts over 5 days. At least 1 person per shift."
        ),
        nl_expected=NLExpectedFacts(
            employee_count=4, num_days=5,
            shift_types=["morning", "night"], min_per_shift=1,
        ),
    ))

    return cases


_ALL_CASES = _build_cases()
