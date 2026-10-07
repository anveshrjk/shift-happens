"""Tests for Pydantic scenario and solution models."""

import pytest
from pydantic import ValidationError

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
from shift_happens.models.solution import (
    Assignment,
    EmployeeMetrics,
    ScheduleMetrics,
    ScheduleSolution,
    SolverStatus,
)


# ==================================================================
# Scenario model tests
# ==================================================================


class TestShiftRequirement:
    def test_valid(self):
        req = ShiftRequirement(day=0, shift_type="morning", min_employees=2)
        assert req.min_employees == 2
        assert req.max_employees is None

    def test_max_gte_min(self):
        req = ShiftRequirement(
            day=0, shift_type="morning", min_employees=2, max_employees=5
        )
        assert req.max_employees == 5

    def test_max_lt_min_raises(self):
        with pytest.raises(ValidationError, match="max_employees"):
            ShiftRequirement(
                day=0, shift_type="morning", min_employees=5, max_employees=2
            )

    def test_negative_day_raises(self):
        with pytest.raises(ValidationError):
            ShiftRequirement(day=-1, shift_type="morning", min_employees=1)


class TestWorkloadLimits:
    def test_defaults(self):
        wl = WorkloadLimits()
        assert wl.max_shifts_per_employee is None
        assert wl.min_shifts_per_employee is None

    def test_min_gt_max_raises(self):
        with pytest.raises(ValidationError, match="min_shifts_per_employee"):
            WorkloadLimits(min_shifts_per_employee=10, max_shifts_per_employee=5)


class TestScenario:
    def _minimal_scenario(self, **overrides):
        defaults = dict(
            num_days=3,
            employees=[
                Employee(id="alice", name="Alice"),
                Employee(id="bob", name="Bob"),
            ],
            shift_types=[ShiftType(name="morning"), ShiftType(name="night")],
        )
        defaults.update(overrides)
        return Scenario(**defaults)

    def test_minimal_valid(self):
        s = self._minimal_scenario()
        assert s.num_days == 3
        assert len(s.employees) == 2
        assert len(s.shift_types) == 2
        assert s.shift_requirements == []

    def test_with_requirements(self):
        s = self._minimal_scenario(
            shift_requirements=[
                ShiftRequirement(day=0, shift_type="morning", min_employees=1),
                ShiftRequirement(day=1, shift_type="night", min_employees=1),
            ]
        )
        assert len(s.shift_requirements) == 2

    def test_duplicate_employee_ids_raises(self):
        with pytest.raises(ValidationError, match="Duplicate employee IDs"):
            Scenario(
                num_days=3,
                employees=[
                    Employee(id="alice", name="Alice"),
                    Employee(id="alice", name="Alice2"),
                ],
                shift_types=[ShiftType(name="morning")],
            )

    def test_duplicate_shift_names_raises(self):
        with pytest.raises(ValidationError, match="Duplicate shift type names"):
            Scenario(
                num_days=3,
                employees=[Employee(id="alice", name="Alice")],
                shift_types=[
                    ShiftType(name="morning"),
                    ShiftType(name="morning"),
                ],
            )

    def test_unknown_shift_in_requirement_raises(self):
        with pytest.raises(ValidationError, match="unknown shift type"):
            self._minimal_scenario(
                shift_requirements=[
                    ShiftRequirement(
                        day=0, shift_type="nonexistent", min_employees=1
                    )
                ]
            )

    def test_day_out_of_range_in_requirement_raises(self):
        with pytest.raises(ValidationError, match="out of range"):
            self._minimal_scenario(
                shift_requirements=[
                    ShiftRequirement(day=99, shift_type="morning", min_employees=1)
                ]
            )

    def test_unknown_employee_in_restriction_raises(self):
        with pytest.raises(ValidationError, match="unknown employee"):
            self._minimal_scenario(
                employee_shift_restrictions=[
                    EmployeeShiftRestriction(
                        employee_id="nobody", shift_type="morning"
                    )
                ]
            )

    def test_unknown_shift_in_restriction_raises(self):
        with pytest.raises(ValidationError, match="unknown shift type"):
            self._minimal_scenario(
                employee_shift_restrictions=[
                    EmployeeShiftRestriction(
                        employee_id="alice", shift_type="nonexistent"
                    )
                ]
            )

    def test_unknown_employee_in_availability_raises(self):
        with pytest.raises(ValidationError, match="unknown employee"):
            self._minimal_scenario(
                employee_availability=[
                    EmployeeAvailability(employee_id="nobody", day=0)
                ]
            )

    def test_availability_day_out_of_range_raises(self):
        with pytest.raises(ValidationError, match="out of range"):
            self._minimal_scenario(
                employee_availability=[
                    EmployeeAvailability(employee_id="alice", day=99)
                ]
            )

    def test_no_employees_raises(self):
        with pytest.raises(ValidationError):
            Scenario(
                num_days=3,
                employees=[],
                shift_types=[ShiftType(name="morning")],
            )

    def test_no_shift_types_raises(self):
        with pytest.raises(ValidationError):
            Scenario(
                num_days=3,
                employees=[Employee(id="alice", name="Alice")],
                shift_types=[],
            )

    def test_zero_days_raises(self):
        with pytest.raises(ValidationError):
            Scenario(
                num_days=0,
                employees=[Employee(id="alice", name="Alice")],
                shift_types=[ShiftType(name="morning")],
            )

    def test_serialization_roundtrip(self):
        s = self._minimal_scenario()
        data = s.model_dump()
        s2 = Scenario.model_validate(data)
        assert s == s2


# ==================================================================
# Solution model tests
# ==================================================================


class TestSolutionModels:
    def test_assignment(self):
        a = Assignment(
            employee_id="alice",
            employee_name="Alice",
            day=0,
            shift_type="morning",
        )
        assert a.employee_id == "alice"

    def test_schedule_solution_infeasible(self):
        sol = ScheduleSolution(status=SolverStatus.INFEASIBLE)
        assert sol.status == SolverStatus.INFEASIBLE
        assert sol.assignments == []
        assert sol.metrics is None

    def test_schedule_solution_optimal(self):
        sol = ScheduleSolution(
            status=SolverStatus.OPTIMAL,
            assignments=[
                Assignment(
                    employee_id="a",
                    employee_name="Alice",
                    day=0,
                    shift_type="morning",
                )
            ],
            objective_value=0.0,
            metrics=ScheduleMetrics(
                total_assignments=1,
                coverage_pct=100.0,
                soft_penalty_total=0,
            ),
        )
        assert sol.status == SolverStatus.OPTIMAL
        assert len(sol.assignments) == 1
