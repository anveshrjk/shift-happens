"""Tests for semantic scenario validation."""

import pytest

from shift_happens.models.scenario import (
    Employee,
    Scenario,
    ShiftRequirement,
    ShiftType,
    WorkloadLimits,
)
from shift_happens.validation.validator import validate_scenario


def _make_scenario(**overrides):
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


class TestValidator:
    def test_valid_scenario(self):
        s = _make_scenario()
        result = validate_scenario(s)
        assert result.valid is True
        assert result.errors == []

    def test_staffing_exceeds_workforce(self):
        s = _make_scenario(
            shift_requirements=[
                ShiftRequirement(day=0, shift_type="morning", min_employees=10)
            ]
        )
        result = validate_scenario(s)
        assert result.valid is False
        assert any("requires at least 10" in e for e in result.errors)

    def test_missing_requirements_warning(self):
        """Warn when some (day, shift) pairs have no requirement."""
        s = _make_scenario()  # no requirements at all
        result = validate_scenario(s)
        assert result.valid is True  # warnings don't make it invalid
        assert len(result.warnings) > 0

    def test_contradictory_workload_limits(self):
        """min_shifts > max_shifts should be caught by Pydantic, but
        the validator also checks days-off consistency."""
        s = _make_scenario(
            workload_limits=WorkloadLimits(
                min_shifts_per_employee=4,
                min_days_off=2,
            ),
            # num_days=3, so max workable = 3-2 = 1, but min_shifts=4
        )
        result = validate_scenario(s)
        assert result.valid is False
        assert any("impossible" in e for e in result.errors)
