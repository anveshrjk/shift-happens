"""Pydantic models for defining a scheduling scenario.

A Scenario is the complete description of a scheduling problem:
employees, shift types, staffing requirements, constraints, and workload limits.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ConstraintPriority(str, Enum):
    """Whether a constraint is hard (must satisfy) or soft (penalized if violated)."""

    HARD = "hard"
    SOFT = "soft"


class ShiftType(BaseModel):
    """A kind of shift, e.g. 'morning', 'afternoon', 'night'."""

    name: str = Field(..., min_length=1, description="Unique shift name")


class Employee(BaseModel):
    """An employee who can be assigned to shifts."""

    id: str = Field(..., min_length=1, description="Unique employee identifier")
    name: str = Field(..., min_length=1, description="Display name")


class ShiftRequirement(BaseModel):
    """Staffing requirement for one shift on one day."""

    day: int = Field(..., ge=0, description="Day index (0-based)")
    shift_type: str = Field(..., min_length=1, description="Name of the ShiftType")
    min_employees: int = Field(..., ge=0, description="Minimum employees needed")
    max_employees: Optional[int] = Field(
        default=None, ge=0, description="Maximum employees allowed (None = no cap)"
    )

    @model_validator(mode="after")
    def max_gte_min(self) -> "ShiftRequirement":
        if self.max_employees is not None and self.max_employees < self.min_employees:
            raise ValueError(
                f"max_employees ({self.max_employees}) must be >= "
                f"min_employees ({self.min_employees})"
            )
        return self


class EmployeeShiftRestriction(BaseModel):
    """An employee cannot (or prefers not to) work a specific shift type."""

    employee_id: str = Field(..., min_length=1)
    shift_type: str = Field(..., min_length=1)
    priority: ConstraintPriority = ConstraintPriority.HARD
    penalty: int = Field(
        default=0,
        ge=0,
        description="Penalty weight if this is a soft constraint",
    )


class EmployeeAvailability(BaseModel):
    """An employee is unavailable on a specific day."""

    employee_id: str = Field(..., min_length=1)
    day: int = Field(..., ge=0, description="Day index (0-based)")
    priority: ConstraintPriority = ConstraintPriority.HARD
    penalty: int = Field(default=0, ge=0)


class WorkloadLimits(BaseModel):
    """Global workload constraints applied to every employee."""

    max_shifts_per_employee: Optional[int] = Field(
        default=None, ge=1, description="Maximum total shifts in the schedule"
    )
    min_shifts_per_employee: Optional[int] = Field(
        default=None, ge=0, description="Minimum total shifts in the schedule"
    )
    max_consecutive_days: Optional[int] = Field(
        default=None, ge=1, description="Maximum consecutive working days"
    )
    min_days_off: Optional[int] = Field(
        default=None, ge=0, description="Minimum days off in the schedule"
    )

    @model_validator(mode="after")
    def min_lte_max(self) -> "WorkloadLimits":
        if (
            self.min_shifts_per_employee is not None
            and self.max_shifts_per_employee is not None
            and self.min_shifts_per_employee > self.max_shifts_per_employee
        ):
            raise ValueError(
                f"min_shifts_per_employee ({self.min_shifts_per_employee}) must be <= "
                f"max_shifts_per_employee ({self.max_shifts_per_employee})"
            )
        return self


class Scenario(BaseModel):
    """Complete scheduling scenario — the single input to the solver."""

    name: str = Field(default="Untitled Scenario", min_length=1)
    num_days: int = Field(..., ge=1, description="Number of days in the schedule")
    employees: list[Employee] = Field(..., min_length=1)
    shift_types: list[ShiftType] = Field(..., min_length=1)
    shift_requirements: list[ShiftRequirement] = Field(default_factory=list)
    employee_shift_restrictions: list[EmployeeShiftRestriction] = Field(
        default_factory=list
    )
    employee_availability: list[EmployeeAvailability] = Field(default_factory=list)
    workload_limits: WorkloadLimits = Field(default_factory=WorkloadLimits)

    # ------------------------------------------------------------------
    # Cross-field validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_references(self) -> "Scenario":
        """Check that all IDs and names referenced in sub-objects exist."""
        employee_ids = {e.id for e in self.employees}
        shift_names = {s.name for s in self.shift_types}

        # Unique employee IDs
        if len(employee_ids) != len(self.employees):
            raise ValueError("Duplicate employee IDs found")

        # Unique shift type names
        if len(shift_names) != len(self.shift_types):
            raise ValueError("Duplicate shift type names found")

        # ShiftRequirement references
        for req in self.shift_requirements:
            if req.shift_type not in shift_names:
                raise ValueError(
                    f"ShiftRequirement references unknown shift type '{req.shift_type}'"
                )
            if req.day >= self.num_days:
                raise ValueError(
                    f"ShiftRequirement day {req.day} is out of range "
                    f"(num_days={self.num_days})"
                )

        # EmployeeShiftRestriction references
        for restr in self.employee_shift_restrictions:
            if restr.employee_id not in employee_ids:
                raise ValueError(
                    f"EmployeeShiftRestriction references unknown employee "
                    f"'{restr.employee_id}'"
                )
            if restr.shift_type not in shift_names:
                raise ValueError(
                    f"EmployeeShiftRestriction references unknown shift type "
                    f"'{restr.shift_type}'"
                )

        # EmployeeAvailability references
        for avail in self.employee_availability:
            if avail.employee_id not in employee_ids:
                raise ValueError(
                    f"EmployeeAvailability references unknown employee "
                    f"'{avail.employee_id}'"
                )
            if avail.day >= self.num_days:
                raise ValueError(
                    f"EmployeeAvailability day {avail.day} is out of range "
                    f"(num_days={self.num_days})"
                )

        # Workload limits vs num_days sanity
        wl = self.workload_limits
        if wl.min_days_off is not None and wl.min_days_off > self.num_days:
            raise ValueError(
                f"min_days_off ({wl.min_days_off}) exceeds num_days ({self.num_days})"
            )
        if (
            wl.max_shifts_per_employee is not None
            and wl.max_shifts_per_employee > self.num_days
        ):
            # Not an error, but we could warn — for now just allow it.
            pass

        return self
