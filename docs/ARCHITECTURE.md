# Architecture

## Overview

```
UI (Streamlit)
  ↓
Orchestration Layer
  ↓
Natural-Language Parser (LLM)
  ↓
Pydantic Problem Model (Scenario)
  ↓
Validator
  ↓
Optimization Model Builder
  ↓
OR-Tools CP-SAT Solver
  ↓
Solution + Metrics
  ↓
Simulation Engine (what-if)
  ↓
Explanation + Visualization
```

## Module Responsibilities

### models/
Pydantic models defining the data contracts:
- `scenario.py` — Input: employees, shifts, constraints, requirements
- `solution.py` — Output: assignments, metrics, solver status

### solver/
`scheduler.py` — Translates a `Scenario` into an OR-Tools CP-SAT model:
- Boolean variables `x[employee, day, shift]`
- Hard constraints (staffing, availability, restrictions)
- Soft constraints (penalties in objective)
- Workload balancing objective
- Returns `ScheduleSolution` with assignments and metrics

### validation/
`validator.py` — Semantic checks beyond Pydantic schema validation:
- Staffing feasibility
- Workload limit consistency
- Missing requirement warnings

### llm/ (stub)
Abstract `LLMProvider` interface for future NL parsing and explanation.

### parser/ (stub)
Natural language → structured `Scenario` dict.

### simulation/ (stub)
What-if scenario modification and comparison.

### explanation/ (stub)
LLM-based solution explanation from structured data.

### ui/ (stub)
Streamlit web interface.

## Data Flow

1. User input (NL or structured) → `Scenario` model
2. `Scenario` → `validate_scenario()` → `ValidationResult`
3. Valid `Scenario` → `solve()` → `ScheduleSolution`
4. `ScheduleSolution` → metrics, explanation, visualization

## Constraint Model

- **Hard constraints**: Must be satisfied. Solver reports infeasible if impossible.
- **Soft constraints**: Penalized in objective. Solver minimizes total penalty.

The solver never invents a schedule. If infeasible, it reports why.
