# SHIFT HAPPENS

**AI-Powered Decision & Simulation Engine**

Shift Happens is an AI-powered decision-support and simulation system for employee shift scheduling. Users describe scheduling problems in natural language, and the system:

1. Parses the problem into a structured representation (LLM)
2. Validates the representation (Pydantic)
3. Solves the optimization problem (OR-Tools CP-SAT)
4. Presents the solution with metrics
5. Supports what-if scenario comparison

## Core Principle

- **LLM** = interpretation, translation, explanation
- **Solver** = actual decision / optimization
- **Python** = orchestration, validation, simulation

The LLM is never the source of truth for constraint satisfaction.

## Current Status

**Phase 1 — Foundation** (implemented):
- Pydantic models for scenarios and solutions
- OR-Tools CP-SAT scheduler with hard/soft constraints
- Input validation
- Basic test suite

**Planned**:
- Phase 2: Constraint validation + metrics + more tests
- Phase 3: Natural-language parsing via LLM
- Phase 4: End-to-end pipeline
- Phase 5: What-if simulation
- Phase 6: Explanation + visualization + UI
- Phase 7: Evaluation benchmark

## Tech Stack

- Python 3.10+
- Pydantic 2.x
- Google OR-Tools CP-SAT
- Pandas
- Streamlit (UI)
- Plotly (visualization)
- Pytest (testing)

## Quick Start

```bash
pip install -r requirements.txt
pytest
```

## Usage (Phase 1 — Python API)

```python
from shift_happens.models.scenario import *
from shift_happens.solver.scheduler import solve

scenario = Scenario(
    num_days=3,
    employees=[Employee(id="alice", name="Alice"), Employee(id="bob", name="Bob")],
    shift_types=[ShiftType(name="morning"), ShiftType(name="night")],
    shift_requirements=[
        ShiftRequirement(day=d, shift_type=s, min_employees=1)
        for d in range(3) for s in ["morning", "night"]
    ],
)

solution = solve(scenario)
print(solution.status)       # optimal
print(solution.assignments)  # list of Assignment objects
print(solution.metrics)      # ScheduleMetrics
```

## License

Academic project — Final Year CSE.
