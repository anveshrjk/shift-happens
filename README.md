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

**Phase 1 — Foundation** ✅:
- Pydantic models for scenarios and solutions
- OR-Tools CP-SAT scheduler with hard/soft constraints
- Input validation
- 44 tests

**Phase 3 — Natural-Language Parsing** ✅:
- LLM provider abstraction (Gemini, Mock)
- NL → Scenario parsing pipeline with error taxonomy
- 52 parser/provider tests

**Phase 4 — End-to-End Pipeline** ✅:
- `schedule()` orchestrator: NL → parse → validate → solve
- CLI: `python -m shift_happens.cli "..."`
- 27 integration tests
- 123 total tests passing

**Planned**:
- Phase 5: What-if simulation
- Phase 6: Explanation + visualization + UI
- Phase 7: Evaluation benchmark

## Tech Stack

- Python 3.10+
- Pydantic 2.x
- Google OR-Tools CP-SAT
- Google Gemini API (`google-genai`, optional)
- Pandas
- Streamlit (UI, planned)
- Plotly (visualization, planned)
- Pytest (testing)

## Quick Start

```bash
pip install -r requirements.txt
pytest
```

## Gemini API Setup (for live NL parsing)

```bash
pip install google-genai
```

Set the API key as an environment variable:
```bash
# Linux/macOS
export GEMINI_API_KEY="your-api-key-here"

# Windows PowerShell
$env:GEMINI_API_KEY = "your-api-key-here"
```

The test suite runs entirely without an API key using `MockLLMProvider`.

## Usage — End-to-End Pipeline (recommended)

```python
from shift_happens.llm.provider import GoogleGeminiProvider
from shift_happens.pipeline import PipelineStatus, schedule

provider = GoogleGeminiProvider()  # uses GEMINI_API_KEY env var

result = schedule(
    "Schedule 5 employees across morning and night shifts for 7 days. "
    "Each shift needs 2 people. Everyone gets at least 2 days off. "
    "Alice cannot work night shifts.",
    provider,
)

if result.status == PipelineStatus.SUCCESS:
    print(f"Schedule created: {result.solution.status.value}")
    print(f"Assignments: {result.solution.metrics.total_assignments}")
    for a in result.solution.assignments:
        print(f"  Day {a.day}: {a.employee_name} → {a.shift_type}")
elif result.status == PipelineStatus.INFEASIBLE:
    print("No feasible schedule found.")
    for e in result.errors:
        print(f"  {e}")
else:
    print(f"Failed ({result.status.value}):")
    for e in result.errors:
        print(f"  - {e}")
```

## Usage — CLI (requires API key)

```bash
python -m shift_happens.cli "Schedule 3 employees for 5 days with morning and night shifts."
python -m shift_happens.cli --file request.txt
python -m shift_happens.cli --timeout 30 "..."
```

## Usage — Direct Scenario (Python API)

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
