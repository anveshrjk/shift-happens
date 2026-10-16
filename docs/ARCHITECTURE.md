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

### llm/
`provider.py` — LLM provider abstraction layer:
- `LLMProvider` — Abstract interface for obtaining structured scheduling data
- `GoogleGeminiProvider` — Concrete implementation using Google Gemini API
- `MockLLMProvider` — Deterministic provider for testing (no network calls)
- `ProviderError` — Exception for API/configuration failures
- Embeds the Pydantic `Scenario` JSON schema in the system prompt
- API key via `GEMINI_API_KEY` environment variable (never hardcoded)

### parser/
`nl_parser.py` — Natural language → validated `Scenario` pipeline:
- `parse_natural_language(text, provider)` — Main entry point
- Returns `ParseResult` with explicit error categories:
  - `EMPTY_INPUT` — No text provided
  - `PROVIDER_ERROR` — LLM API failure
  - `INVALID_JSON` — LLM returned non-dict output
  - `NOT_SCHEDULING` — LLM identified a non-scheduling request
  - `SCHEMA_ERROR` — Output doesn't match Pydantic Scenario model
  - `SEMANTIC_ERROR` — Valid structure but logically infeasible
- Successful results include a solver-ready `Scenario` and any warnings

### simulation/ (stub)
What-if scenario modification and comparison.

### explanation/ (stub)
LLM-based solution explanation from structured data.

### ui/ (stub)
Streamlit web interface.

## Data Flow

### NL Parsing Pipeline
```
Natural-language text
  → LLMProvider.parse_scenario()
  → raw dict (untrusted)
  → Scenario.model_validate() (Pydantic)
  → validate_scenario() (semantic checks)
  → ParseResult (with Scenario or errors)
```

### Solving Pipeline
```
ParseResult.scenario
  → solve() (OR-Tools CP-SAT)
  → ScheduleSolution (assignments, metrics, status)
```

## Constraint Model

- **Hard constraints**: Must be satisfied. Solver reports infeasible if impossible.
- **Soft constraints**: Penalized in objective. Solver minimizes total penalty.

The solver never invents a schedule. If infeasible, it reports why.

## Trust Boundaries

The LLM's output is always treated as **untrusted input**:
1. The LLM produces a raw dict — never Python code or solver instructions
2. Pydantic validates the dict against the Scenario schema
3. Semantic validation checks logical feasibility
4. Only a fully validated Scenario reaches the deterministic solver
