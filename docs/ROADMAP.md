# Roadmap

## Phase 1 — Foundation ✅
- Project structure and modular package layout
- Pydantic models (Scenario, Solution) with cross-field validation
- OR-Tools CP-SAT solver with hard/soft constraints
- Semantic input validation
- Basic test suite (44 tests)

> Phase 2 (Constraint Validation + Metrics) was absorbed into Phase 1 during implementation. Extended validation, richer metrics, and edge-case handling were completed as part of the foundation work.

## Phase 3 — Natural-Language Parsing ✅
- LLM provider abstraction (`LLMProvider`, `GoogleGeminiProvider`, `MockLLMProvider`)
- NL → structured scenario pipeline (`parse_natural_language()`)
- Schema-embedded system prompt with auto-generated Pydantic JSON schema
- Structured output validation with explicit error taxonomy
- 52 parser/provider tests (all deterministic, no API key required)

## Phase 4 — End-to-End Pipeline ✅
- `schedule()` orchestrator: NL → parse → validate → solve → `PipelineResult`
- `PipelineStatus` taxonomy: SUCCESS, INFEASIBLE, PARSE_FAILED, SOLVER_ERROR, INTERNAL_ERROR
- CLI entry point: `python -m shift_happens.cli`
- 27 integration tests

## Phase 5 — What-If Simulation ✅
- `simulate()` engine with deep-copy scenario modification
- Change types: employee unavailability, staffing adjustments, workload limit changes
- Assignment diff and metrics diff computation
- Original scenario/solution preservation guarantees
- Status taxonomy: SUCCESS, INFEASIBLE, INVALID_CHANGE, VALIDATION_FAILED
- 34 simulation tests

## Phase 6 — Explanation + Visualization + UI ✅
- Grounded schedule explanations from solver data (no LLM)
- Streamlit web interface with Natural Language and Quick Demo modes
- Schedule tables, shift heatmaps, workload distribution charts
- What-if simulation interface with side-by-side comparison
- Coverage and utilization metrics display
- 4 UI tests

## Phase 7 — Evaluation Benchmark ✅
- 35 fixed benchmark scenarios across 12 categories
- Automated solver outcome evaluation
- Independent constraint verification
- Simulation correctness evaluation
- Optional live NL parsing evaluation
- JSON, CSV, and text summary reporting
- CLI with filtering, timeout, and output options
- 65 benchmark tests

## Future Work
- Prompt engineering improvements for edge-case NL inputs
- Multi-day constraint linking (e.g., rest-after-night-shift rules)
- Schedule persistence and session management
- Additional scheduling domains beyond employee shifts
- Comparative evaluation across different LLM providers
- Performance optimization for large-scale scenarios (50+ employees)
