# Roadmap

## Phase 1 — Foundation ✅
- Project structure
- Pydantic models (Scenario, Solution)
- OR-Tools CP-SAT solver
- Input validation
- Basic test suite

## Phase 2 — Constraint Validation + Metrics
- Extended validation rules
- Richer metrics computation
- More comprehensive tests
- Edge case handling

## Phase 3 — Natural-Language Parsing ✅
- LLM provider abstraction (`LLMProvider`, `GoogleGeminiProvider`, `MockLLMProvider`)
- NL → structured scenario pipeline (`parse_natural_language()`)
- Schema-embedded system prompt with Pydantic JSON schema
- Structured output validation with explicit error taxonomy
- 52 parser/provider tests (all deterministic, no API key required)

## Phase 4 — End-to-End Pipeline (partial)
- NL → parser → validator → solver → solution ✅ (tested in `test_parser.py`)
- Error handling across the pipeline ✅
- Integration tests ✅

## Phase 5 — What-If Simulation
- Scenario modification API
- Re-solve and compare
- Diff metrics between scenarios

## Phase 6 — Explanation + Visualization + UI
- LLM-based solution explanation
- Plotly schedule visualizations
- Streamlit UI

## Phase 7 — Evaluation Benchmark
- Benchmark scenario suite
- Parsing accuracy measurement
- Solver performance metrics
- Multi-configuration comparison
