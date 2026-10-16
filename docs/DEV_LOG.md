# Development Log

## Phase 1 — Foundation
**Date**: 2026-10-08

### Completed
- Initialized Git repository
- Created project structure with modular package layout
- Implemented Pydantic models:
  - `Scenario` with cross-field validation (referential integrity)
  - `ScheduleSolution` with status, assignments, metrics
- Implemented OR-Tools CP-SAT solver (`scheduler.py`):
  - Boolean assignment variables
  - Hard constraints: one-shift-per-day, staffing requirements, unavailability, shift restrictions
  - Soft constraints: penalties in objective
  - Workload limits: max/min shifts, consecutive days, days off
  - Workload balancing objective
  - Infeasibility diagnosis
- Implemented semantic validator (`validator.py`)
- Created stub modules for future phases (LLM, parser, simulation, explanation, UI)
- Created test suite
- Created project documentation

### Decisions Made
- See `docs/DECISIONS.md`

### Known Limitations
- No NL parsing yet (Phase 3)
- No UI (Phase 6)
- Infeasibility diagnosis is best-effort heuristic
- Soft penalty tracking in metrics is approximate

## Phase 3 — Natural-Language Parsing
**Date**: 2026-10-16

### Completed
- Implemented LLM provider abstraction (`shift_happens/llm/provider.py`):
  - `LLMProvider` abstract interface with `parse_scenario()` and `explain_solution()`
  - `GoogleGeminiProvider` using the `google-genai` package
  - `MockLLMProvider` for deterministic, offline testing
  - `ProviderError` exception for API/configuration failures
  - System prompt embeds auto-generated Pydantic JSON schema
- Implemented NL parsing pipeline (`shift_happens/parser/nl_parser.py`):
  - `parse_natural_language(text, provider)` → `ParseResult`
  - Explicit error taxonomy: `EMPTY_INPUT`, `PROVIDER_ERROR`, `INVALID_JSON`, `NOT_SCHEDULING`, `SCHEMA_ERROR`, `SEMANTIC_ERROR`
  - Three-stage validation: LLM output → Pydantic model → semantic validator
  - `ParseResult.success` property for clean success/failure checking
- Added 52 new tests (`tests/test_parser.py`):
  - Valid parsing and field preservation
  - Solver integration (parsed → solve → feasible solution)
  - Empty/whitespace input rejection
  - Provider/API error handling
  - Invalid/malformed LLM output
  - Missing fields, invalid types, referential integrity
  - Semantic validation errors (distinct from schema errors)
  - Non-scheduling request detection
  - Edge cases (minimal scenario, large scenario, soft constraints, extra fields)
  - Provider abstraction (mock behavior, missing API key, missing package)
  - ParseResult dataclass behavior
- Updated dependency manifests (`pyproject.toml`, `requirements.txt`)
- Updated documentation (`ARCHITECTURE.md`, `DECISIONS.md`, `ROADMAP.md`, `README.md`)

### Decisions Made
- D7: Google Gemini as default LLM provider
- D8: Schema-embedded system prompt
- D9: Explicit error taxonomy in ParseResult
- D10: google-genai as optional dependency

### Test Results
- 96 total tests: 44 Phase 1 + 52 Phase 3
- All 96 passing
- No tests require API key or network access

### Known Limitations
- `explain_solution()` not yet implemented (Phase 6)
- Live LLM parsing not verified in automated tests (requires API key)
- Prompt engineering may need tuning for edge-case NL inputs
- No UI (Phase 6)
