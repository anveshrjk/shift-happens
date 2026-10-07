# AGENTS.md

This file provides context for AI coding assistants working on this project.

## Project

Shift Happens — AI-Powered Decision & Simulation Engine for employee shift scheduling.

## Architecture

Modular Python project:

- `shift_happens/models/` — Pydantic data models (Scenario, Solution)
- `shift_happens/solver/` — OR-Tools CP-SAT scheduler
- `shift_happens/validation/` — Semantic input validation
- `shift_happens/llm/` — LLM provider abstraction (stub)
- `shift_happens/parser/` — NL→structured parser (stub)
- `shift_happens/simulation/` — What-if engine (stub)
- `shift_happens/explanation/` — AI explanation (stub)
- `shift_happens/ui/` — Streamlit UI (stub)
- `tests/` — Pytest test suite

## Key Rules

1. The LLM must NEVER be the source of truth for optimization decisions.
2. All LLM output is treated as untrusted input and validated via Pydantic.
3. The solver (OR-Tools CP-SAT) makes all scheduling decisions.
4. Never fabricate solutions — report infeasibility honestly.
5. Keep the codebase small and understandable.
6. Do not add technologies without demonstrated need.

## Testing

```bash
pytest                    # run all tests
pytest tests/test_solver.py  # solver tests only
```

## Development

See `docs/ROADMAP.md` for phase plan and `docs/DECISIONS.md` for design rationale.
