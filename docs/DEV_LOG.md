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
