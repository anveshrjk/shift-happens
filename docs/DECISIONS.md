# Design Decisions

## D1: OR-Tools CP-SAT over MIP solvers
**Decision**: Use OR-Tools CP-SAT as the optimization engine.
**Rationale**: CP-SAT handles integer/boolean constraint satisfaction natively, is free, well-documented, and sufficient for shift scheduling. No need for commercial solvers.

## D2: Pydantic for data contracts
**Decision**: Use Pydantic v2 models as the single source of truth for data structure.
**Rationale**: Provides validation, serialization, and schema generation. LLM output can be validated against these models before reaching the solver.

## D3: LLM as interpreter, not decision-maker
**Decision**: The LLM translates NL→structured data and explains results. It never solves the optimization.
**Rationale**: LLMs are unreliable for constraint satisfaction. Deterministic solvers provide provably correct results.

## D4: Hard/soft constraint distinction
**Decision**: Constraints are tagged as HARD or SOFT at the model level.
**Rationale**: Users need to express preferences vs. requirements. The solver handles both natively via enforcement vs. objective penalties.

## D5: Employee shift scheduling as initial domain
**Decision**: Build for one domain first, design for extensibility.
**Rationale**: Focusing on one domain keeps the codebase small and testable. The architecture (Scenario→Solver→Solution) can accommodate new domains later.

## D6: No LangChain / vector DB / microservices
**Decision**: Minimal dependency stack.
**Rationale**: The problem doesn't require document retrieval, multi-agent coordination, or distributed systems. Adding them would increase complexity without benefit.
