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

## D7: Google Gemini as default LLM provider
**Decision**: Use the Google Gemini API (`google-genai` package) with `gemini-2.0-flash` as the default model.
**Rationale**: Gemini provides structured output capabilities, supports system instructions, and has a generous free tier. The provider abstraction allows swapping to any other LLM (OpenAI, Anthropic, local models) by implementing the `LLMProvider` interface.

## D8: Schema-embedded system prompt
**Decision**: Embed the Pydantic `Scenario.model_json_schema()` directly in the LLM system prompt.
**Rationale**: This ensures the LLM always sees the exact current schema without manual synchronization. The schema is auto-generated from the models, so changes to the Pydantic models automatically propagate to the prompt.

## D9: Explicit error taxonomy in ParseResult
**Decision**: Distinguish five error categories: `EMPTY_INPUT`, `PROVIDER_ERROR`, `INVALID_JSON`/`NOT_SCHEDULING`, `SCHEMA_ERROR`, and `SEMANTIC_ERROR`.
**Rationale**: Each category represents a different pipeline stage failing. Schema errors (Pydantic rejected the dict) and semantic errors (valid structure, infeasible logic) require different user-facing messages and recovery actions. Collapsing them obscures the root cause.

## D10: google-genai as optional dependency
**Decision**: `google-genai` is listed as an optional dependency (`[llm]` extras group), not a core requirement.
**Rationale**: The test suite runs entirely with `MockLLMProvider` — no API key or network access needed. The solver, validator, and models work without any LLM package installed. Only live NL parsing requires `google-genai`.
