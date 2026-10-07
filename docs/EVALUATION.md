# Evaluation Plan

## Objective

Demonstrate measurable system behavior through a benchmark of scheduling scenarios.

## Metrics

| Metric | Description |
|--------|-------------|
| Parsing validity | % of NL inputs correctly parsed to valid Scenario |
| Scenario validity | % of parsed Scenarios passing validation |
| Feasible-solution rate | % of valid Scenarios yielding feasible solutions |
| Hard-constraint satisfaction | % of hard constraints satisfied in solution |
| Solver runtime | Wall-clock time for solver (seconds) |
| Objective value | Soft-penalty objective from solver |
| Scenario comparison correctness | Accuracy of what-if diff metrics |

## Benchmark Design

To be created in Phase 7:
- 20+ NL scheduling scenarios of varying complexity
- Known-feasible and known-infeasible cases
- Edge cases (single employee, many constraints, etc.)
- Comparison across prompting strategies / model sizes

## Status

Not yet implemented. Benchmark creation planned for Phase 7.
