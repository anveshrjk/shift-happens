# Evaluation Plan

## Objective

Demonstrate measurable system correctness through a reproducible benchmark of scheduling scenarios.

## Implemented Benchmark

The evaluation benchmark consists of 35 fixed scenarios across 12 categories:

| Category | Cases | Description |
|----------|-------|-------------|
| Trivial | 2 | Minimum viable scheduling problems |
| Standard | 5 | Realistic small-to-medium problems |
| Workload | 4 | Problems with workload limits (max shifts, days off, consecutive) |
| Availability | 3 | Employee unavailability constraints |
| Restrictions | 2 | Shift type restrictions |
| Max cap | 1 | Shift staffing caps (min and max employees) |
| Edge | 4 | Edge cases (single day, many employees, no requirements, exact staffing) |
| Invalid | 1 | Validation-rejected scenario (overstaff) |
| Infeasible | 3 | Solver-determined infeasible scenarios |
| Large | 2 | Performance testing (15 emp × 3 shifts × 14 days) |
| Simulation | 6 | What-if simulation sub-cases |
| NL | 2 | NL parsing test cases (evaluated only with `--live-nl`) |

## Metrics

| Metric | Description |
|--------|-------------|
| Validation correctness | Whether semantic validation correctly accepts/rejects each scenario |
| Solver outcome accuracy | Whether solver produces expected result (feasible or infeasible) |
| Independent verification | Whether feasible solutions satisfy all constraints |
| Simulation correctness | Whether what-if changes produce expected outcomes |
| Original preservation | Whether simulations leave original data unmodified |
| Solver runtime | Wall-clock time per case (mean, p50, p95, max) |
| Coverage | Percentage of required shift slots filled |
| Utilization | Average employee utilization |

## Running the Benchmark

```bash
# Default offline evaluation:
python -m shift_happens.benchmark

# With live NL evaluation:
python -m shift_happens.benchmark --live-nl

# Filter to specific cases:
python -m shift_happens.benchmark --filter sim
```

Reports are generated in `benchmark_results/` as JSON, CSV, and text summary.

## Design Principles

- **Fixed dataset** — Scenarios are deterministic and versioned. No random generation.
- **Offline by default** — No API key or network access required for the default benchmark.
- **Separation** — Validation, solver, and verification metrics are reported independently.
- **Independent verification** — A separate verifier checks constraints, independent of the solver's internal logic.
- **Honest reporting** — Validation-rejected cases are not counted as solver successes.
