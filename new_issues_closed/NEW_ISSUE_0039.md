stale
source: pr-22322_a
# Benchmark Lacks Row-Based Baseline for Multi-Column GROUP BY

## Summary
The benchmark in `datafusion/core/benches/multi_group_by.rs` only generates `Int32` grouping columns, which routes all multi-column cases to the vectorized `GroupValuesColumn` path. The benchmark's stated goal is to understand the crossover where row-based grouping (`GroupValuesRows`) can outperform the per-column approach, but the current setup does not measure row-based behavior at all.

## Scope
- Component: DataFusion benchmarks
- File: `datafusion/core/benches/multi_group_by.rs`
- Related execution path:
  - `datafusion/physical-plan/src/aggregates/group_values/mod.rs`
  - `datafusion/physical-plan/src/aggregates/group_values/multi_group_by/mod.rs`

## Current Behavior
- Benchmark data generation creates only `Int32` columns for group-by keys.
- For multi-column group-by, when all types are supported, DataFusion selects `GroupValuesColumn`.
- `Int32` is a supported type, so all benchmark cases use `GroupValuesColumn`.
- No measured results for `GroupValuesRows` are produced.

## Expected Behavior
Each benchmark scenario (cardinality x column count) should include comparable runs for both implementations:
1. A vectorized path (`GroupValuesColumn`) run.
2. A row-based path (`GroupValuesRows`) run.

This allows direct comparison and validates any claimed crossover threshold.

## Why This Matters
- The benchmark currently cannot substantiate claims about when row-based grouping is better.
- Results may be interpreted as implementation-neutral while actually measuring only one implementation.
- Follow-up optimization decisions risk being guided by incomplete evidence.

## Reproduction / Validation Notes
1. Inspect benchmark input schema in `datafusion/core/benches/multi_group_by.rs`.
2. Inspect `new_group_values` selection logic in `datafusion/physical-plan/src/aggregates/group_values/mod.rs`.
3. Confirm `Int32` support in `datafusion/physical-plan/src/aggregates/group_values/multi_group_by/mod.rs`.
4. Run benchmark and verify only columnar path is exercised for multi-column cases.

## Proposed Fix Options
1. Add paired benchmark variants per case:
   - Existing schema (supported types) for `GroupValuesColumn`.
   - Schema including at least one unsupported grouping type to force `GroupValuesRows`.
2. Alternatively, benchmark `GroupValues` implementations directly in a lower-level bench.
3. Ensure benchmark naming explicitly indicates implementation (`rows` vs `column`).

## Acceptance Criteria
1. Every benchmark case has explicit `rows` and `column` variants, or equivalent direct implementation benches.
2. Output names clearly identify implementation under test.
3. Documentation/comments in the benchmark match what is measured.
4. Benchmark compiles and runs with existing benchmark command.

## Suggested Labels
- `benchmark`
- `performance`
- `good first issue` (optional)
