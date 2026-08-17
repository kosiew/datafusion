source: pr-23628_a
# Add an engine-level benchmark for grouped nested `first_value` / `last_value`

## Problem
DataFusion has nested-value support in `FirstLastGroupsAccumulator`, and `datafusion/functions-aggregate/benches/first_last.rs` benchmarks its List, Struct, Map, and `List<Struct>` accumulator paths directly.

`datafusion/core/benches/aggregate_query_sql.rs` exercises SQL planning and execution for grouped `first_value` / `last_value`, but only with scalar payloads. There is no reproducible engine-level Criterion workload for grouped ordered first/last aggregates over wide nested values across multiple input batches.

Existing nested SQLLogicTests and the `ScalarValue::compact()` regression test cover correctness and source-buffer compaction. This issue adds performance coverage; it does not claim an existing retention bug.

## Why it matters
The accumulator benchmark cannot measure SQL planning, physical planning, grouping, and execution together. An engine-level workload makes regressions in that path measurable while retaining the focused accumulator benchmark for lower-level diagnosis.

## Invariant / desired behavior
A reproducible engine-level benchmark executes the normal SQL planning and execution path for grouped `first_value(value ORDER BY key)` and `last_value(value ORDER BY key)` over wide nested payloads spanning multiple batches.

The workload must make its nested shape, batch count, group cardinality, ordering, and seed explicit so results are comparable across revisions and machines.

## Proposed direction
Add a focused Criterion benchmark under `datafusion/core/benches/`, extending `aggregate_query_sql.rs` only if its fixture remains clear.

Build a deterministic in-memory table with a nested value column (preferably `List<Struct<...>>`), ordering key, and group key. Run grouped ordered `first_value` and `last_value` queries over multiple batches. Include at least two group-cardinality cases so aggregation state behavior is visible under different live-group counts.

Use Criterion for elapsed time. If memory is measured, document the tool, command, and measurement scope. Do not present DataFusion memory-pool reservations as process peak RSS unless that equivalence is established.

## Scope
### In
- Add a reproducible SQL/planning/execution benchmark for grouped nested `first_value` and `last_value`.
- Use wide nested payloads across multiple input batches.
- Cover at least two group cardinalities.
- Document workload parameters and benchmark commands.
- Optionally document a separately collected memory measurement with explicit scope.

### Out
- Changing first/last aggregate semantics or implementation.
- Fixing source-buffer retention; `GenericValueState` already compacts stored nested winners.
- Replacing or duplicating `datafusion/functions-aggregate/benches/first_last.rs`.
- Broad benchmarking of every nested Arrow type.
- Adding CI performance gates.

## Acceptance criteria
- [ ] A release Criterion benchmark executes SQL-level grouped `first_value` and `last_value` over a nested type supported by `FirstLastGroupsAccumulator`.
- [ ] Input has multiple batches and candidate rows per group.
- [ ] Benchmark includes at least two group-cardinality cases.
- [ ] Benchmark comments document nested shape, batch count, group cardinality, ordering, and deterministic data generation.
- [ ] Criterion measures elapsed time for the normal SQL planning and execution path.
- [ ] Any memory-measurement instructions name the tool and state exactly what memory they measure.
- [ ] Existing nested first/last SQLLogicTests pass before interpreting benchmark results.

## Tests / verification
- Run the nested first/last SQLLogicTests for correctness.
- Run the release benchmark, for example:
  `cargo bench -p datafusion --bench aggregate_query_sql -- first_last_nested`
- For historical comparisons, run the unchanged benchmark with the same machine, allocator, parameters, and seed on both revisions.

## Notes / open questions
- Decide whether the benchmark belongs in `aggregate_query_sql.rs` or a dedicated core benchmark file once the nested fixture is implemented.
- If process-level peak memory is useful, choose a platform-appropriate external measurement tool and document its limitations.
