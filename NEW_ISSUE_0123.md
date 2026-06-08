source: pr-23628_a
# Add an engine-level benchmark for grouped nested `first_value` / `last_value`

## Problem
Nested value types already use `FirstLastGroupsAccumulator` with `GenericValueState`, and `datafusion-functions-aggregate/benches/first_last.rs` already benchmarks List, Struct, Map, and `List<Struct>` update, merge, and evaluation paths.

That benchmark is intentionally accumulator-level: it uses arrays directly rather than SQL planning/execution, does not model many independent input batches, and does not report process-level peak memory. `datafusion/core/benches/aggregate_query_sql.rs` exercises the SQL path for scalar `first_value` / `last_value` only.

Consequently, there is no end-to-end performance workload for grouped `first_value(value ORDER BY key)` / `last_value(value ORDER BY key)` over wide nested values that can expose retained source-batch buffers or regressions in the planner/executor path.

## Why it matters
Wide nested payloads with many candidate rows per group are the workload where compact per-group winners matter most. The existing microbenchmark protects accumulator behavior, but cannot quantify end-to-end query cost or memory retention across input batches.

## Invariant / desired behavior
An engine-level benchmark must execute the normal SQL planning and execution path over multiple batches of wide nested values. Its workload must retain enough candidate rows per group to distinguish state proportional to live groups and winning payloads from state that retains every source batch containing a candidate.

## Proposed direction
Add a focused Criterion benchmark under `datafusion/core/benches/`, extending `aggregate_query_sql.rs` only if its fixture can express the workload clearly.

Build a deterministic multi-batch in-memory table with a wide nested value column, preferably `List<Struct<...>>`, an ordering key, and configurable group cardinality. Benchmark grouped `first_value` and `last_value` SQL queries. Keep the existing `functions-aggregate` microbenchmark as complementary coverage; do not duplicate it.

Use Criterion for elapsed-time comparisons. Capture peak memory with a documented measurement whose scope is explicit. Do not present `MemoryPool` reservations as process memory unless they are proven to include retained source buffers.

## Scope
### In
- Add a reproducible SQL/planning/execution benchmark for grouped nested `first_value` and `last_value`.
- Use wide nested payloads spanning multiple input batches and many candidate rows per group.
- Exercise group cardinalities that materially change retention behavior.
- Document workload parameters, comparison commands, and peak-memory measurement scope.

### Out
- Changing first/last aggregate semantics or implementation.
- Replacing the existing nested accumulator microbenchmark.
- Broad benchmarking of every nested Arrow type.
- Adding CI performance gates.

## Acceptance criteria
- [ ] A release Criterion benchmark executes SQL-level grouped `first_value` and `last_value` over a nested type supported by `FirstLastGroupsAccumulator`.
- [ ] Input uses multiple batches and enough rows per group to exercise winner retention across batch boundaries.
- [ ] Benchmark comments document nested shape, batch count, group cardinality, and why each matters.
- [ ] Elapsed time is measured with Criterion.
- [ ] Peak-memory measurement and its scope are documented; it does not overclaim process-level memory from pool-only metrics.
- [ ] Results are verified for correctness before performance comparisons.
- [ ] Historical comparisons apply the identical benchmark to both `e2d80a9f3e^` and `e2d80a9f3e` (for example, by cherry-picking the benchmark commit), using the same machine, allocator, parameters, and seed.

## Tests / verification
- Run the new benchmark in release mode on the target revision.
- For historical comparison, apply the benchmark unchanged to both revisions, then compare Criterion baselines and peak-memory measurements.
- Run the relevant nested first/last SQLLogicTests to confirm query results before interpreting measurements.

## Notes / open questions
- Determine whether a DataFusion memory-pool metric captures source-buffer retention. If not, use and document process-level peak RSS or an allocation profiler.
