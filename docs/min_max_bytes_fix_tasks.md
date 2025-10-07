# Tasks to Resolve `MinMaxBytesState` Performance Regression

## Root Cause Summary

`MinMaxBytesState::update_batch` defers enabling the dense scratch table until
*after* a batch completes. The dense path is gated on
`scratch_dense_limit > 0`, but that limit is derived only after the input loop
finishes via the post-batch density heuristic.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L470-L632】
During the first batch `scratch_dense_enabled` is `false` and
`total_data_bytes` is still `0`, so every row is forced through the sparse
`HashMap` path even when the groups are perfectly dense.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L480-L574】

Benchmarks such as `min bytes dense groups` and `min bytes large dense groups`
instantiate a fresh accumulator and feed it a single dense batch. Because the
dense scratch table is only initialised *after* the loop, those benchmarks
never take the O(1) dense path. Each run now inserts and probes a `HashMap`
entry per row, explaining the 100%+ slowdowns reported by Criterion while
offering no reuse to amortise that cost.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L480-L574】

## Remediation Tasks

1. **Enable the dense scratch path during the initial batch when appropriate.**
   * Either pre-compute the density heuristic (e.g. number of unique groups vs.
     max group id) before iterating so the dense scratch table can be
     initialised up-front, or allow the loop to promote groups from the sparse
     map into the dense table as soon as the heuristic threshold is crossed.
   * Ensure the dense path continues to respect the existing `scratch_dense_limit`
     growth rules and still falls back to the sparse `HashMap` for truly sparse
     workloads.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L470-L632】
2. **Benchmark both dense and sparse scenarios.**
   * Re-run the four Criterion benchmarks once the dense-path fix lands to
     validate that dense cases recover while sparse/monotonic cases stay
     improved.
   * Capture the before/after numbers in the PR description to prevent future
     regressions.
3. **Add coverage around first-batch behaviour.**
   * Extend the unit tests in `min_max_bytes.rs` to assert that a single dense
     batch enables the dense scratch table and avoids populating the sparse
     map, while sparse workloads keep using the map.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L470-L632】
