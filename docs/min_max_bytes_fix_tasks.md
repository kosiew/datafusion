# Tasks to Resolve `MinMaxBytesState` Performance Regression

## Root Cause Summary

`MinMaxBytesState::update_batch` defers enabling the dense scratch table until
*after* a batch completes. The dense path is only activated when
`scratch_dense_limit` is non-zero, but that limit is derived at the end of the
function from the groups touched in the current batch.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L500-L575】
Consequently the first batch processed by a fresh accumulator is always routed
through the sparse `HashMap` scratch, even if the groups are perfectly dense.

Benchmarks such as `min bytes dense groups` and `min bytes large dense groups`
instantiate a brand-new accumulator and feed it a single dense batch, so the
workload never benefits from the dense scratch path. Each run now builds and
probes a `HashMap` entry per row, introducing the 100%+ slowdowns reported by
Criterion while offering no reuse to amortise that cost.

## Remediation Tasks

1. **Enable the dense scratch path during the initial batch when appropriate.**
   * Derive the density heuristics (e.g. `max_group_index` vs. `unique_groups`)
     *before* iterating over the values so that the first batch can opt into
     dense scratch storage immediately.
   * Alternatively, stage the dense scratch initialisation lazily during the
     batch so that once the heuristics trigger, subsequent rows in the same
     batch switch away from the `HashMap` path.
   * Keep the sparse fallback for workloads with scattered group ids.
2. **Benchmark both dense and sparse scenarios.**
   * Re-run the four Criterion benchmarks once the dense-path fix lands to
     validate that dense cases recover while sparse/monotonic cases stay
     improved.
   * Capture the before/after numbers in the PR description to prevent future
     regressions.
3. **Add coverage around first-batch behaviour.**
   * Extend the unit tests in `min_max_bytes.rs` to assert that a single dense
     batch enables the dense scratch table and avoids populating the sparse
     map, while sparse workloads keep using the map.
