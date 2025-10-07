# Tasks to Resolve `MinMaxBytesState` Performance Regression

## Root Cause Summary

`MinMaxBytesState::update_batch` now keeps the previous batch's dense scratch
limit and only grows it when the _current_ batch appears dense according to the
`evaluate_dense_candidate` heuristic.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L815-L834】 When a new
range of group identifiers arrives (for example the next 512 monotonic IDs), the
first rows fall outside `self.scratch_dense_limit`, so they take the slow sparse
path that performs `HashMap` lookups and inserts for every row.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L568-L677】 Only after the batch accumulates
enough unique groups does the heuristic permit a dense expansion, at which point
we migrate all of the already-processed groups from the sparse map into the
dense table within the same batch.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L861-L875】 Dense benchmarks therefore spend a sizeable
prefix of each batch in the sparse path and immediately pay an additional
migration cost, which explains the 30–70% regressions even though the incoming
data is perfectly dense.【F:datafusion/functions-aggregate/benches/min_max_bytes.rs†L33-L102】

## Remediation Tasks

1. **Short-circuit contiguous dense growth.**
   * Teach the dense branch to expand `self.scratch_dense_limit` immediately
     when it encounters the next contiguous group identifier instead of routing
     through the sparse map first.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L568-L705】 This keeps monotonic and large dense workloads on the
     fast path and avoids the mid-batch migration work.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L861-L875】
2. **Preserve density knowledge across batches.**
   * Rework `evaluate_dense_candidate` (or add a complementary guard) so that a
     batch extending a previously dense range inherits that context instead of
     starting from the sparse path each time.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L800-L834】 This may involve tracking the highest
     dense index seen so far or the expected next dense group id.
   * Ensure the revised heuristic still avoids allocating `scratch_dense`
     proportional to `total_num_groups` when the workload is genuinely sparse.
3. **Add regression coverage for dense detours.**
   * Extend the unit tests to assert that dense scenarios never increment
     `dense_sparse_detours`, catching any future fallback to the sparse path.
     【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L512-L720】【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L1040-L1080】
   * Update the Criterion benchmarks (or add new ones) to cover monotonic
     multi-batch workloads so we can verify dense throughput before merging
     changes.【F:datafusion/functions-aggregate/benches/min_max_bytes.rs†L33-L102】
