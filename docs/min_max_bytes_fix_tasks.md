# Tasks to Resolve `MinMaxBytesState` Performance Regression

## Root Cause Summary

`MinMaxBytesState::update_batch` now performs a "prepass" before processing the
batch so it can decide whether to enable the dense scratch table immediately.
That prepass walks every `(group_index, value)` pair, inserting each touched
group into `scratch_sparse` just long enough to count the unique groups and the
maximum group id.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L517-L552】

Dense workloads trigger this prepass on their very first batch because
`scratch_dense_limit` starts at zero. The code then clears the map and walks
the batch a second time to perform the actual comparisons, meaning dense
workloads pay for two full scans plus a `HashMap` insertion per row before any
real work happens.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L517-L684】 The extra hashing and allocation
overhead dwarfs the savings from enabling the dense scratch table slightly
earlier, which is why the dense Criterion benchmarks regressed by 50–110%.

## Remediation Tasks

1. **Eliminate the prepass `HashMap` churn.**
   * Rework the density heuristic so it does not require inserting every group
     into `scratch_sparse` before the real loop. Track first-touch statistics
     using the existing `scratch_group_ids`/epoch machinery or another
     low-overhead structure that avoids hashing each row twice.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L517-L684】
   * Confirm that dense workloads no longer perform redundant
     `HashMap` operations while sparse workloads still avoid allocating large
     dense buffers.
2. **Benchmark both dense and sparse scenarios.**
   * Re-run the four Criterion benchmarks once the dense-path fix lands to
     validate that dense cases recover while sparse/monotonic cases stay
     improved.
   * Capture the before/after numbers in the PR description to prevent future
     regressions.
3. **Add coverage around the density heuristic.**
   * Extend the unit tests in `min_max_bytes.rs` so dense single-batch workloads
     assert that `scratch_sparse` is left empty (no prepass insertions) while
     sparse workloads still exercise the map path.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L517-L838】
