# Tasks to Resolve `MinMaxBytesState` Performance Regression

## Root Cause Summary

The new dense/sparse scratch design eagerly grows `scratch_dense` to
`total_num_groups` on every `update_batch` and fills each slot with a
fresh `ScratchEntry` before the batch is processed.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L487-L547】
This reintroduces an `O(total_num_groups)` initialization cost that the
previous hash-map based scratch avoided for short-lived accumulators.
Workloads such as `min bytes dense groups` and `min bytes large dense groups`
instantiate a brand new accumulator for a single batch, so they repeatedly pay
for allocating and zeroing tens of thousands of scratch entries without ever
reusing them, which explains the double-digit slowdowns reported by Criterion.

## Remediation Tasks

1. **Defer or eliminate eager dense scratch initialization.**
   * Grow `scratch_dense` lazily (e.g. per touched chunk) or gate its maximum
     size so that single-batch dense workloads do not have to clear
     `O(total_num_groups)` entries up front.
   * Ensure sparsely indexed groups continue to use the hash map fallback so we
     keep the wins for monotonic and sparse inputs.
2. **Benchmark both dense and sparse scenarios.**
   * Re-run the four Criterion benchmarks once the dense-path fix lands to
     validate that dense cases recover while sparse/monotonic cases stay
     improved.
   * Capture the before/after numbers in the PR description to prevent future
     regressions.
3. **Add coverage around scratch reuse.**
   * Extend the unit tests in `min_max_bytes.rs` to exercise back-to-back dense
     and sparse batches, asserting that scratch allocations stay bounded and
     `total_data_bytes` accounting remains correct across updates.
