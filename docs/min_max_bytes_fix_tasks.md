# Tasks to Resolve `MinMaxBytesState` Performance Regression

## Root Cause Summary

The squashed change replaces the per-batch scratch vectors with a single
`HashMap<usize, ScratchLocation>` to track which groups were touched and where
their candidate value lives.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L414-L507】
While this avoids the prior `Vec::resize` churn for sparse and monotonic group
ids, it now forces every row to execute a hash lookup (and later a removal)
when updating a group. Dense workloads—where batches cover almost every group
id—suddenly pay the hashing and pointer chasing cost on every iteration, which
explains the 130%+ regressions in the dense benchmarks.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L472-L507】

## Remediation Tasks

1. **Restore an O(1) dense-path for scratch lookups.**
   * Re-introduce a contiguous scratch table (e.g. `Vec<Entry>`) or another
     cache-friendly structure so dense batches can look up locations via direct
     indexing rather than hashing.
   * Maintain the sparse-friendly behaviour by combining the table with an epoch
     counter or side lookup structure so untouched slots do not need to be
     cleared between batches.
2. **Benchmark the hybrid design.**
   * Extend Criterion runs to cover all four benchmark scenarios to confirm we
     keep the sparse/monotonic wins without regressing dense inputs.
   * Capture baseline results before and after the fix for future regressions.
3. **Add focused tests around scratch bookkeeping.**
   * Exercise multiple dense and sparse batches in unit tests to verify the
     scratch table is reused correctly and no longer depends on per-row hash
     operations.
   * Assert that memory accounting (`total_data_bytes`) remains correct across
     updates.
