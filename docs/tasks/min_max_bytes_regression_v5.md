# Tasks to Fix `MinMaxBytesState` Sequential Dense Regression

## Root Cause

The sequential-dense fast path introduced in commit 352e69847 allocates a fresh
`locations` vector sized to `total_num_groups` on **every** batch via
`vec![SequentialDenseLocation::ExistingMinMax; total_num_groups]`. When the
accumulator processes high-cardinality workloads, `total_num_groups` tracks the
number of distinct groups observed so far, so later batches repeatedly allocate
and zero increasingly large buffers. This recreates the quadratic allocation
pattern the refactor set out to eliminate and explains the severe slowdowns seen
in benchmarks such as "dense reused accumulator" and "sequential dense large
stable".【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L721-L735】【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L975-L1036】

## Remediation Tasks

1. **Reuse scratch storage for the sequential dense path.**
   * Replace the per-call `vec![..; total_num_groups]` with storage owned by
     `MinMaxBytesState` (e.g. reuse `scratch_dense`/`scratch_group_ids` or add a
     dedicated buffer) so the fast path only grows when the domain expands and
     avoids repeated allocation/zeroing across batches.
   * Ensure the buffer tracks which entries were touched so we only update
     `self.min_max` for groups that supplied a winning value.

2. **Guard against regression with targeted benchmarks/tests.**
   * Extend the Criterion suite to assert that sequential dense workloads with
     large `total_num_groups` no longer trigger per-batch allocations.
   * Add a unit test that exercises multiple sequential batches while verifying
     the reused buffer’s capacity and the absence of quadratic growth in
     `size()`.

3. **Validate heuristic interactions.**
   * After reusing the buffer, re-run the full `min_max_bytes` benchmark matrix
     to confirm dense reuse is restored without hurting sparse modes.
   * Inspect `record_batch_stats` telemetry to ensure `unique_groups`/`max_group_index`
     remain correct so adaptive mode switches continue to work as intended.
