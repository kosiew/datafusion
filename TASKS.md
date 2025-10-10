# Tasks: Fix MinMaxBytesAccumulator Regression

## Summary of Observed Failure
- Fuzzer `aggregate_fuzz::test_min` reports wrong results when computing `MIN(binaryview)` grouped by `dictionary_utf8_low, utf8_low` (e.g. expected `0x0309`, observed `0x055ded`).
- Baseline context (no optimizations) and optimized context diverge, indicating MinMaxBytesAccumulator fails to update minima across batches for byte-oriented data types.

## Investigation Notes
- The mismatch occurs in `MinMaxBytesAccumulator` introduced by commit range `1eb9d9ac6^..4f3c320f3`.
- The incorrect value matches an earlier batch, suggesting new smaller values arriving in later batches are not applied.
- The failure is isolated to byte-oriented aggregates (`BinaryView` / `Binary` / `Utf8`) while numeric aggregates for the same group remain correct.

## Tasks
1. **Reproduce with Minimal Test Case**
   - Build a deterministic unit test for `MinMaxBytesState` covering multiple batches with the same group id where the second batch contains the new minimum (for `Binary`, `BinaryView`, and `Utf8` arrays).
   - Confirm which workload mode (`DenseInline`, `Simple`, `SparseOptimized`) is active when the failure occurs.

2. **Audit Dense and Sparse Update Paths**
   - Inspect `update_batch_dense_inline_impl` and `update_batch_sparse_impl` to ensure `set_value` executes whenever `cmp(new_val, existing_val)` is true, especially when batches reuse scratch structures.
   - Verify mark bookkeeping (`dense_inline_marks`, `scratch_sparse`, `scratch_dense`) resets correctly between batches so later minima are not skipped.

3. **Implement Fix**
   - Apply targeted fixes in the offending path (likely DenseInline mark handling or Sparse scratch reuse) so later batches correctly update the stored minima.
   - Ensure memory accounting (`total_data_bytes`, `populated_groups`) stays consistent after the fix.

4. **Add Regression Coverage**
   - Extend the unit tests added in step 1 to cover all byte array variants and multiple workload modes.
   - Run `cargo test -p datafusion --test fuzz aggregate_fuzz::test_min` to validate the fix against the original failure.

5. **Document Changes**
   - Update release notes or relevant documentation describing the regression and the fix to inform downstream users.
