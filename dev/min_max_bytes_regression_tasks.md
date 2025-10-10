# Min/max bytes regression tasks

## Root cause synopsis
`record_batch_stats` enables `dense_inline_marks_ready` as soon as a dense inline batch is
observed (see the assignment around lines 1250-1260 of
`min_max_bytes.rs`). On the very next batch, `update_batch_dense_inline_impl` notices the
flag and unconditionally calls `prepare_dense_inline_marks` at the top of the function
(around lines 900-910). That helper resizes and zeroes the `dense_inline_marks` vector to
`total_num_groups`, even if the sequential-dense fast path never needs any marks. Stable
sequential workloads such as the `min bytes sequential stable groups` benchmark therefore
incur a 512 KiB zero-fill on the second batch (65,536 groups × 8 bytes), despite all
subsequent iterations remaining perfectly sequential and eventually "committing" to the
fast path. The extra allocation shows up as the ~2-4% regressions in the criterion suite.

## Tasks
1. Defer the call to `prepare_dense_inline_marks` until we actually abandon the sequential
   fast path (i.e. once `fast_path` flips to `false`). This keeps the zeroing cost out of
   purely sequential runs while retaining the existing behaviour for workloads that really
   need the marks table.
2. Add a regression test that drives two or more sequential dense batches and asserts that
   `dense_inline_marks` remains empty (or at least that its epoch never advances). The
   current `sequential_dense_reuses_allocation_across_batches` test is a good starting
   point.
3. Re-run the `min bytes sequential stable groups` and `min bytes sequential dense large
   stable` benchmarks (or equivalent micro-benchmarks) to ensure the fix removes the
   regressions without affecting the sparse/high-cardinality improvements.
