source: pr-22991_a
# Refactor: Centralize TopK heap-boundary encoding and checks

## Summary

`TopK` currently derives heap-boundary information in two places:

1. dynamic filter pushdown (`TopK::update_filter`)
2. prefix early completion (`TopK::attempt_early_completion`)

Both paths depend on the same local heap boundary: `self.heap.max()`, the current worst row still kept by the TopK heap. The implementation is correct, but the boundary-related work is spread across multiple methods and has duplicated comparison/control-flow.

Refactor the local heap-boundary handling into a small private helper so full sort-key threshold comparison, scalar threshold extraction, and common-prefix comparison are easier to reason about and harder to accidentally diverge.

## Current state

Relevant file:

- `datafusion/physical-plan/src/topk/mod.rs`

Relevant code:

- `TopK::update_filter`
  - reads `self.heap.max()`
  - compares the max row bytes against `TopKDynamicFilters::threshold_row`
  - extracts scalar threshold values with `heap.get_threshold_values(&self.expr)`
  - builds and publishes the dynamic filter expression
- `TopK::attempt_early_completion`
  - reads `self.heap.max()` independently
  - computes the common-prefix row for the current batch's last row
  - computes the common-prefix row for the local heap max row
  - finishes when the batch prefix is strictly greater than the heap boundary prefix
- `TopKDynamicFilters`
  - stores only the shared full sort-key threshold row and dynamic filter expression:

```rust
pub struct TopKDynamicFilters {
    threshold_row: Option<Vec<u8>>,
    expr: Arc<DynamicFilterPhysicalExpr>,
}
```

There is no `TopKThreshold` type and no shared common-prefix threshold in the current codebase.

## Problem

The local heap-boundary concept is implicit. `update_filter` and `attempt_early_completion` both reason about the worst kept heap row, but each method performs its own extraction and comparison logic.

This makes future changes fragile because boundary-related behavior can drift, for example:

- full sort-key threshold comparison changes in `update_filter` but prefix-boundary comparison is not reviewed alongside it
- scalar threshold extraction and row-byte threshold comparison are no longer clearly tied to the same heap max row
- `attempt_early_completion` grows more special cases around prefix encoding without a named boundary helper
- lock-gap recheck logic in `update_filter` remains harder to scan because threshold construction and publication are mixed together

The goal is not to change semantics. The goal is to name and isolate the local heap boundary so reviewers can see that all derived threshold data comes from the same heap max row.

## Proposed refactor

Introduce a small private helper in `topk/mod.rs` for local heap-boundary handling.

Possible shape:

```rust
struct TopKHeapBoundary<'a> {
    row: &'a TopKRow,
}

impl<'a> TopKHeapBoundary<'a> {
    fn full_sort_key(&self) -> &[u8];

    fn is_more_selective_than(&self, current_threshold: Option<&[u8]>) -> bool;

    fn threshold_values(&self, heap: &TopKHeap, expr: &[PhysicalSortExpr])
        -> Result<Option<Vec<ScalarValue>>>;

    fn prefix_row(&self, topk: &TopK, scratch: &mut Rows) -> Result<()>;
}
```

Exact naming and ownership can differ. Keep it private to `topk/mod.rs`.

The refactor should make call sites read more like:

```rust
let Some(boundary) = self.current_heap_boundary() else {
    return Ok(());
};

if !boundary.is_more_selective_than(self.filter.read().threshold_row.as_deref()) {
    return Ok(());
}

let Some(thresholds) = boundary.threshold_values(&self.heap, &self.expr)? else {
    return Ok(());
};
```

For early completion, use a helper that makes the comparison intent explicit:

```rust
if self.batch_prefix_exceeds_heap_boundary(batch, boundary)? {
    self.finished = true;
}
```

## Goals

- Make the local heap-boundary concept explicit.
- Keep full sort-key threshold bytes and scalar predicate values tied to the same heap max row.
- Make prefix early-completion comparison easier to scan.
- Reduce duplicated boundary extraction and comparison code.
- Preserve behavior exactly.
- Keep changes private to `topk/mod.rs`; no public API change.

## Non-goals

- Do not change TopK semantics.
- Do not change dynamic filter pushdown behavior.
- Do not add a shared common-prefix threshold.
- Do not change partitioning or `SortExec` planning behavior.
- Do not introduce public types.

## Suggested implementation steps

1. Add a private helper for accessing the current heap boundary (`self.heap.max()`).
2. Move full sort-key selectivity comparison into that helper or a small named function.
3. Move heap-max prefix encoding/comparison behind a named helper used by `attempt_early_completion`.
4. Update `TopK::update_filter` to construct/read the boundary once and use named helpers for the read-lock fast path and write-lock recheck.
5. Update `TopK::attempt_early_completion` so the code says directly that the batch's last prefix is compared with the local heap boundary prefix.
6. Keep existing tests passing; add tests only if the refactor exposes an uncovered edge.

## Tests

At minimum run:

```bash
cargo test -p datafusion-physical-plan topk --lib
```

Relevant existing coverage in current code includes:

- `topk::tests::test_try_finish_marks_finished_with_prefix`
- `topk::tests::test_try_finish_fires_when_filter_rejects_entire_batch`
- `topk::tests::test_topk_marks_filter_complete`

If behavior is intentionally unchanged, no SQLLogicTest should be needed.

## Expected benefit

This reduces future regression risk around TopK dynamic filtering and prefix early completion by giving the local heap boundary a single, named implementation point. It should make the code easier to review without changing runtime behavior.
