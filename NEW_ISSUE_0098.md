source: pr-22991_a
# Refactor: Centralize TopK threshold and prefix-boundary handling

## Summary

`TopK` now uses the same heap-derived boundary in two related optimizations:

1. dynamic filter pushdown (`TopK::update_filter`)
2. prefix early completion (`TopK::attempt_early_completion`)

The important invariant is that the full sort-key threshold and the common-prefix threshold must describe the same heap row. Today that invariant is spread across `update_filter`, `attempt_early_completion`, `encode_topk_common_prefix_row`, and `TopKThreshold`.

Refactor this into a small helper/type so threshold construction, comparison, and prefix-boundary checks are owned in one place.

## Context

Recent TopK dynamic-filter work added shared thresholds for partition-preserving `SortExec`. Each output partition has a local `TopK` heap, while all partitions share one `TopKDynamicFilters` instance.

A local heap can publish its current worst kept row as the shared threshold. Other partitions can then:

- use the full sort-key threshold to reject rows through the dynamic filter, and
- use the common-prefix threshold to stop early once their ordered input has moved past the shared boundary.

That makes this invariant central:

> A shared threshold's full sort-key row and common-prefix row must be encoded from the same `TopKRow` / input row.

Current relevant code:

- `datafusion/physical-plan/src/topk/mod.rs`
  - `TopK::update_filter`
  - `TopK::attempt_early_completion`
  - `TopK::encode_topk_common_prefix_row`
  - `TopKThreshold`

## Problem

The implementation is correct, but the invariant is implicit and split across several methods:

- `update_filter` obtains `heap.max()`, compares the full sort-key bytes, builds a predicate, then separately calls `encode_topk_common_prefix_row(max_row)`.
- `attempt_early_completion` compares the current batch prefix first against the shared threshold prefix and then against a locally re-encoded heap max prefix.
- `TopKThreshold` stores both byte rows, but does not own the logic for constructing a candidate threshold from a heap row or testing prefix completion.

This is fragile because future TopK changes could accidentally compare or publish mismatched pieces, for example:

- full sort-key bytes from one heap row but prefix bytes from another,
- a predicate built from threshold scalar values that no longer match the stored threshold row,
- shared-prefix early exit using different comparison rules from local-prefix early exit,
- duplicated threshold selectivity checks diverging over time.

## Proposed direction

Introduce a small private abstraction in `topk/mod.rs` that owns heap-boundary construction and comparison.

Possible shape:

```rust
struct TopKBoundary {
    full_sort_key_row: Vec<u8>,
    common_prefix_row: Option<Vec<u8>>,
    threshold_values: Vec<ScalarValue>,
}

impl TopKBoundary {
    fn try_from_heap_max(topk: &TopK, row: &TopKRow) -> Result<Option<Self>>;
    fn is_more_selective_than(&self, current: &TopKThreshold) -> bool;
    fn finishes_prefix(&self, batch_common_prefix: &[u8]) -> bool;
    fn into_threshold(self) -> TopKThreshold;
}
```

Exact naming can differ. The key is to make the call sites express the invariant directly:

```rust
let Some(candidate) = self.current_heap_boundary()? else {
    return Ok(());
};

if !candidate.beats_shared_threshold(&self.filter) {
    return Ok(());
}

let predicate = Self::build_filter_expression(&self.expr, candidate.threshold_values())?;
self.publish_threshold(candidate, predicate)?;
```

For early completion, use one helper for both shared and local checks:

```rust
if self.shared_boundary_finishes(batch_common_prefix)
    || self.local_boundary_finishes(batch_common_prefix)?
{
    self.finished = true;
}
```

## Goals

- Make the threshold invariant explicit in code.
- Keep full sort-key threshold, common-prefix threshold, and scalar predicate values tied to one heap row.
- Reduce duplicate comparison/control-flow between shared and local prefix early-exit paths.
- Keep this private to `topk/mod.rs`; no public API change.
- Preserve behavior exactly.

## Non-goals

- Do not change TopK semantics.
- Do not change dynamic filter pushdown behavior.
- Do not change partitioning or `SortExec` planning behavior.
- Do not add new public types unless there is a separate API need.

## Suggested implementation steps

1. Add a private helper for encoding a heap row into a threshold/boundary object.
2. Move full sort-key selectivity comparison into that helper.
3. Move common-prefix completion comparison into that helper or a closely related helper.
4. Update `TopK::update_filter` to construct one candidate boundary and publish it only if still more selective.
5. Update `TopK::attempt_early_completion` to use the same comparison helper for shared and local thresholds.
6. Keep existing tests passing; add focused tests only if refactor exposes an uncovered edge.

## Tests

At minimum run:

```bash
cargo test -p datafusion-physical-plan topk --lib
```

Useful existing coverage includes:

- `topk::tests::test_shared_filter_can_finish_partition_before_local_heap_is_full`
- `topk::tests::test_shared_prefix_threshold_boundary_cases`
- `topk::tests::test_early_completion_marks_finished_with_prefix`
- `topk::tests::test_early_completion_fires_when_filter_rejects_entire_batch`
- `topk::tests::test_early_completion_fires_when_batch_makes_no_replacements`
- `sorts::sort::tests::test_preserved_topk_filter_waits_for_all_sort_partitions`

If behavior is intentionally unchanged, no SQLLogicTest should be needed.

## Expected benefit

This reduces the chance of future correctness regressions in TopK dynamic filtering, especially around partition-preserving `SortExec`, null ordering, descending ordering, and prefix early exit. It also makes the code easier to review because the shared/local threshold contract is encoded once instead of inferred from several call sites.
