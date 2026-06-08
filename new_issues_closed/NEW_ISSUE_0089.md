source: pr-22878_a
# Refactor: Centralize final SortExec output metric observation

## Summary

`SortExec` has several fast paths that can return the final output stream without going through the normal streaming merge path. Each of those paths must record baseline output metrics (`output_rows`, `output_batches`, `output_bytes`, and `end_time`) exactly once.

Recent fixes made this invariant explicit by wrapping final fast-path streams in `ObservedStream`, or by relying on final merge streams to update the same baseline metrics. The logic is correct, but the responsibility is still spread across multiple branches and types.

Refactor the sort output construction so the invariant is encoded in one small abstraction rather than repeated per-branch wrappers and booleans.

## Problem

Final sort-output metric observation is currently decided in multiple places:

- `datafusion/physical-plan/src/sorts/sort.rs`
  - `ExternalSorter::in_mem_sort_stream`
  - Empty input fast path
  - Single in-memory batch fast path
  - In-place concat/sort fast path
  - Multi-batch streaming merge path
- `datafusion/physical-plan/src/sorts/multi_level_merge.rs`
  - `MultiLevelMergeBuilder::merge_sorted_runs_within_mem_limit`
  - Empty merge fast path
  - Single in-memory stream fast path
  - Single spill-file fast path
  - Multi-stream / spill merge paths

The current structure relies on each branch remembering whether it is producing:

1. final `SortExec` output, which must update baseline output metrics, or
2. an intermediate sorted run, which must not update final output metrics.

This is easy to regress when adding a new fast path or changing the sort/merge flow.

## Desired invariant

For every `SortExec` partition:

- The final output stream records baseline metrics exactly once.
- Intermediate sorted runs do not update final output metrics.
- Fast paths that bypass `StreamingMergeBuilder` still record:
  - `output_rows`
  - `output_batches`
  - `output_bytes`
  - `end_time`
- Merge paths that already receive final baseline metrics are not additionally wrapped in a way that double-counts output metrics.

## Suggested approach

Introduce a small private helper or wrapper in the sort module that makes final-output observation explicit.

Possible shape in `ExternalSorter`:

```rust
fn observe_if_output(
    &self,
    stream: SendableRecordBatchStream,
    is_output_stream: bool,
) -> SendableRecordBatchStream {
    if is_output_stream {
        Box::pin(ObservedStream::new(
            stream,
            self.metrics.baseline.clone(),
            None,
        ))
    } else {
        stream
    }
}
```

Possible shape in `MultiLevelMergeBuilder`:

```rust
fn observe_output(
    &self,
    stream: SendableRecordBatchStream,
) -> SendableRecordBatchStream {
    Box::pin(ObservedStream::new(stream, self.metrics.clone(), None))
}
```

Then use the helper only for branches that return a stream directly and would otherwise bypass metric collection.

Do **not** blindly wrap every returned stream. Some merge paths already update the intended baseline metrics through `StreamingMergeBuilder` / merge stream metrics. Wrapping those again could double-count output rows, batches, or bytes.

## Scope

In scope:

- Local refactor under `datafusion/physical-plan/src/sorts/`.
- Remove duplicated `ObservedStream::new(... baseline ..., None)` blocks.
- Make final-vs-intermediate output observation easier to audit.
- Preserve existing metric behavior exactly.

Out of scope:

- Changing metric names or semantics.
- Changing spill behavior.
- Changing sort output ordering, batching, fetch behavior, or memory reservation behavior.
- Large redesign of `ExternalSorter` or `MultiLevelMergeBuilder`.

## Acceptance criteria

- Final output metrics remain correct for:
  - empty sort output
  - single in-memory batch
  - small in-memory concat/sort path
  - multi-batch in-memory merge path
  - spill path with a single final spill file
  - spill path requiring multi-level merge
- Intermediate sorted runs do not increment final baseline output metrics.
- No double-counting of final output metrics.
- The final-output observation logic is centralized enough that adding a new direct-return fast path has an obvious place to attach metrics.
- Existing sort metric tests pass.

## Test guidance

Run targeted physical-plan sort tests, including metric tests around empty and small outputs:

```bash
cargo test -p datafusion-physical-plan sorts::sort::tests::empty_sort_stream_should_report_end_time
cargo test -p datafusion-physical-plan sorts::sort::tests::should_return_stream_with_batches_in_the_requested_size_and_update_metrics
```

Also run broader sort-related tests if the refactor touches merge/spill behavior:

```bash
cargo test -p datafusion-physical-plan sorts::sort
cargo test -p datafusion-physical-plan sorts::multi_level_merge
```

If a SQL-visible metric assertion exists or is added, prefer a focused regression test that verifies final output metrics are emitted for the fast path being protected.

## Risk

Low-to-medium.

The intended change is mostly mechanical, but metric ownership is subtle. The main risk is wrapping a stream that already updates baseline metrics, causing double-counting. Keep the helper scoped to direct-return fast paths and preserve existing final/intermediate distinctions.
