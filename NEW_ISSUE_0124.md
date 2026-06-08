source: pr-24015_a
# Convert simple final hash aggregate streams to async generators

## Problem
`FinalHashAggregateStream` and `SingleHashAggregateStream` still implement nearly identical hand-written `poll_next` state machines:

- `datafusion/physical-plan/src/aggregates/hash_stream.rs`
- `datafusion/physical-plan/src/aggregates/single_stream.rs`

Each only consumes input, transitions its hash table to output, and emits materialized batches. This duplicates typestate enums, `ControlFlow` transition plumbing, and manual `Poll` handling that `PartialReduceHashAggregateStream` now avoids with `async_try_stream`.

## Why it matters
The duplicated poll-state machinery obscures the actual aggregate lifecycle and makes cleanup, metrics, and error-path changes needlessly error-prone. The just-migrated partial-reduce stream establishes a smaller canonical implementation for this shape.

## Invariant / desired behavior
For both streams, conversion must preserve:

- input consumption and the input error boundary;
- hash-table aggregation, soft group-limit behavior in final aggregation, and output ordering;
- baseline/output metrics and spill metric registration;
- memory reservation updates and prompt input/table release before the final output batch is yielded;
- cancellation and terminal-error resource release.

## Proposed direction
Convert `FinalHashAggregateStream` and `SingleHashAggregateStream` independently to the `async_try_stream` pattern used by `partial_reduce_stream.rs` and `ordered_partial_stream.rs`:

- move each table into the generator through an `Option` field;
- wrap the generator in `RecordBatchStreamAdapter` and `ObservedStream` so baseline output metrics remain centralized;
- retain explicit reservation cleanup before the last emitted batch;
- keep final aggregation's soft-limit transition as a named helper/branch in the generator.

Do not extract a generic aggregate-stream abstraction: the marker-specific aggregation and final soft-limit semantics should remain local and readable.

## Scope
### In
- Convert `FinalHashAggregateStream` in `datafusion/physical-plan/src/aggregates/hash_stream.rs`.
- Convert `SingleHashAggregateStream` in `datafusion/physical-plan/src/aggregates/single_stream.rs`.
- Add focused regression coverage for multi-batch output and final-batch resource release for each converted lifecycle.

### Out
- `PartialHashAggregateStream`; its skip-aggregation transition is a larger, separate migration.
- `GroupedHashAggregateStream`, ordered-final aggregation, joins, sorts, and other streams with spilling, replay, or multi-stream coordination.
- Behavioral, planning, schema, or metric-surface changes.

## Acceptance criteria
- [ ] Final and single hash streams no longer implement bespoke `poll_next`/`ControlFlow` lifecycle state machines.
- [ ] Both return schema-carrying observed generator streams with equivalent baseline metrics.
- [ ] Final aggregation retains the soft group-limit path and stops reading input after the limit transition.
- [ ] The table and reservation are released before yielding the final output batch.
- [ ] No public API or aggregate result/schema behavior changes.

## Tests / verification
- Unit regression tests with a small batch size to force multiple output batches for final and single aggregation.
- Unit test for final aggregation's soft group-limit path.
- Memory-pool test that polls the final batch while retaining the stream and verifies the aggregate reservation is released.
- `cargo test -p datafusion-physical-plan aggregates --lib`.

## Notes / open questions
- `PartialHashAggregateStream` is also a plausible future generator candidate, but its `SkippingAggregation` state needs its own design/test review and should not be coupled to this cleanup.
