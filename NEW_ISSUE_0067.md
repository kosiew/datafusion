source: memory-limited-22641a
# Refactor: Add Global Left-Match Tracking for Memory-limited NestedLoopJoin

## Summary

`NestedLoopJoinExec` memory-limited spill fallback currently cannot safely run for join types that need final left-side match state when the right side has multiple partitions. The fallback path builds per-left-chunk `JoinLeftData` with local completion tracking, so it cannot know whether all right partitions have finished probing a left row before emitting left-final results.

Add global left-match coordination for the memory-limited path so spill fallback can be re-enabled for multi-partition right inputs for:

- `JoinType::Left`
- `JoinType::LeftSemi`
- `JoinType::LeftAnti`
- `JoinType::LeftMark`
- `JoinType::Full`

## Context

A correctness guard now disables spill fallback for these join types when `right_partition_count > 1`. This is safe but conservative: under memory pressure, affected joins now return OOM instead of using the memory-limited fallback.

Relevant code:

- `datafusion/physical-plan/src/joins/nested_loop_join.rs`
  - `NestedLoopJoinExec::execute`
  - `SpillState` / `SpillStateActive`
  - `handle_buffering_left_memory_limited`
  - `handle_emit_left_unmatched`
  - `update_matched_bitmap`
- `datafusion/physical-plan/src/joins/utils.rs`
  - `need_produce_result_in_final`
  - final bitmap helpers

Current right-side unmatched tracking already has a global accumulator in the memory-limited path (`global_right_bitmaps`). Left-side tracking needs an equivalent design before fallback can be safely enabled.

## Problem

In the standard path, the full left side is buffered once and `JoinLeftData` is created with `probe_threads_counter = right_partition_count`. This ensures left-final output is emitted only after all right partitions have probed.

In the memory-limited path, the left side is processed in chunks and each chunk creates a new `JoinLeftData` with `probe_threads_counter = 1`. This makes unmatched-left / semi / anti / mark decisions local to one right partition pass and one left chunk.

For multi-partition right inputs, this can produce incorrect SQL results if a left row appears unmatched in one right partition but matches in another.

## Goal

Implement global left-match tracking for memory-limited NLJ so left-final output is based on matches across all right partitions and all right passes for a left row/chunk.

Once implemented and tested, remove or relax the guard that disables fallback for left-final join types with `right_partition_count > 1`.

## Non-goals

- Do not change optimizer join selection.
- Do not change public APIs unless unavoidable.
- Do not alter join semantics for non-spill paths.
- Do not regress existing right-side global unmatched tracking.
- Do not introduce unbounded memory use without memory accounting.

## Proposed Approach

1. Design a global left-match accumulator for memory-limited execution.
   - It must represent left rows across chunks or provide a stable chunk identity.
   - It must merge matches from all right partitions before emitting left-final rows.
   - It must support `Left`, `LeftSemi`, `LeftAnti`, `LeftMark`, and `Full`.

2. Extend `SpillStateActive` or nearby memory-limited state.
   - Mirror the existing right-side global bitmap pattern where appropriate.
   - Account memory for any global bitmaps/reservations.
   - Keep per-chunk memory released when safe.

3. Adjust state transitions.
   - Avoid `EmitLeftUnmatched` until global left-match state is complete for the relevant rows.
   - Ensure all right partitions have reported completion before final left emission.

4. Preserve FULL join behavior.
   - FULL requires both left-final and right-final state.
   - Existing `global_right_bitmaps` handling must still work.

5. Re-enable fallback only after correctness is proven.
   - Replace the conservative guard with a more precise capability check.

## Acceptance Criteria

- Memory-limited fallback is correct for multi-partition right input for:
  - `Left`
  - `LeftSemi`
  - `LeftAnti`
  - `LeftMark`
  - `Full`
- Results match the non-memory-limited path for all affected join types.
- Existing single-partition fallback behavior remains unchanged.
- Existing right-family fallback behavior remains unchanged.
- Memory used by global left-match state is accounted for.
- Tests cover both matched and unmatched rows across different right partitions.

## Suggested Tests

Add targeted unit tests in `datafusion/physical-plan/src/joins/nested_loop_join.rs`:

1. Multi-partition right + tight memory + `Left`
   - A left row matches only in a non-first right partition.
   - Assert no null-padded unmatched duplicate is emitted.

2. Multi-partition right + tight memory + `LeftAnti`
   - A left row matches in a later partition.
   - Assert it is excluded.

3. Multi-partition right + tight memory + `LeftSemi`
   - A left row matches in a later partition.
   - Assert it is emitted once.

4. Multi-partition right + tight memory + `LeftMark`
   - A left row matches in a later partition.
   - Assert mark column is `true`.

5. Multi-partition right + tight memory + `Full`
   - Include both unmatched-left and unmatched-right rows.
   - Assert no duplicate/unexpected null-padded rows.

6. Compare each affected join type against the standard non-memory-limited path.

## Risks

- State-machine complexity: NLJ fallback already has multiple states and replay passes.
- Memory accounting: global left bitmaps may live longer than per-chunk buffers.
- Duplicate emission: left-final rows must be emitted exactly once.
- FULL join interaction: both global left and global right unmatched tracking must compose correctly.
- Concurrency: multiple right partitions can execute independently and must not race shared match state.

## Validation

Run targeted tests:

```bash
cargo test -p datafusion-physical-plan nlj_memory_limited -- --nocapture
cargo test -p datafusion-physical-plan nested_loop_join -- --nocapture
```

Then broader crate validation:

```bash
cargo test -p datafusion-physical-plan
```

If behavior becomes SQL-visible via planner paths, add SQLLogicTest coverage under `datafusion/sqllogictest/test_files/`.

## Implementation Size

Estimated size: Medium/Large.

Likely files:

- `datafusion/physical-plan/src/joins/nested_loop_join.rs`
- Possibly `datafusion/physical-plan/src/joins/utils.rs` for shared bitmap helpers

## Decision

Defer from the correctness-guard PR. This should be tracked as a follow-up refactor/enhancement because it restores fallback availability but requires careful state coordination and memory accounting.
