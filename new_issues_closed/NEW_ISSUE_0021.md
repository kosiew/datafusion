created #22641
source: pr-21833_a
# Issue: Correctness guard: disable memory-limited NestedLoopJoin fallback for LEFT-family joins with multi-partition right side

## Summary

Memory-limited NestedLoopJoin execution can still produce incorrect unmatched-left output when:
1. The right side has multiple output partitions (target_partitions > 1), AND
2. Join type requires final unmatched-left emission for LEFT-family behavior (LEFT, LEFT ANTI, LEFT MARK, and any join mode using left-final emission semantics)

**Root cause:** In memory-limited fallback, per-chunk unmatched-left emission uses a per-stream probe completion counter initialized to 1, instead of coordinating across all right partitions like the single-pass path does. This can cause a left row to be marked "unmatched" by partition 0 before partition 1 finishes probing and matches it.

**Impact:** Query results can become incorrect and memory-dependent, violating LEFT-family join semantics under memory pressure.

## Why this matters

- **Correctness:** LEFT-family join results can differ based on memory availability, breaking deterministic SQL semantics.
- **Severity:** Large inputs (where fallback activates) are the exact scenarios where this breaks, not edge cases.
- **User impact:** Queries may emit incorrect null-padded unmatched-left rows under memory pressure.

## Current status in codebase

- **Current behavior (introduced in 36b1927b05):** despite broader RIGHT/FULL fallback support in that change, FULL JOIN with multiple right partitions is intentionally blocked from memory-limited fallback (`full_join_multi_partition` guard).
- **Still open:** LEFT-family join paths are still allowed to fallback and still use per-chunk local completion (`probe_threads_counter = 1`), which does not coordinate unmatched-left emission globally across right partitions.

## Current behavior and gap

### Single-pass path behavior

The normal path buffers the full left side and uses right partition count to coordinate unmatched-left emission only after all right partitions finish probing.

Relevant code:
- datafusion/physical-plan/src/joins/nested_loop_join.rs around collect_left_input call in execute
- datafusion/physical-plan/src/joins/nested_loop_join.rs around process_left_unmatched (report_probe_completed gate)

### Memory-limited fallback behavior

In fallback, each left chunk builds JoinLeftData with probe_threads_counter set to 1 (per stream/per partition chunk instance), so unmatched-left gating is local rather than global across right partitions.

Relevant code:
- datafusion/physical-plan/src/joins/nested_loop_join.rs in handle_buffering_left_memory_limited where JoinLeftData::new is created with AtomicUsize::new(1)

## Reproduction concept

1. Force memory-limited fallback with a tight memory limit.
2. Ensure right side is partitioned into multiple partitions (target_partitions > 1 and plan shape that preserves multi-partition probe behavior).
3. Use LEFT JOIN (or LEFT ANTI / LEFT MARK) with a predicate where a given left row matches only rows in a non-local right partition.
4. Observe that another partition may emit that left row as unmatched.

Expected:
- No unmatched-left row for left rows that match in any right partition.

Observed risk:
- Duplicate or incorrect null-padded unmatched-left output.

## Proposed fix (correctness guard, not refactor)

The most direct fix is to **disable memory-limited fallback for LEFT-family join modes that require unmatched-left emission** when `right_partition_count > 1`, until proper cross-partition coordination is implemented (tracked separately in [NEW_ISSUE_0022](NEW_ISSUE_0022.md)).

This trades off spill-fallback availability for correctness:
- LEFT, LEFT ANTI, LEFT MARK (and other left-final-emission variants) will OOM instead of spill when the right side has multiple partitions.
- FULL remains unchanged: it is already blocked today.
- INNER, RIGHT, RIGHT SEMI, RIGHT ANTI, RIGHT MARK continue to spill normally.
- Future PRs can re-enable spill for LEFT-family joins with shared chunk-state coordination.

Alternative (more complex): implement cross-partition coordination now as part of this PR fix (see design in [NEW_ISSUE_0022](NEW_ISSUE_0022.md) for guidance).

### Exact join-type scope for fallback-disable guard

Guard should explicitly cover:
- `JoinType::Left`
- `JoinType::LeftAnti`
- `JoinType::LeftMark`

Guard should explicitly not cover:
- `JoinType::Inner`, `JoinType::Right`, `JoinType::RightSemi`, `JoinType::RightAnti`, `JoinType::RightMark` (spill fallback remains enabled)
- `JoinType::Full` (already blocked by existing `full_join_multi_partition` guard)

## Acceptance criteria (disabling-fallback approach)

1. **Correctness restored:** LEFT-family joins with multi-partition right input no longer produce duplicate/incorrect unmatched-left rows.
   - Implementation: Add a join-type check in `initiate_fallback` that returns Disabled state (or error) for `JoinType::Left | JoinType::LeftAnti | JoinType::LeftMark` when `right_partition_count > 1`.

2. **Regression test:** Add deterministic unit test in nested_loop_join.rs
   - Setup: multi-partition right side, LEFT JOIN, tight memory limit.
   - Expected: deterministic error with stable substring, for example: `memory-limited NestedLoopJoin fallback disabled for LEFT-producing join with multi-partition right side`.
   - Non-goal for this PR: test should not assert full formatted error text, only stable substring.
   - Verify: query still succeeds with single-partition right side.

3. **FULL behavior unchanged:** Existing FULL JOIN multi-partition fallback-disabled behavior remains intact.

4. **No regressions on other join types:**
   - INNER, RIGHT, RIGHT SEMI, RIGHT ANTI, RIGHT MARK continue to spill.
   - Existing memory-limited test suite green.

5. **Documentation:**
   - Code comment explaining why LEFT-family joins don't fallback with multi-partition right side.
   - Link to [NEW_ISSUE_0022](NEW_ISSUE_0022.md) for future architectural fix.

## Suggested tests

### Unit test in nested_loop_join.rs

```rust
#[tokio::test]
async fn test_nlj_memory_limited_left_join_multi_partition_fallback_disabled() -> Result<()> {
    // Setup: multi-partition right side + tight memory limit to force fallback path
    // Build data so at least one left row matches only in a non-local right partition.

    // Expectation:
    // - multi-partition LEFT join under memory pressure should not spill fallback;
    //   it should fail fast (OOM/disabled fallback) until global coordination exists.
    // - single-partition LEFT join should continue to succeed.
    Ok(())
}
```

## Non-goals

- Implementing cross-partition chunk-state sharing in this PR (deferred to [NEW_ISSUE_0022](NEW_ISSUE_0022.md)).
- Changing optimizer rules or join selection logic.

## Definition of done

1. Check-in disables memory-limited fallback for LEFT-producing joins with multi-partition right side.
2. New unit test verifies the error/disabled behavior.
3. Existing FULL JOIN fallback-disabled behavior remains covered.
4. Existing memory-limited test suite remains green (other join types unaffected).
5. Code comment links to [NEW_ISSUE_0022](NEW_ISSUE_0022.md) for the longer-term fix.
6. PR description references this correctness issue explicitly.
