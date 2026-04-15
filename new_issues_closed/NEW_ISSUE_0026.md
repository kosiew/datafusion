create #22667
source: pr-22015_a
# Issue: Add support for `array_arg(DISTINCT x)` in sliding window execution

## Summary
#22015 added sliding-window support for `array_arg(x)` by introducing retract-based execution for the non-distinct accumulator path.

However, `array_arg(DISTINCT x)` is still not supported in sliding (bounded) window frames because the DISTINCT accumulator does not implement retract semantics.

This creates a feature gap where:
- `array_arg(x)` works in bounded sliding windows
- `array_arg(DISTINCT x)` does not

## Background and context
The recent implementation enabled sliding window support by adding `retract_batch` for the regular `array_arg` accumulator. That path supports efficient incremental updates as frame boundaries move.

In contrast, the DISTINCT path uses a separate accumulator (`DistinctArrayAggAccumulator`) that currently tracks state as a set of unique values and does not provide `retract_batch` support.

Sliding-window aggregate execution requires retract support when the frame start moves forward. If an accumulator does not support retract, the engine rejects sliding execution.

## Current behavior (problem)
`array_arg(DISTINCT x)` in bounded window frames fails at execution time due to missing retract support for the DISTINCT accumulator path.

Likely failure class:
- Not implemented for sliding accumulator because `retract_batch` is unavailable.

## Expected behavior
`array_arg(DISTINCT x)` should be executable in bounded sliding frames (`ROWS`, `RANGE`, `GROUPS`) with correct SQL semantics.

For each output row:
- Return unique values from the current frame only
- Respect DISTINCT semantics across frame movement
- Preserve existing `IGNORE NULLS` behavior
- Preserve existing restrictions/behavior for DISTINCT combined with ORDER BY

## Reproduction
Use a bounded frame that forces retract operations:

```sql
CREATE TABLE t(ts INT, val TEXT) AS VALUES
  (1, 'A'), (2, 'B'), (3, 'A'), (4, 'C'), (5, 'B');

SELECT array_arg(DISTINCT val)
OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)
FROM t;
```

Expected result:

```text
[A]
[A, B]
[A, B]
[A, C]
[B, C]
```

(Exact ordering may depend on ORDER BY behavior for DISTINCT aggregate state.)

## Why this matters
- SQL completeness: DISTINCT is a core aggregate modifier and should behave consistently across window frame types.
- User expectations: enabling `array_arg(x)` sliding support without DISTINCT creates surprising asymmetry.
- Practical analytics: de-duplicated rolling lists are common in event streams and session analytics.

## Root cause hypothesis
- Non-distinct path (`ArrayAggAccumulator`) now supports retract.
- DISTINCT path (`DistinctArrayAggAccumulator`) stores only a HashSet of unique values.
- HashSet-only state cannot safely retract by row removal when duplicate values may remain in-frame.

To support retract correctly, DISTINCT state must track multiplicity (counts), not only membership.

## Proposed implementation direction
1. Introduce retract-capable DISTINCT state
- Replace/extend HashSet state with refcounted membership:
  - `HashMap<ScalarValue, u64>` for value counts
  - Optional companion structure for deterministic ordered output if ORDER BY is required

2. Implement `retract_batch` for DISTINCT accumulator
- For each outgoing row in retract input:
  - If value is null and `IGNORE NULLS`, skip
  - Decrement count
  - Remove key when count reaches zero

3. Keep update/merge semantics consistent
- `update_batch`: increment refcount for each seen value
- `merge_batch`: merge refcounts from states correctly

4. Preserve existing DISTINCT + ORDER BY constraints
- Keep current validation that ORDER BY expressions must be argument-equivalent for DISTINCT form

5. Ensure output stability expectations are explicit
- If unordered DISTINCT output remains nondeterministic, tests should validate set equality where appropriate
- If ORDER BY provided, enforce deterministic sequence according to existing semantics

## Test plan
Add focused coverage in both unit and SQL behavior tests.

1. SQL logic tests (new section in existing sliding window test file or a dedicated file)
- `array_arg(DISTINCT val)` over `ROWS BETWEEN 1 PRECEDING AND CURRENT ROW`
- `RANGE` frame with value gaps causing multi-row retract
- `GROUPS` frame with duplicated sort keys

2. Null handling tests
- DISTINCT + `IGNORE NULLS` in sliding frames
- DISTINCT without `IGNORE NULLS` where null appears and leaves frame

3. Duplicate lifecycle tests
- Value appears multiple times; retract removes one occurrence but value must remain
- Value fully leaves frame; key must be removed

4. ORDER BY interaction tests
- Valid case: `array_arg(DISTINCT x ORDER BY x)`
- Invalid case should still error: `array_arg(DISTINCT x ORDER BY y)`

5. Regression/perf sanity
- Ensure bounded sliding windows no longer fail due to missing retract
- Basic sanity check that state growth remains bounded by frame cardinality

## Acceptance criteria
- `array_arg(DISTINCT x)` works for bounded sliding frames requiring retract.
- Results are semantically correct for duplicates, nulls, and frame movement.
- Existing non-distinct behavior and current DISTINCT restrictions remain intact.
- New SQL logic tests and unit tests cover retract-specific edge cases.
- No regression for existing `array_arg(x)` sliding window behavior.

## Out of scope
- Broad redesign of all DISTINCT aggregate internals.
- Changing documented DISTINCT + ORDER BY semantics beyond current constraints.
- Global deterministic ordering guarantees for unordered DISTINCT outputs.

## Candidate implementation touchpoints
- `datafusion/functions-aggregate/src/array_agg.rs`
- `datafusion/physical-expr/src/aggregate.rs`
- `datafusion/sqllogictest/test_files/array_agg_sliding_window.slt`

## Notes
This issue intentionally targets the feature gap exposed after sliding support for non-distinct `array_arg(x)` landed in `341410518^..bcac7c9f3`.
