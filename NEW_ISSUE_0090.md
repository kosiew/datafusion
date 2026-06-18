source: pr-22893_a
# Cleanup: Name build-row and matchable-map presence checks in hash join

## Summary

Hash join now correctly distinguishes two facts after build collection:

1. The original build side has zero rows.
2. The build-side hash map has zero matchable rows.

This distinction matters because under `NullEquality::NullEqualsNothing`, build rows with NULL join keys are intentionally omitted from the map. A build side can therefore contain rows while `Map::is_empty()` is true.

The current code already handles this correctly in `HashJoinStream::state_after_build_ready` by checking both:

- `left_data.batch().num_rows() == 0`
- `left_data.map().is_empty()`

It also centralizes the join-type predicate for matchable-map emptiness with `JoinType::empty_map_produces_empty_result()`.

The remaining cleanup is naming the two presence checks on `JoinLeftData` so future call sites do not need to repeat raw `batch().num_rows()` / `map().is_empty()` logic.

## Motivation

The correctness/refactor concern from PR #22893 has already been addressed. However, the important invariant is still easier to preserve if the API names both dimensions directly:

- physical build batch row presence
- matchable hash-map entry presence

Today readers must infer the distinction from local variables in `state_after_build_ready`:

```rust
let build_empty = left_data.batch().num_rows() == 0;
let map_empty = left_data.map().is_empty();
```

That is correct, but helper names would make future uses harder to misuse and make comments/tests easier to connect to the code.

## Suggested cleanup

Add small helper methods on `JoinLeftData` or an equivalent local impl area:

```rust
impl JoinLeftData {
    fn has_build_rows(&self) -> bool {
        self.batch().num_rows() > 0
    }

    fn has_matchable_build_rows(&self) -> bool {
        !self.map().is_empty()
    }
}
```

Then update `HashJoinStream::state_after_build_ready` to use them:

```rust
let build_empty = !left_data.has_build_rows();
let map_empty = !left_data.has_matchable_build_rows();
```

Keep the existing predicates:

- `JoinType::empty_build_side_produces_empty_result()`
- `JoinType::empty_map_produces_empty_result()`

No behavior change intended.

## Files likely involved

- `datafusion/physical-plan/src/joins/hash_join/stream.rs`
- wherever `JoinLeftData` is defined in `datafusion/physical-plan/src/joins/hash_join/`

## Testing plan

This is a readability-only cleanup. Existing hash join tests should be enough if no behavior changes.

Run focused tests such as:

```bash
cargo test -p datafusion-physical-plan hash_join
```

If helpers are made public within the module and have non-trivial placement, a tiny unit test is optional but not required.

## Acceptance criteria

- `JoinLeftData` exposes named helpers for:
  - original build-side row presence
  - matchable hash-map row presence
- `state_after_build_ready` uses the helpers instead of raw checks.
- No SQL/result behavior changes.
- Existing hash join tests pass.

## Scope notes

This is now a small cleanup issue, not a broad refactor or correctness fix. The main semantic split and safe early completion logic are already present in the current codebase.
