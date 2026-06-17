source: pr-22893_a
# Refactor: Separate build-row presence from matchable hash-map presence in hash join

## Summary

Hash join currently uses `Map::is_empty()` in places where the code may actually need one of two different facts:

1. The original build side has zero rows.
2. The build-side hash map has zero matchable rows.

After PR #22893, these are no longer equivalent. Under `NullEquality::NullEqualsNothing`, build rows with NULL in any join key are intentionally omitted from the map because they can never match. Therefore a build side can contain rows while `Map::is_empty()` is true.

This semantic split is correct, but the API makes future misuse easy. Refactor the hash join internals to expose the distinction directly and centralize the join-type predicates that decide which fact is needed.

## Motivation

The PR fixed a correctness issue by changing `HashJoinStream::state_after_build_ready` to check `left_data.batch().num_rows() == 0` instead of `left_data.map().is_empty()`. That avoids incorrectly completing joins whose output depends on unmatched build rows, such as `Left`, `Full`, `LeftAnti`, and mark joins.

However, this also removed a safe optimization for join types whose result is empty whenever there are no matchable build rows, even if the original build side contains only NULL-key rows. For example:

- `Inner`: no matchable build rows means no output.
- `LeftSemi`: no matchable build rows means no output.
- `RightSemi`: no matchable build rows means no output.

Today the code has to reason about this through raw calls like:

- `left_data.batch().num_rows() == 0`
- `left_data.map().is_empty()`
- `join_type.empty_build_side_produces_empty_result()`

That makes the core invariant implicit and risks future regressions where `Map::is_empty()` is accidentally used as a proxy for “build side has no rows”.

## Proposed refactor

Introduce explicit helper methods / predicates with names that encode the invariant:

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

Then centralize the join-type decisions, for example:

```rust
impl JoinType {
    /// Result is necessarily empty when the original build side has zero rows.
    fn empty_when_build_side_has_no_rows(self) -> bool { ... }

    /// Result is necessarily empty when the build side has no matchable keys,
    /// even if it contains rows that must be emitted as unmatched rows by other join types.
    fn empty_when_build_side_has_no_matchable_rows(self) -> bool { ... }
}
```

The exact names can differ, but the API should make both dimensions explicit:

- physical build batch row count
- matchable map entry count

Then update state transitions / fast paths to use the right predicate at each layer.

## Candidate behavior

`HashJoinStream::state_after_build_ready` can preserve correctness and recover safe early exits with logic like:

```rust
if !left_data.has_build_rows()
    && join_type.empty_when_build_side_has_no_rows()
{
    HashJoinStreamState::Completed
} else if !left_data.has_matchable_build_rows()
    && join_type.empty_when_build_side_has_no_matchable_rows()
{
    HashJoinStreamState::Completed
} else {
    HashJoinStreamState::FetchProbeBatch
}
```

Expected `empty_when_build_side_has_no_matchable_rows` candidates:

- `Inner`
- `LeftSemi`
- `RightSemi`

Do not include join types that must emit unmatched build rows or probe rows, such as:

- `Left`
- `Right`
- `Full`
- `LeftAnti`
- `RightAnti`
- `LeftMark`
- `RightMark`

Review exact mark / anti semantics before finalizing the predicate.

## Files likely involved

- `datafusion/physical-plan/src/joins/hash_join/stream.rs`
- `datafusion/physical-plan/src/joins/hash_join/exec.rs`
- `datafusion/physical-plan/src/joins/mod.rs`
- `datafusion/common/src/join_type.rs` if the predicates belong on `JoinType`
- hash join tests in `datafusion/physical-plan/src/joins/hash_join/exec.rs`

## Testing plan

Add targeted regression tests that force a build side with rows but no matchable map entries:

- all build join keys NULL under `NullEquality::NullEqualsNothing`
- at least one non-NULL probe batch to detect unnecessary probe scanning if observable
- both `PartitionMode::CollectLeft` and `PartitionMode::Partitioned`

Cover at least:

1. `Inner`: returns zero rows and can complete without scanning probe side.
2. `LeftSemi`: returns zero rows and can complete without scanning probe side.
3. `RightSemi`: returns zero rows and can complete without scanning probe side.
4. `Left`, `Full`, `LeftAnti`, `LeftMark`: still emit required unmatched build rows.
5. `Right`, `RightAnti`, `RightMark`: still emit required probe-side rows where applicable.

If probe-scan avoidance is hard to assert directly, use an execution plan test source that records poll count, or keep correctness tests plus a focused unit test for the predicate.

## Acceptance criteria

- Code no longer relies on ambiguous `Map::is_empty()` calls where build-row presence is required.
- Helper names distinguish original build rows from matchable hash-map entries.
- Safe early completion is restored for join types whose output is empty when there are no matchable build rows.
- Existing all-null build-key correctness tests continue to pass.
- New tests cover the predicate split and prevent reintroducing the old outer/anti/mark join bug.

## Risk / compatibility

This is an internal refactor. It should not change SQL results. The only intended behavior change is avoiding unnecessary probe-side work in cases where the output is provably empty.
