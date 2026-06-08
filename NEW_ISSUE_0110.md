source: dynamic-filter-01-22772a
# Evaluate table-driven shared-bounds expression-policy tests

## Problem
The new `shared_bounds.rs` expression-policy tests repeat setup and assertion flow across `CollectLeft` and `Partitioned` cases. A table-driven layout could reduce repetition, but the current explicit tests make each policy branch easy to read.

The risk is that a table-driven refactor may hide important mode-specific behavior, especially:
- `CollectLeft` with no membership/bounds does not update the dynamic filter.
- `Partitioned` empty/all-empty/canceled paths produce explicit fallback expressions.
- `Partitioned` may intentionally use or skip `CASE` depending on partition state.

## Why it matters
These tests are characterization tests for future dynamic-filter refactors. If test structure hides branch-specific intent, later production refactors can accidentally change fallback policy while tests remain hard to diagnose.

## Invariant / desired behavior
Every test case must still map clearly to one expression-policy rule, and failures must identify the violated rule without relying on brittle expression strings.

## Proposed direction
Only table-drive these tests if the table keeps policy intent explicit:
- use named cases;
- keep expected shape structural, e.g. an enum such as `ExpectedFilterShape`;
- keep `CollectLeft` and `Partitioned` groups separate if combining them reduces clarity;
- avoid snapshot/string-only assertions.

If the table makes the tests harder to read than the current explicit tests, leave the tests as-is.

## Scope
### In
- Refactor test code in `datafusion/physical-plan/src/joins/hash_join/shared_bounds.rs` only.
- Preserve the same inputs, expected expression shapes, and assertions.
- Improve or preserve failure messages.

### Out
- No production code changes.
- No dynamic-filter behavior changes.
- No public or test-visible API changes.
- No added coverage for `Map`, multi-column joins, or extra scalar types as part of this refactor.

## Acceptance criteria
- [ ] All existing expression-policy scenarios remain covered.
- [ ] Each case has a descriptive name tied to the policy branch it protects.
- [ ] Assertions remain structural (`InListExpr`, `BinaryExpr`, `CaseExpr`, literal bool), not display-string snapshots.
- [ ] Failure output is at least as clear as the current explicit tests.
- [ ] If clarity regresses, no change is made.

## Tests / verification
- `cargo test -p datafusion-physical-plan shared_bounds`

## Notes / open questions
- This is optional cleanup. Current explicit tests are acceptable and may be preferable if table-driving obscures mode-specific invariants.
