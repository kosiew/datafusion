stale issue
source: pr-22311_a
# Issue: Add SQLLogicTest Coverage for Empty-Pattern regexp_count with start/flags

## Summary
`regexp_count` behavior for empty patterns was fixed, but SQL-visible regression coverage currently only asserts the 2-argument path (`regexp_count('abc', '')`). The fix also affects call paths that include `start` and `flags`, and those paths are not pinned by SQLLogicTest yet.

## Why This Matters
- The motivating bug was user-visible SQL behavior.
- Unit tests validate internals, but SLT is the contract layer for end-user SQL semantics.
- Without SQL-level assertions for `start` and `flags`, regressions can reappear in one argument-shape path while still passing current SLT coverage.

## Current Gap
In [datafusion/sqllogictest/test_files/regexp/regexp_count.slt](datafusion/sqllogictest/test_files/regexp/regexp_count.slt), coverage includes:
- `SELECT regexp_count('abc', '');`

Missing SQL-visible empty-pattern cases include:
- `SELECT regexp_count('abc', '', 2);`
- `SELECT regexp_count('abc', '', 1, 'i');`
- Boundary behavior for `start` at or beyond string end (if intended semantics are stable and documented).

## Desired Behavior
For empty regex pattern `''`, all supported argument shapes should follow consistent semantics:
- 2 args: `regexp_count(str, '')`
- 3 args: `regexp_count(str, '', start)`
- 4 args: `regexp_count(str, '', start, flags)`

The SQL contract should explicitly lock expected counts for representative values and boundary positions.

## Proposed Changes
1. Extend `regexp_count.slt` with focused queries for empty-pattern calls through 3-arg and 4-arg signatures.
2. Add at least one boundary case for `start`:
   - one-past-end semantics (`start = char_len + 1`)
   - beyond-end semantics (`start > char_len + 1`)
3. Keep tests minimal and deterministic, avoiding overlap with broader unit-test internals.

## Acceptance Criteria
- SLT includes empty-pattern assertions for 2-arg, 3-arg, and 4-arg call shapes.
- At least one explicit `start` boundary behavior is asserted at SQL level.
- New assertions pass under `cargo test --test sqllogictests` from `datafusion/sqllogictest`.
- The expected results are documented by test cases and unambiguous for future reviewers.

## Out of Scope
- Refactoring `regexp_count_inner` internals.
- Changing public semantics beyond what the recent fix already intended.

## References
- PR review note: [PR_REVIEW_01.md](PR_REVIEW_01.md)
- SQL test file: [datafusion/sqllogictest/test_files/regexp/regexp_count.slt](datafusion/sqllogictest/test_files/regexp/regexp_count.slt)
