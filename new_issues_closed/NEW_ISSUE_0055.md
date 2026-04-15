stale
source: pr-22274_a
# Issue 01: Centralize repeat count and offset preflight in nested repeat implementation

## Summary
The repeat implementation currently performs count-derived capacity and offset validation in multiple places with duplicated logic. This creates drift risk, where one path is hardened while another path still allows unchecked arithmetic or panic-prone offset construction.

A shared preflight helper should compute and validate totals and offset bounds once, then feed both scalar and list repeat paths.

## Scope Origin
From [PR_REVIEW_01.md](PR_REVIEW_01.md#L26-L31).

## Background
The nested repeat implementation has two major paths:
- Scalar or non-list repeat path in [datafusion/functions-nested/src/repeat.rs](datafusion/functions-nested/src/repeat.rs#L171)
- List-input repeat path in [datafusion/functions-nested/src/repeat.rs](datafusion/functions-nested/src/repeat.rs#L246)

Both derive repeated counts, capacities, and offsets from the same input count array, but do so independently.

## Problem Statement
Duplicated preflight and offset logic increases the chance of inconsistent safety behavior.

Recent review findings already show this drift pattern:
- One path can allocate large capacity before confirming representability in output offset type.
- Another path still performs unchecked sentinel capacity arithmetic and relies on panic-prone offset construction helpers for untrusted count-derived lengths.

This means overflow and offset safety guarantees are not defined in one place, and are easy to regress.

## Why This Matters
- Correctness: repeated-count arithmetic must always return execution errors, not panic or OOM edge behavior.
- Maintainability: duplicated logic makes future fixes easy to miss.
- Reviewability: shared invariant enforcement clarifies expected behavior and lowers audit cost.
- Testability: one helper makes it easier to add targeted regression tests.

## Proposed Change
Add a local preflight helper for repeat operations that:
- Normalizes count values the same way as runtime logic.
- Computes all required totals with checked arithmetic.
- Validates offset-type representability with output offset type conversions before any allocation.
- Returns precomputed totals and any reusable derived values required by both paths.

Use this helper in both:
- Scalar repeat path in [datafusion/functions-nested/src/repeat.rs](datafusion/functions-nested/src/repeat.rs#L171)
- List repeat path in [datafusion/functions-nested/src/repeat.rs](datafusion/functions-nested/src/repeat.rs#L246)

Replace duplicated ad hoc preflight math with helper output.

## Out of Scope
- Broad refactor of repeat semantics beyond preflight and offset safety.
- Changing SQL-visible semantics for null or non-positive counts.
- Cross-crate API extraction.

## Acceptance Criteria
1. Both repeat paths consume a shared preflight helper for count-derived totals and offset-bound checks.
2. No count-derived allocation happens before required offset-type representability is validated.
3. Sentinel additions for offset vectors use checked arithmetic.
4. Panic-prone offset constructors are not used with untrusted count-derived totals when equivalent checked construction is available.
5. Existing behavior for valid inputs remains unchanged.
6. Overflow and out-of-range offset scenarios return execution errors with stable, actionable messages.

## Test Plan
### Unit tests
- Add focused tests near repeat implementation that validate:
  - Offset bound violation is returned before large capacity allocation paths.
  - Sentinel offset capacity arithmetic handles overflow via error, not panic.
  - Shared preflight helper returns expected totals for mixed null, zero, and positive counts.

### SQL-level coverage
- Add or extend SQL logic tests for representative overflow and offset-bound failures where practical.
- Ensure failures are SQL-visible as execution errors.

## Risk and Mitigation
- Risk: accidental behavior change in valid repeat outputs.
  - Mitigation: preserve existing successful path outputs and add regression tests for normal and edge inputs.
- Risk: error message drift.
  - Mitigation: reuse existing error wording where possible.

## Suggested Labels
- bug
- refactor
- safety
- good first issue (optional, if maintainers agree on scope)

## Definition of Done
- Shared preflight helper implemented and used by both repeat paths.
- Targeted unit tests and SQL-level coverage added.
- No new panic/OOM edge from count-derived overflow or offset-bound failures.
- CI checks pass for touched crates and relevant test suites.
