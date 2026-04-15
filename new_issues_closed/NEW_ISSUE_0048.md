created #22685
source: pr-22408_a
# Issue: Centralize checked timestamp scaling for DATE_BIN

## Summary
DATE_BIN currently performs timestamp unit scaling in multiple call sites before invoking the binning function. This logic is duplicated across scalar and array paths and primarily uses direct multiplication. The existing approach increases maintenance cost and risks inconsistent overflow/error behavior across code paths.

## Source
- Review note in [PR_REVIEW_01.md](PR_REVIEW_01.md#L24)
- Current implementation in [datafusion/functions/src/datetime/date_bin.rs](datafusion/functions/src/datetime/date_bin.rs)

## Problem Statement
The DATE_BIN implementation contains repeated patterns that convert input values to nanoseconds using expressions like value * scale in multiple closures and match arms.

Examples include:
- Timestamp scalar mapping logic in [datafusion/functions/src/datetime/date_bin.rs](datafusion/functions/src/datetime/date_bin.rs#L593)
- Timestamp array mapping logic in [datafusion/functions/src/datetime/date_bin.rs](datafusion/functions/src/datetime/date_bin.rs#L719)
- Time scalar paths in [datafusion/functions/src/datetime/date_bin.rs](datafusion/functions/src/datetime/date_bin.rs#L648)
- Time array paths in [datafusion/functions/src/datetime/date_bin.rs](datafusion/functions/src/datetime/date_bin.rs#L758)

Because scaling is spread across many locations:
- Overflow handling is not centrally enforced by one helper.
- Error mapping behavior can diverge between scalar and array code paths.
- Future fixes must be replicated in several places.

## Why This Matters
- Correctness: unit scaling is arithmetic on i64 values and can overflow for edge inputs.
- Consistency: DATE_BIN should produce predictable behavior regardless of scalar vs array path.
- Maintainability: duplicated conversion logic increases risk of regressions.

## Proposed Change
Introduce a small helper local to DATE_BIN implementation to centralize checked scaling:

- Helper shape: value_to_nanos(value: i64, scale: i64) -> Result<i64>
- Internals: use checked_mul and return a structured execution/compute error on overflow.
- Adoption: replace direct value * scale call sites in DATE_BIN scalar and array paths.
- Error behavior: keep existing external behavior, but standardize internal overflow handling and message shape.

Optional follow-on helper for reverse conversion can be considered if it reduces duplication further, but initial scope should focus on input scaling.

## Scope
In scope:
- Refactor in [datafusion/functions/src/datetime/date_bin.rs](datafusion/functions/src/datetime/date_bin.rs)
- Update/add unit tests in the same module test section as needed

Out of scope:
- Broad cross-function refactors outside DATE_BIN
- Behavioral changes unrelated to scaling overflow handling

## Acceptance Criteria
1. All DATE_BIN input scaling to nanoseconds uses the new checked helper.
2. No direct value * scale remains in DATE_BIN timestamp/time scalar and array scaling paths where overflow is possible.
3. Scalar and array paths produce consistent overflow handling semantics.
4. Existing DATE_BIN tests continue to pass.
5. New or updated tests cover at least one overflow edge for scaled input conversion.

## Test Plan
- Run targeted function tests for date_bin module.
- Add/adjust unit tests in [datafusion/functions/src/datetime/date_bin.rs](datafusion/functions/src/datetime/date_bin.rs) to validate helper-based overflow handling.
- If SQL-visible behavior is affected by error-to-NULL mapping, add/extend SQL logic tests in [datafusion/sqllogictest/test_files/datetime/timestamps.slt](datafusion/sqllogictest/test_files/datetime/timestamps.slt).

## Implementation Notes
- Keep refactor crate-scoped and minimal.
- Reuse existing DataFusion error conventions already used in DATE_BIN.
- Preserve current API and output types.

## Priority
Medium

## Estimated Effort
Small to medium
