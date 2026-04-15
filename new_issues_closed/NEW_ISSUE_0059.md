Verdict: Not high-impact as written.
 Good cleanup only if narrow: helper for “read nullable Int64 row value”; callers
 keep policy explicit.

source: pr-22508_a
# Issue 01: Introduce a Shared Nullable Count Row Helper for Nested Array Functions

## Summary
Add a small helper in datafusion/functions-nested for extracting per-row count or size inputs from nullable Int64 arguments in a consistent way across array_remove, array_replace, array_repeat, and array_resize.

## Background
The current implementations in remove.rs, replace.rs, repeat.rs, and resize.rs all perform similar count handling in row loops:
- Branch between scalar-style single value and per-row array values.
- Interpret nullable Int64 values as optional row counts.
- Apply per-function behavior for negative or zero counts.

This logic is repeated with slight local differences, increasing the risk of drift and subtle contract regressions.

## Problem Statement
Per-row count extraction and null handling is duplicated and manually encoded in multiple places. This duplication makes it harder to:
- Audit SQL-visible null semantics.
- Evolve behavior safely.
- Keep edge-case handling aligned.

## Goals
- Provide one shared helper API that normalizes how nullable Int64 row counts are read.
- Keep behavior identical for existing functions.
- Keep function-specific semantics explicit and outside the helper where needed.

## Non-Goals
- Do not unify all row null semantics into one rule.
- Do not change SQL behavior or function signatures.
- Do not optimize unrelated internals.

## Proposed Design
1. Add a focused helper module, for example datafusion/functions-nested/src/utils/row_count.rs.
2. Introduce a lightweight abstraction for row count access, such as:
   - single count reused for all rows, or
   - per-row optional count stream.
3. Expose a small API surface:
   - fetch row count as Option<i64> for an index,
   - preserve source nulls,
   - avoid hidden coercion rules.
4. Keep downstream interpretation in each function (for example, clamping non-positive values) to preserve per-function semantics.

## Acceptance Criteria
- A reusable helper exists and is used by at least one nested array function.
- Existing behavior remains unchanged for null, zero, negative, and positive count cases.
- Helper API is documented with clear semantic boundaries.
- No measurable regression in existing relevant test suites.

## Test Plan
- Add unit tests for helper behavior:
  - scalar count present,
  - scalar count null,
  - per-row counts with mixed null/non-null,
  - bounds and index access behavior.
- Run targeted suites:
  - datafusion-functions-nested unit tests for touched functions,
  - sqllogictest files covering array_remove, array_replace, array_repeat, array_resize.

## Risks and Mitigations
- Risk: Helper accidentally encodes policy that belongs in callers.
  - Mitigation: Keep helper narrow and policy-free; review for semantic leakage.
- Risk: Subtle SQL behavior drift.
  - Mitigation: Preserve current behavior with regression tests before migration.

## Suggested Execution Order
1. Land helper with unit tests.
2. Migrate one function first (array_replace or array_remove).
3. Validate behavior parity before wider adoption.
