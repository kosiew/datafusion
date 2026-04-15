source: pr-22306_a
# Issue: Centralize factorial result construction and Decimal256 bounds validation

## Background
The factorial UDF returns `Decimal256(76, 0)`. Today, scalar and array execution paths validate this contract differently:

- Scalar path constructs `ScalarValue::Decimal256` directly.
- Array path uses `Decimal256Array::with_precision_and_scale(76, 0)` for validation.

This split creates an invariant gap and allows behavior drift between scalar and array execution.

## Problem Statement
The current implementation does not enforce the same Decimal256 precision constraint at one shared boundary before exposing results. As a result:

- Scalar and array paths can diverge on overflow/validation behavior.
- Precision contract enforcement depends on call path instead of function semantics.
- Future edits can reintroduce mismatch by changing one path but not the other.

## Why This Matters
- Correctness: Every factorial result must fit the declared return type `Decimal256(76, 0)`.
- Consistency: Scalar and array inputs should produce equivalent success/error behavior for the same values.
- Maintainability: A single helper reduces duplicated logic and prevents subtle contract regressions.

## Scope
In scope:
- Add one shared checked helper used by both scalar and array paths.
- Ensure helper validates Decimal256 precision/scale constraints before producing output values.
- Route both scalar and array flows through this helper.
- Add regression coverage around the precision boundary.

Out of scope:
- Broad refactors outside factorial.
- Changes to factorial semantics for negatives or null handling.
- Global decimal behavior changes in unrelated UDFs.

## Proposed Solution
Introduce a small helper layer that centralizes the factorial output contract:

1. Compute factorial as i256 (existing arithmetic behavior can remain).
2. Validate that computed value fits `Decimal256(76, 0)`.
3. Return a validated Decimal256-compatible value for both scalar and array construction.

Possible shape:
- `compute_factorial_checked_decimal(n: i64) -> Result<i256>`
- or
- `factorial_decimal_value(n: i64) -> Result<Option<i256>>` (if null semantics are included)

Design goals:
- One contract check location.
- One canonical overflow/error path for precision violations.
- No behavior skew by execution mode.

## Acceptance Criteria
1. Scalar path no longer constructs unvalidated Decimal256 factorial values directly.
2. Array path uses the same contract-checking helper as scalar path.
3. `56!` succeeds and is representable as `Decimal256(76, 0)`.
4. `57!` fails with an execution error due to Decimal256 precision overflow.
5. Added tests cover both scalar and array code paths at the boundary.
6. Existing factorial behavior (nulls, negatives, normal small values) remains unchanged.

## Test Plan
Minimum required tests:

- Unit/integration tests for helper behavior:
  - `factorial(56)` succeeds.
  - `factorial(57)` errors due to precision limit.

- SQL-visible regression tests (SLT preferred for user-facing behavior):
  - Scalar query path boundary checks.
  - Array/column query path boundary checks.

Recommended assertions:
- Error class remains `Execution error`.
- Error message clearly indicates factorial overflow/precision overflow context.

## Risks and Mitigations
- Risk: Slight error-message changes may break brittle tests.
  - Mitigation: Assert stable, intentional message fragments in new tests.

- Risk: Duplicate validation could remain accidentally.
  - Mitigation: Ensure both paths call the shared helper and remove path-specific ad hoc checks where possible.

## Implementation Notes
- Keep changes localized to factorial implementation and its tests.
- Prefer existing DataFusion error macros and conventions.
- Keep public API unchanged unless strictly required.

## Definition of Done
- Shared helper implemented and used by scalar + array paths.
- Boundary tests added and passing.
- No regressions in existing factorial tests.
- Lint/tests relevant to touched crates pass.
