Good issue if goal = tidy macro + parity tests.
 Not strong if prioritizing major maintainability/perf/architecture wins.

 Best framing: “small targeted deduplication with tests,” not “high impact.”


source: pr-22308_a
# Issue: Consolidate typed unary math evaluation in macro-generated unary functions

## Summary
The unary math UDF macro currently duplicates Float64 and Float32 array evaluation logic in separate match arms, including validator invocation and error conversion.

This duplication increases maintenance overhead and creates a repeated surface for subtle inconsistencies when adding or changing domain validators.

## Source Context
- Review source: [PR_REVIEW_01.md](PR_REVIEW_01.md#L18)
- Implementation location: [datafusion/functions/src/macros.rs](datafusion/functions/src/macros.rs#L286)

The relevant section in [datafusion/functions/src/macros.rs](datafusion/functions/src/macros.rs#L286) contains near-identical `try_unary` branches for:
- Float64 path: [datafusion/functions/src/macros.rs](datafusion/functions/src/macros.rs#L287)
- Float32 path: [datafusion/functions/src/macros.rs](datafusion/functions/src/macros.rs#L303)

## Problem Statement
Within `make_math_unary_udf`, the Float64 and Float32 branches each perform the same sequence:
1. Convert the input array into a typed primitive array.
2. Call `try_unary` with type-specific lambda.
3. If a validator exists, run validator logic.
4. Convert validator errors into `ArrowError::ComputeError`.
5. Apply type-specific intrinsic (for example, `f64::$UNARY_FUNC` or `f32::$UNARY_FUNC`).
6. Wrap result as `ArrayRef`.

Only numeric type wiring and value cast differ, while control flow and validation/error mapping are duplicated.

## Why This Matters
- Higher change cost: New domain checks or validator semantics need edits in two places.
- Drift risk: One branch can diverge (message, cast behavior, validator call shape) from the other.
- Readability cost: Macro body is longer and harder to reason about for future function additions.

## Proposed Refactor
Introduce a small shared adapter for unary evaluation to centralize validator + error mapping behavior.

Potential approaches:
1. Local typed helper function inside generated module, parameterized over Arrow primitive type and Rust scalar type.
2. Closure adapter helper that wraps validator handling and returns a closure consumable by `try_unary`.

Preferred direction:
- Keep macro call site unchanged.
- Add one helper used by both Float64 and Float32 branches.
- Keep current type-specific execution and return-type behavior intact.

## Non-Goals
- No behavior changes to function signatures or return-type coercion.
- No expansion of accepted argument types.
- No broad macro redesign beyond unary evaluation deduplication.

## Acceptance Criteria
1. `make_math_unary_udf` no longer duplicates validator and error-conversion logic across Float64/Float32 paths.
2. Runtime behavior for existing unary math functions remains unchanged.
3. Error text and error class remain compatible with current expectations.
4. Unsupported-type branch behavior remains unchanged.
5. Existing tests pass; targeted new tests validate parity for both float widths.

## Suggested Tests
Add focused tests in the functions crate that exercise validator path symmetry across Float64 and Float32:
- Negative-domain validator example (such as sqrt negative input) for Float64 array input.
- Same for Float32 array input.
- Verify null propagation and successful non-negative values still behave as expected.

Reference area for macro behavior:
- [datafusion/functions/src/macros.rs](datafusion/functions/src/macros.rs#L286)

## Risks and Mitigations
- Risk: Generic helper introduces type-bound complexity in macro-expanded code.
  - Mitigation: Keep helper local, minimal, and with explicit type annotations.
- Risk: Behavior drift in validator conversion.
  - Mitigation: Preserve current conversion path to `ArrowError::ComputeError` and add width-parity tests.

## Effort Estimate
Small to medium.
- Refactor complexity: low to moderate.
- Validation and tests: moderate.
- Expected review surface: mostly confined to [datafusion/functions/src/macros.rs](datafusion/functions/src/macros.rs).

## Implementation Notes for Assignee
- Start by extracting only the shared validator/error mapping logic.
- Avoid touching unrelated macro branches.
- Prefer a targeted change set plus tests to keep review risk low.