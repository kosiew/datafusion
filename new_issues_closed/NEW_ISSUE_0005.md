created #22061
# Issue Draft 01

## Title
Refactor scalar min/max dispatch from macro arms to function-based logic

## Summary
`min_max_scalar` in `datafusion/functions-aggregate-common/src/min_max.rs` was extracted to make error-path control flow testable, but the core scalar dispatch is still encoded in the large `min_max_scalar_impl!` macro. This macro packs type matching, validation, recursion, and return-flow behavior into non-local expansion paths that are harder to step through, test in isolation, and evolve safely.

This issue proposes replacing macro-heavy scalar dispatch with a function-first implementation that preserves current semantics and error messages while improving testability and maintainability.

## Current State
- Scalar dispatch is implemented in `min_max_scalar_impl!` with many match arms.
- Dictionary recursion and dictionary-key-type invariants are embedded inside macro expansion.
- Decimal compatibility checks and some incompatible-type errors are spread across macro arms.
- `min_max_scalar` currently wraps macro invocation and maps `Ordering` to `min` or `max` semantics.

## Problem
- Macro expansion hides control flow and increases debugging friction.
- It is difficult to unit test small dispatch decisions without exercising large macro branches.
- Future behavior changes risk reintroducing non-local control-flow mistakes.
- Compiler errors and review diffs are harder to reason about in a large macro body.

## Goals
- Move scalar dispatch to regular Rust functions with explicit control flow.
- Preserve all externally visible behavior:
  - Same min/max results for all currently supported scalar variants.
  - Same dictionary semantics and key-type mismatch invariants.
  - Same decimal precision/scale checks.
  - Same NaN behavior.
  - Same error class and equivalent error text for incompatible inputs.
- Keep the change crate-scoped and reviewable.

## Non-Goals
- No dictionary batch optimization work in this issue.
- No semantic changes to aggregate min/max behavior.
- No broad refactor of batch kernels outside scalar dispatch.

## Proposed Approach
1. Introduce function-based dispatch helpers:
   - `fn min_max_scalar(lhs: &ScalarValue, rhs: &ScalarValue, ordering: Ordering) -> Result<ScalarValue>` remains the entrypoint.
   - Add internal helpers to isolate concerns, such as:
     - `fn min_max_scalar_same_variant(...) -> Result<ScalarValue>`
     - `fn min_max_dictionary_scalar(...) -> Result<ScalarValue>`
     - `fn ensure_decimal_compatibility(...) -> Result<()>`
2. Replace macro match arms with explicit `match (lhs, rhs)` in one or more functions.
3. Keep current compacting and cloning behavior where relevant.
4. Preserve current dictionary rewrap behavior when key types match.
5. Remove `min_max_scalar_impl!` after parity is proven.

## Implementation Checklist
- [ ] Add function-based scalar dispatch helpers in `datafusion/functions-aggregate-common/src/min_max.rs`.
- [ ] Keep behavior parity for all scalar variants currently handled in macro arms.
- [ ] Preserve dictionary recursive comparison semantics.
- [ ] Preserve error behavior for incompatible scalar combinations.
- [ ] Remove `min_max_scalar_impl!` once parity tests pass.

## Testing Plan
- Extend unit tests in `datafusion/functions-aggregate-common/src/min_max.rs` with table-driven parity checks over representative scalar pairs.
- Keep and validate dictionary-focused tests:
  - dictionary vs scalar compare-by-inner-value
  - dictionary vs dictionary same key type rewrap
  - dictionary key type mismatch error
  - dictionary vs incompatible scalar error
- Add targeted tests for decimal precision/scale mismatch and fixed-size-binary size mismatch.
- Run crate-scoped tests:
  - `cargo test -p datafusion-functions-aggregate-common min_max`

## Acceptance Criteria
- `min_max_scalar_impl!` is removed.
- `min_max_scalar` behavior is unchanged for all existing test cases.
- New tests cover control-flow edges previously hidden in macro expansion.
- No regressions in aggregate min/max tests that depend on scalar comparison.

## Risks and Mitigations
- Risk: subtle semantic drift during macro-to-function rewrite.
  - Mitigation: parity-focused tests and incremental refactor with small commits.
- Risk: accidental allocation/cloning regressions.
  - Mitigation: preserve clone/compact patterns; review hot-path changes carefully.

## Related Context
This follows review feedback that a function-first implementation would improve debuggability and reduce the chance of reintroducing non-local control-flow surprises.
