source: pr-22293_a
# [Refactor] Centralize checked byte-size and offset accounting for variable-size string builders

## Summary
DataFusion currently has mixed safety behavior for variable-size string output construction.

`repeat` now performs explicit checked accounting before writing output buffers, but many other string UDF paths still rely on ad hoc capacity estimates and panic-prone builder internals (`expect("byte array offset overflow")`).

This issue proposes introducing shared, fallible helpers (or a fallible builder API) that return `DataFusionError` instead of panicking when cumulative byte offsets overflow.

## Motivation
We should enforce a consistent no-panic contract for runtime string UDF execution, especially on extreme or adversarial inputs.

Today:
- `repeat` contains local checked logic (`repeat_len`, `calculate_capacities`) to prevent overflows before array construction.
- `GenericStringArrayBuilder` still has panic paths for cumulative offset conversion.
- Several string/unicode UDF call sites still depend on heuristic or ad hoc capacity math and then append through panic-capable builder methods.

This inconsistency makes correctness auditing harder and increases risk that a code path regresses to panic behavior.

## Current Evidence
Representative locations:

1. Checked, local approach in `repeat`
- `datafusion/functions/src/string/repeat.rs`
- Uses checked multiplication/addition and explicit max-size validation before builder use.

2. Panic-capable builder internals
- `datafusion/functions/src/strings.rs`
- `GenericStringArrayBuilder` methods convert cumulative byte length to offsets via `expect("byte array offset overflow")`.

3. UDF call sites that currently rely on builder behavior + ad hoc pre-sizing
- `datafusion/functions/src/string/replace.rs`
- `datafusion/functions/src/string/common.rs`
- `datafusion/functions/src/unicode/initcap.rs`
- `datafusion/functions/src/unicode/substrindex.rs`

## Problem Statement
Variable-size string output assembly lacks a single, reusable, fallible accounting path for:
- Per-row output byte length validation
- Total output byte accumulation validation
- Offset type bound validation (`i32` for `Utf8`, `i64` for `LargeUtf8`)

As a result, failure behavior differs by function:
- Some functions return structured execution errors.
- Others can panic from builder internals when offsets exceed type bounds.

## Desired Outcome
All variable-size string-producing UDF code paths should fail with `DataFusionError` (not panic) when output byte-size or offset limits are exceeded.

## Proposed Approach
Implement a shared fallible mechanism and migrate call sites incrementally.

### Option A (preferred): Fallible builder extension
Add fallible append APIs to `GenericStringArrayBuilder` (and related builder wrappers):
- `try_append_value(&mut self, value: &str) -> Result<()>`
- `try_append_placeholder(&mut self) -> Result<()>`
- `try_append_with(...) -> Result<()>`

Internally replace `expect(...)` on offset conversion with explicit checked conversion returning `exec_err!` (or equivalent `DataFusionError`).

Pros:
- Strongest safety boundary at the lowest level.
- Prevents accidental panic regressions in new UDF code.

### Option B: Shared preflight accounting helper
Provide reusable helpers for per-row and total-capacity checked accounting and keep existing append API.

Pros:
- Smaller surface change.

Cons:
- Safety depends on every caller using preflight correctly; easier to regress.

### Recommendation
Do both in sequence:
1. Introduce shared checked accounting helper(s).
2. Add fallible builder APIs and migrate high-risk call sites.
3. Keep existing panic APIs temporarily for compatibility; deprecate internally.

## Scope
In scope:
- `datafusion/functions/src/strings.rs` builder APIs and internals
- String UDF call sites that build variable-size outputs in `datafusion/functions/src/string/*` and `datafusion/functions/src/unicode/*`
- Regression tests for overflow/no-panic behavior

Out of scope:
- Non-string builders
- Broad refactors unrelated to offset/capacity overflow behavior

## Implementation Plan
1. Add shared checked helpers for:
- `checked_repeat_len` style multiplication
- checked cumulative capacity addition
- checked conversion to target offset type

2. Introduce fallible builder append methods:
- Return `Result<()>` on offset overflow
- Preserve current performance characteristics where possible

3. Migrate selected UDFs first:
- `replace`
- case conversion path in `string/common`
- `initcap`
- `substrindex`
- other variable-size emitters discovered during audit

4. Add targeted overflow tests per migrated UDF path.

5. Optionally deprecate direct panic append methods for internal usage and enforce fallible variants in new code.

## Acceptance Criteria
- No panic on cumulative offset overflow in migrated string UDF paths.
- Overflow conditions return stable `DataFusionError` messages.
- Existing behavior/performance remains unchanged for normal-sized inputs.
- New tests cover:
  - per-item length overflow
  - cumulative output overflow
  - null-handling with overflow-adjacent inputs
- CI passes for affected crates and relevant SQLLogicTests.

## Testing Strategy
Minimum validation:
- Unit tests in `datafusion-functions` for each migrated UDF with overflow-triggering inputs.
- Keep/extend existing `repeat` overflow regression test coverage.
- Run targeted suites:
  - `cargo test -p datafusion-functions`
  - `cargo test -p datafusion-sqllogictest --test sqllogictests string_literal`

## Risks and Mitigations
Risk: API churn in builder usage across many UDFs.
Mitigation: additive fallible methods first; migrate incrementally.

Risk: performance regressions from additional checks.
Mitigation: precompute capacities when possible, keep hot-path checks minimal, benchmark representative string workloads.

Risk: inconsistent error text.
Mitigation: centralize error construction in helper(s).

## Estimated Effort
Medium.
- Initial helper + builder changes: moderate
- Call-site audit/migration + tests: moderate

## Origin
Identified as a high-impact, out-of-scope refactor opportunity during PR review of `repeat` overflow handling improvements.
