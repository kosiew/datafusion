source: pr-22564_a
# Refactor Issue 02: Make `log` simplification domain-aware by sharing validation semantics with runtime evaluation

## Title
Centralize `log` domain validation so algebraic simplification cannot bypass runtime errors

## Area
- `datafusion/functions/src/math/log.rs`
- Expression simplification for scalar UDFs
- SQL-visible math function semantics

## Source
From `PR_REVIEW_01.md` → `## High-impact refactor opportunities (out of scope)`:

> Centralize log-domain validation across runtime and simplification

## Problem
`log` domain validation and `log` algebraic simplification are implemented as separate pieces of logic. Runtime evaluation validates zero values in `invoke_with_args`, but simplification can independently rewrite expressions such as:

- `log(a, a) => 1`
- `log(a, power(a, b)) => b`
- `log(a, 1) => 0`

Those identities are only valid inside the function domain. If simplification removes the `log` call before runtime evaluation, invalid inputs may never reach the validation path.

The result is optimizer-dependent behavior: the same logical expression can return a constant after simplification even though direct runtime evaluation would reject it.

## Concrete Failure Modes
Examples that should guide the refactor:

```sql
-- Should not simplify to 1 and bypass validation
SELECT log(0, 0);
```

```sql
-- Should not simplify to 2 when the value expression can evaluate to zero
SELECT log(column1, power(column1, 2))
FROM (VALUES (0), (2)) AS t(column1);
```

```sql
-- Should not simplify to 1 when the value/base column can be zero
SELECT log(column1, column1)
FROM (VALUES (0), (2)) AS t(column1);
```

A related edge case exists even when the base is a valid literal, because `power(base, expr)` can underflow to zero at runtime:

```sql
SELECT log(10.0, power(10.0, column1))
FROM (VALUES (-400.0), (2.0)) AS t(column1);
```

If this is simplified to `column1`, the first row returns `-400.0`; without the rewrite, `power(10.0, -400.0)` evaluates to `0.0`, and `log(10.0, 0.0)` should raise `cannot take logarithm of zero`.

## Why This Matters
- **Correctness:** Optimizer rewrites must preserve runtime semantics, including errors.
- **Predictability:** SQL users should not see different results depending on whether an expression was folded.
- **Maintainability:** Duplicated domain logic between runtime and simplification is easy to drift.
- **Future safety:** New `log` rewrites can accidentally reintroduce the same class of bug unless the domain contract is explicit.

## Root Cause
The current design treats algebraic identities as local pattern rewrites, but the validity of those identities depends on runtime domain conditions:

- the log value must not be zero under the current behavior;
- the base must be valid for the algebraic identity being applied;
- expression values that are not provably non-zero must not be eliminated if doing so skips validation.

Runtime has access to actual row values. Simplification usually does not. That means simplification needs a conservative proof model, not ad hoc literal checks.

## Refactor Goals
1. Define one clear domain model for `log`.
2. Reuse that model in simplification decisions where possible.
3. Make simplification conservative when validity is unknown.
4. Preserve existing valid simplifications when they are provably semantics-preserving.
5. Add targeted tests proving optimizer and runtime behavior stay aligned.

## Non-Goals
- Do not redesign all math function validation in one PR.
- Do not change documented `ln`, `log10`, or other math function behavior unless directly required.
- Do not add broad optimizer infrastructure unless the `log`-local refactor proves insufficient.
- Do not remove runtime validation; simplification checks are only a guard against invalid rewrites.

## Proposed Design
Introduce explicit helper logic in the `log` module to classify whether a simplification is safe.

Possible shape:

```rust
enum DomainProof {
    Valid,
    Invalid,
    Unknown,
}
```

Potential helpers:

```rust
fn prove_log_base_domain(expr: &Expr, data_type: &DataType) -> Result<DomainProof>
fn prove_log_value_non_zero(expr: &Expr, data_type: &DataType) -> Result<DomainProof>
fn can_eliminate_log_call(base: &Expr, value: &Expr, ...) -> Result<bool>
```

The exact API can vary, but it should make these rules obvious:

- `Invalid` literal cases must not be rewritten into successful constants.
- `Unknown` expression cases should preserve the `log` call when eliminating it would skip runtime value validation.
- Only `Valid` / provably safe cases may use identities that remove the `log` call.

## Rewrite Policy
Recommended conservative policy:

### `log(base, 1) => 0`
Only apply if the base is provably valid and the value literal `1` is valid under current `log` semantics.

### `log(base, base) => 1`
Only apply if the value/base expression is provably non-zero and the base is valid. For non-literal expressions, preserve the `log` call unless there is a reliable non-zero proof.

### `log(base, power(base, exponent)) => exponent`
Only apply if both are true:

1. the base is provably valid; and
2. the value expression `power(base, exponent)` is provably non-zero for all rows.

A valid literal base alone is not sufficient, because `power(valid_base, exponent)` can still evaluate to zero via underflow.

## Testing Plan
Add coverage in both simplification unit tests and SQL-level regression tests.

### Unit tests
Extend simplification tests to cover:

- literal invalid cases are not folded to constants;
- column cases are not folded when the value may be zero;
- valid literal identities still fold where safe;
- `power(valid_literal_base, column_exponent)` is not eliminated unless non-zero can be proven.

### SQLLogicTests
Add or keep SQL-visible tests for:

```sql
SELECT log(0);
SELECT log(2, 0);
SELECT log(0, 0);
```

```sql
SELECT log(column1, column1)
FROM (VALUES (0.0), (2.0)) AS t(column1);
```

```sql
SELECT log(column1, power(column1, 2))
FROM (VALUES (0.0), (2.0)) AS t(column1);
```

```sql
SELECT log(10.0, power(10.0, column1))
FROM (VALUES (-400.0), (2.0)) AS t(column1);
```

Expected behavior: queries whose runtime value argument evaluates to zero should raise `cannot take logarithm of zero`; simplification must not turn them into successful constants or column projections.

## Acceptance Criteria
- All `log` simplification rules that eliminate the `log` call are gated by explicit domain-safety helpers.
- `log(0, 0)` does not simplify to `1`.
- `log(col, col)` does not bypass runtime validation for rows where `col = 0`.
- `log(col, power(col, b))` does not bypass runtime validation for rows where the computed value is zero.
- `log(valid_literal_base, power(valid_literal_base, expr))` does not bypass runtime validation when the computed value may underflow to zero.
- Existing safe simplifications remain covered by tests.
- Targeted DataFusion function tests and relevant SQLLogicTests pass.

## Suggested Implementation Steps
1. Inventory existing `log` simplification rules in `datafusion/functions/src/math/log.rs`.
2. Add small helper predicates / proof enum for base validity and value non-zero status.
3. Update each rewrite to call the helper before returning `ExprSimplifyResult::Simplified`.
4. Keep `ExprSimplifyResult::Original` for unknown or unsafe cases.
5. Add unit tests for simplifier behavior.
6. Add SQLLogicTests for end-to-end runtime parity.
7. Run targeted tests:

```bash
cargo test -p datafusion-functions log
cargo test -p datafusion --test expr_api simplification
cargo test -p datafusion-sqllogictest --test sqllogictests math
```

Adjust exact test commands to match current workspace test names.

## Risks
- **Reduced optimization:** Some previously folded expressions may remain as runtime calls. This is acceptable if the rewrite was not provably semantics-preserving.
- **Decimal and cast edge cases:** Literal proof helpers must respect DataFusion scalar casting behavior.
- **NaN / infinity behavior:** Helpers must not accidentally change existing behavior for invalid bases unless that behavior is intentionally updated and tested.

## Suggested Labels
- refactor
- correctness
- optimizer
- functions
- sql-semantics
- tests
