created #22820
source: pr-22755_a
# Clearly gate sliding `SUM(DISTINCT)` type support

## Summary

`SUM(DISTINCT)` over sliding/bounded window frames is routed through the normal `SUM` type-dispatch path, but the current sliding distinct accumulator only supports `Int64`. This can produce confusing user-visible failures for other `SUM` return types that DataFusion otherwise supports, such as unsigned integers, floats, decimals, and durations.

This issue should make the supported type set explicit and reject unsupported sliding `SUM(DISTINCT)` types with a clear error at the earliest practical layer. Generalizing support to more types can be handled later as a separate feature.

## Context

Relevant code:

- `datafusion/functions-aggregate/src/sum.rs`
- `Sum::create_sliding_accumulator`
- `SlidingDistinctSumAccumulator::try_new`

Current shape:

```rust
fn create_sliding_accumulator(
    &self,
    args: AccumulatorArgs,
) -> Result<Box<dyn Accumulator>> {
    if args.is_distinct {
        macro_rules! helper_distinct {
            ($t:ty, $dt:expr) => {
                Ok(Box::new(SlidingDistinctSumAccumulator::try_new(&$dt)?))
            };
        }
        downcast_sum!(args, helper_distinct)
    } else {
        // non-distinct sliding sum path
    }
}
```

`downcast_sum!` accepts the supported `SUM` return types, including `UInt64`, `Float64`, decimals, durations, and `Int64`. However, the distinct sliding accumulator rejects everything except `Int64`:

```rust
pub fn try_new(data_type: &DataType) -> Result<Self> {
    // TODO support other numeric types
    if *data_type != DataType::Int64 {
        return exec_err!("SlidingDistinctSumAccumulator only supports Int64");
    }
    // ...
}
```

This means the dispatch contract and implementation contract are not aligned.

## Problem

From the SQL/user perspective, `SUM` supports more than `BIGINT`/`Int64`. For non-window aggregates and non-distinct sliding sums, supported numeric types work through the normal `SUM` path.

For sliding `SUM(DISTINCT)`, the broad dispatch path appears to accept all supported `SUM` return types, then fails only when constructing the accumulator. The error is implementation-oriented and does not clearly describe the unsupported operation.

Example cases to cover or reject clearly:

```sql
-- float
SELECT SUM(DISTINCT f) OVER (
  ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
) FROM t;

-- decimal
SELECT SUM(DISTINCT d) OVER (
  ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
) FROM t;

-- unsigned-like values after coercion to UInt64, if reachable from SQL/types
SELECT SUM(DISTINCT u) OVER (
  ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
) FROM t;

-- duration values, if supported by SQL coercion/test setup
SELECT SUM(DISTINCT duration_col) OVER (
  ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
) FROM t;
```

## Desired behavior

Gate unsupported sliding `SUM(DISTINCT)` types explicitly.

Expected behavior:

- `Int64` sliding `SUM(DISTINCT)` keeps working.
- Unsupported types fail before or during physical accumulator construction with a clear capability error.
- The error message names the operation and type, for example:

```text
SUM(DISTINCT) over sliding window frames is only supported for Int64, got Decimal128(20, 2)
```

- Type dispatch should not imply support for types that are immediately rejected by `SlidingDistinctSumAccumulator::try_new`.

## Suggested approach

- Avoid routing distinct sliding accumulators through the broad `downcast_sum!` helper unless all those types are actually supported.
- Add a small dedicated helper for sliding distinct sum support, currently accepting only `DataType::Int64`.
- Use that helper from `Sum::create_sliding_accumulator` for `args.is_distinct`.
- Keep the unsupported-type check close to `create_sliding_accumulator` or another physical-expression layer so the failure is tied to the window aggregate capability, not the accumulator internals.
- Leave full support for floats, decimals, unsigned integers, and durations to a future feature issue.

## Why this matters

This is a user-facing contract mismatch:

- `SUM` advertises/coerces multiple numeric return types.
- `create_sliding_accumulator` currently dispatches through the broad supported-type path.
- The concrete sliding distinct accumulator only implements `Int64`.

The mismatch can cause confusing failures and makes future maintenance harder because support appears broader at one layer than it is at another.

## Suggested tests

Add SQLLogicTests under `datafusion/sqllogictest/test_files/window.slt` or a focused existing window test file.

Coverage:

- `SUM(DISTINCT)` sliding frame over `BIGINT` still succeeds, including NULLs and duplicates.
- Unsupported type returns the intended clear error. Prefer one SQL-reachable type such as `DOUBLE` or `DECIMAL`.
- If useful, add a Rust unit test for unsupported types that are hard to express through SQL, such as `UInt64` or `Duration`.

Suggested command:

```bash
cargo test -p datafusion-functions-aggregate sum --lib
cargo test -p datafusion-sqllogictest --test sqllogictests -- window.slt
```

## Acceptance criteria

- The supported type set for sliding `SUM(DISTINCT)` is explicit.
- `Int64` behavior, including NULL/empty-frame behavior, remains correct.
- Unsupported types produce a clear operation-specific error.
- No unsupported type fails only because `SlidingDistinctSumAccumulator::try_new` rejects it after broad dispatch.
- Regression coverage exists for both the supported `Int64` path and at least one unsupported-type error.

## Scope

This is a scoped correctness/refactor issue. It should not generalize sliding `SUM(DISTINCT)` to new types. It should only make the current `Int64`-only support explicit and user-facing errors clearer.

## Notes

This was observed while reviewing PR apache/datafusion#22755. The PR fixed NULL handling for the existing `Int64` sliding distinct sum path; this issue is broader and pre-existing, so it should be handled separately.
