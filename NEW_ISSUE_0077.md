source: pr-22755_a
# Generalize or clearly gate sliding `SUM(DISTINCT)` type support

## Summary

`SUM(DISTINCT)` over sliding/bounded window frames is routed through the normal `SUM` type-dispatch path, but the sliding distinct accumulator only supports `Int64`. This can produce surprising runtime failures for other `SUM` input/return types that DataFusion otherwise supports, such as unsigned integers, floats, decimals, and durations.

The implementation should either:

1. generalize the sliding distinct sum accumulator across all supported `SUM` return types, or
2. reject unsupported sliding `SUM(DISTINCT)` types earlier with a clear planning-time capability error.

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
        // ...
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

This means the UDAF-level type contract and the sliding accumulator implementation contract are not aligned.

## Problem

From the SQL/user perspective, `SUM` supports more than `BIGINT`/`Int64`. For non-window aggregates and non-distinct sliding sums, supported numeric types work through the normal `SUM` path.

But for sliding `SUM(DISTINCT)`, type dispatch first appears to accept all supported `SUM` types, then fails only when constructing the accumulator. This is surprising and can surface as an execution/runtime error rather than as an explicit unsupported operation during planning.

Example cases to investigate:

```sql
-- unsigned-like values after coercion to UInt64, if reachable from SQL/types
SELECT SUM(DISTINCT u) OVER (
  ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
) FROM t;

-- float
SELECT SUM(DISTINCT f) OVER (
  ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
) FROM t;

-- decimal
SELECT SUM(DISTINCT d) OVER (
  ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
) FROM t;

-- duration/interval-like types if supported by SQL coercion
SELECT SUM(DISTINCT duration_col) OVER (
  ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
) FROM t;
```

## Desired behavior

Pick one explicit direction.

### Option A: Generalize support

Implement sliding distinct sum for each `SUM` return type accepted by `downcast_sum!`.

Expected behavior:

- Sliding `SUM(DISTINCT)` works for the same type families as regular `SUM` where semantics are well-defined.
- NULL handling remains correct:
  - NULL input slots are ignored.
  - Frames with no non-null values return NULL.
- DISTINCT semantics remain based on value equality for the relevant type.
- Sum arithmetic preserves existing wrapping/overflow behavior used by the corresponding `SUM` implementation.

Possible implementation direction:

- Make `SlidingDistinctSumAccumulator` generic over Arrow primitive types, similar to `SlidingSumAccumulator<T>`.
- Store counts keyed by the native value where legal.
- For types that cannot safely/idiomatically be hash keys, introduce a deliberate representation or choose Option B for those types.
- Keep a non-null distinct-value count or use `counts.is_empty()` to drive NULL-on-empty-frame behavior.

### Option B: Gate unsupported types earlier

If supporting every `SUM` type is not desired now, reject unsupported sliding `SUM(DISTINCT)` types at the correct abstraction layer with a clear message.

Expected behavior:

- Unsupported types fail before execution when possible.
- Error message names the unsupported type and operation, for example:

```text
SUM(DISTINCT) over sliding window frames is only supported for Int64, got Decimal128(20, 2)
```

- The type dispatch should not imply support for types that will be rejected immediately by the accumulator.

Possible implementation direction:

- Avoid routing distinct sliding accumulators through the broad `downcast_sum!` helper unless all those types are actually supported.
- Add a dedicated dispatch helper for sliding distinct sum.
- Keep the unsupported-type check close to the planner/physical expression layer if possible, so users get earlier feedback.

## Why this matters

This is a contract mismatch:

- `SUM` advertises/coerces multiple numeric return types.
- `create_sliding_accumulator` dispatches through the broad supported-type path.
- The concrete sliding distinct accumulator only implements `Int64`.

The mismatch can cause confusing user-visible failures and makes future maintenance harder because support appears broader at one layer than it is at another.

## Suggested tests

Add SQLLogicTests under `datafusion/sqllogictest/test_files/window.slt` or a focused existing window test file.

If generalizing support:

- `SUM(DISTINCT)` sliding frame over `BIGINT` with NULLs and duplicates.
- `SUM(DISTINCT)` sliding frame over `DOUBLE`/`FLOAT` if supported and hash/equality semantics are acceptable.
- `SUM(DISTINCT)` sliding frame over `DECIMAL`, including duplicate decimal values and all-NULL frames.
- Any unsigned/duration type coverage that can be expressed through SQL or a Rust unit test.
- Frames that transition from non-empty to all-NULL/empty after retraction.

If gating support:

- SQL test or unit test asserting unsupported types return the intended clear error.
- Keep existing `BIGINT` success coverage.

## Acceptance criteria

- The supported type set for sliding `SUM(DISTINCT)` is explicit and enforced at the right layer.
- No supported type fails only because `SlidingDistinctSumAccumulator::try_new` rejects it after broad dispatch.
- NULL/empty-frame behavior remains correct.
- Regression coverage exists for both successful supported types and unsupported-type errors, depending on chosen direction.

## Notes

This was observed while reviewing PR apache/datafusion#22755. The PR fixed NULL handling for the existing `Int64` sliding distinct sum path; this issue is broader and pre-existing, so it should be handled separately.
