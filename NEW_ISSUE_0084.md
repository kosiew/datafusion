source: pr-22813_a
# Refactor Spark `round` numeric dispatch to share scalar/array helpers

## Summary

`datafusion/spark/src/function/math/round.rs` has separate scalar and array dispatch paths for Spark-compatible `round`. The actual rounding kernels are shared (`round_integer`, `round_float`, `round_decimal`), but type conversion, overflow handling, and result construction are still duplicated across many `ScalarValue` and `DataType` arms.

This makes future behavior fixes easy to apply to only one path, or to apply with slightly different overflow/error/null behavior between scalar and array execution.

## Current state

The function `spark_round` dispatches on `ColumnarValue`:

- `ColumnarValue::Array(array)` branches by `array.data_type()`.
- `ColumnarValue::Scalar(sv)` branches by `ScalarValue` variant.

The array side uses macros for many cases:

- `impl_integer_array_round!`
- `impl_float_array_round!`
- `impl_decimal_array_round!`

But scalar handling repeats equivalent per-type logic inline:

- integer conversion to `i64`
- call to `round_integer`
- ANSI vs non-ANSI overflow behavior
- conversion back to the original integer width
- construction of the matching `ScalarValue` variant
- decimal calls preserving precision and input scale
- float calls preserving the input float type

`UInt64` also has custom logic in both array and scalar paths for rejecting values above `i64::MAX`.

## Why this matters

Spark `round` has type-sensitive semantics:

- result type must match input type
- nulls must remain null
- arrays and scalars should produce the same values for the same logical input
- ANSI mode changes overflow behavior
- decimal results must preserve declared precision/scale
- `UInt64` has a special conversion boundary because the integer kernel operates through `i64`

Keeping these rules split across scalar and array branches increases regression risk. For example, a future fix to overflow handling, `UInt64`, or Spark rounding compatibility could update the array macro but miss the scalar branch, or vice versa.

## Proposed direction

Introduce small shared helpers that encode the common per-type contracts once, then call them from both scalar and array dispatch.

Possible shape:

1. Add helper functions for narrowing rounded integer results:

```rust
fn round_narrow_integer<T>(value: T, scale: i32, enable_ansi_mode: bool) -> Result<T>
where
    T: TryFrom<i64> + Into<i64> + Copy,
{
    let rounded = round_integer(value.into(), scale, enable_ansi_mode)?;
    if enable_ansi_mode {
        T::try_from(rounded).map_err(|_| ...)
    } else {
        Ok(rounded as T)
    }
}
```

The exact bounds may need adjustment for unsigned types and Arrow native types, but the goal is to centralize the round/narrow/overflow contract.

2. Keep a dedicated helper for `UInt64` because it cannot always be converted into `i64`:

```rust
fn round_u64(value: u64, scale: i32, enable_ansi_mode: bool) -> Result<u64> {
    let value = i64::try_from(value).map_err(|_| ...)?;
    round_integer(value, scale, enable_ansi_mode).map(|v| v as u64)
}
```

3. Add scalar constructors as thin wrappers, or use compact macros only for constructing the correct `ScalarValue` variant.

4. Keep array macros, but have them call the same helpers as scalar handling instead of duplicating conversion rules inside the macro.

## Acceptance criteria

- `spark_round` has one shared implementation for each core per-type contract:
  - signed integer round + narrow
  - unsigned integer round + narrow
  - `UInt64` special case
  - float round
  - decimal round preserving original decimal data type
- Scalar and array paths call the same helper for equivalent types.
- Existing behavior is preserved, including:
  - null handling
  - result type preservation
  - ANSI overflow errors
  - non-ANSI wrapping behavior
  - decimal precision/scale preservation
  - `UInt64 > i64::MAX` error behavior
  - Float16/Float32/Float64 Spark-compatible rounding
- Error messages remain actionable and include enough context to identify the value/type/scale where practical.
- No broad API changes outside the Spark `round` implementation unless clearly justified.

## Suggested tests

At minimum, keep existing SQLLogicTests and unit tests passing. Add focused regression coverage if any helper extraction changes behavior around boundaries:

- scalar and array parity for each family:
  - signed ints
  - unsigned ints
  - `UInt64`
  - floats
  - decimals
- ANSI overflow vs non-ANSI wrapping for narrow integer output types
- null scalar and nullable array behavior
- decimal precision/scale preservation after rounding

Good validation commands:

```bash
cargo test -p datafusion-spark round --quiet
# and the Spark round SQLLogicTest shard, if available in the local workflow
```

## Scope notes

This is a refactor only. It should not change Spark `round` semantics or broaden supported input types. Behavior changes should be split into separate issues/PRs with explicit SQLLogicTest coverage.
