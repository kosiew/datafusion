created #22987
source: pr-22610_a
# Centralize `date_bin` per-row mapping for scalar and array inputs

## Summary

`date_bin_impl` has separate scalar and array execution paths that each implement timestamp/time scaling, binning, and per-row error handling. This duplication has already allowed scalar and array behavior to diverge: a scalar negative sub-second timestamp with a month interval returned `NULL` / later the expected value, while the equivalent array path returned an execution error.

Refactor the implementation so scalar and array inputs share the same per-row mapping logic wherever possible.

## Motivation

The core invariant for `date_bin` is:

> For a given stride, source value, origin, and data type, scalar and array inputs should produce the same per-row result. Recoverable per-row binning errors should become `NULL` consistently in both paths, while invalid shared inputs such as unsupported stride/origin should remain query errors.

Today this invariant is enforced by duplicated code and tests rather than by structure. The array branch has helper logic for timestamp arrays, while scalar timestamp arms repeat the same pattern per timestamp unit. Time inputs also duplicate scale/bin/modulo behavior between scalar and array branches.

This makes future fixes risky: a change to overflow handling, modulo behavior, timezone preservation, or error-to-NULL semantics can easily be applied to one path and missed in the other.

## Current behavior / context

Recent work fixed issues around:

- negative sub-second timestamps before the epoch with month intervals
- scalar vs array error behavior for per-row binning failures
- checked scaling from second/millisecond/microsecond timestamp units to nanoseconds
- distinguishing per-row source errors from invalid shared origin errors

The code is now correct for the covered regressions, but `date_bin_impl` still contains separate scalar and array branches with repeated logic for:

- timestamp unit scale lookup
- checked scale-to-nanoseconds conversion
- applying the stride function
- converting binned nanoseconds back to the original time unit
- converting per-row failures to `NULL`
- preserving timestamp timezone metadata for array outputs
- time-of-day modulo behavior for `TIME` inputs

## Proposed refactor

Introduce a small internal abstraction that centralizes the per-row mapping rules. Possible shape:

1. A helper for timestamp values:

```rust
fn date_bin_timestamp_value<T: ArrowTimestampType>(
    value: i64,
    origin: i64,
    stride: i64,
    stride_fn: BinFunction,
) -> Option<i64> {
    let scale = timestamp_scale(T::UNIT);
    checked_scale_to_nanos(value, scale)
        .and_then(|scaled| stride_fn(stride, scaled, origin))
        .map(|binned| binned / scale)
        .ok()
}
```

Then use this helper in both scalar and array paths.

2. Similar helpers for `TIME` values, or a generic helper parameterized by scale and output conversion:

```rust
fn date_bin_time_value(
    value: i64,
    scale: i64,
    origin: i64,
    stride: i64,
    stride_fn: BinFunction,
) -> Option<i64> {
    checked_scale_to_nanos(value, scale)
        .and_then(|scaled| stride_fn(stride, scaled, origin))
        .map(|binned| (binned % NANOSECONDS_IN_DAY) / scale)
        .ok()
}
```

3. Keep shared-input validation outside the per-row mapper:

- stride type validation
- non-zero stride check
- TIME stride `< 1 day` validation
- origin scaling / origin overflow handling
- source/origin type compatibility errors

This preserves the intended distinction:

- bad shared inputs -> error
- bad per-row source value -> `NULL`

## Acceptance criteria

- Scalar and array timestamp inputs call the same helper for per-row timestamp binning.
- Scalar and array time inputs call the same helper or clearly equivalent shared helper for per-row time binning.
- Existing behavior is preserved for:
  - `Timestamp(Nanosecond|Microsecond|Millisecond|Second)`
  - timestamp timezones
  - `Time32(Second|Millisecond)`
  - `Time64(Microsecond|Nanosecond)`
  - `NULL` inputs
  - per-row overflow / out-of-range values -> `NULL`
  - invalid shared origin scaling -> query error
  - unsupported stride/origin/source types -> existing errors
- Existing regression tests continue to pass.
- Add or keep tests that explicitly compare scalar vs array results for at least:
  - negative sub-second timestamp with month interval
  - source scaling overflow
  - month interval out-of-range binning

## Suggested tests

Run:

```bash
cargo test -p datafusion-functions test_date_bin --lib
cargo test -p datafusion-sqllogictest --test sqllogictests -- date_bin_errors
```

Consider adding a compact unit test table that invokes `date_bin` once with a scalar source and once with a one-row array source, then asserts the same result/nullness for each regression case.

## Risk / notes

This should be an internal refactor only. No public API change intended.

Main risk is accidentally changing existing error boundaries. Keep the shared-input vs per-row-source distinction explicit in helper names and tests.
