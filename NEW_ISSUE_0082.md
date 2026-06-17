pr: 22823
source: scaling-overflow-02-22685a
# Extract Common TIME Binning Helpers in DATE_BIN

## Summary

Refactor DATE_BIN TIME scalar and array paths to share common helper logic for binning TIME values. The current implementation repeats the same sequence across `Time32Second`, `Time32Millisecond`, `Time64Microsecond`, and `Time64Nanosecond` branches: validate TIME origin, scale source value to nanoseconds, call the binning function, modulo by one day, and scale back to the original TIME unit.

## Problem

`datafusion/functions/src/datetime/date_bin.rs` has duplicated TIME binning code in both scalar and array paths. Recent overflow work made the scaling step safer with `value_to_nanos`, but the broader TIME binning flow remains repeated.

Duplicated logic increases risk that future fixes to DATE_BIN TIME behavior are applied inconsistently across:
- scalar vs array inputs.
- `Time32Second` vs `Time32Millisecond`.
- `Time64Microsecond` vs `Time64Nanosecond`.

This is especially risky because DATE_BIN currently has nuanced error behavior:
- scalar TIME paths map `stride_fn` binning errors to `NULL`.
- array TIME paths map `stride_fn` binning errors to Arrow compute errors.
- scaling overflow should be a normal error, not a panic or wrap.

## Evidence

In `datafusion/functions/src/datetime/date_bin.rs`, the touched TIME branches repeat this pattern:

1. Check origin compatibility:

```rust
if !is_time {
    return exec_err!("DATE_BIN with Time64 source requires Time64 origin");
}
```

2. Convert source value to nanoseconds:

```rust
let nanos = value_to_nanos(*x, NANOS_PER_MICRO)?;
```

3. Bin using shared DATE_BIN logic:

```rust
match stride_fn(stride, nanos, origin) { ... }
```

4. Wrap within one day:

```rust
let nanos = binned_nanos % NANOSECONDS_IN_DAY;
```

5. Scale back to the output unit:

```rust
Some(nanos / NANOS_PER_MICRO)
```

Array paths repeat the same shape inside `try_unary`, with error conversion to `ArrowError::ComputeError`.

## Desired Behavior

DATE_BIN TIME behavior should remain unchanged, but the repeated scalar/array code should be centralized into helpers that make the invariants explicit.

The helpers should preserve:
- TIME origin compatibility errors.
- scalar binning-error-to-NULL behavior.
- array binning-error-to-Arrow-compute-error behavior.
- checked scaling through `value_to_nanos`.
- modulo-by-day behavior.
- return type/unit for each TIME variant.

## Scope

In scope:
- Add private helper(s) in `datafusion/functions/src/datetime/date_bin.rs` near the DATE_BIN implementation.
- Refactor TIME scalar branches to use a shared scalar helper.
- Refactor TIME array branches to use a shared array helper where practical.
- Add characterization tests if any behavior is not already covered.
- Keep timestamp paths unchanged unless a small helper naturally shares with them without increasing complexity.

Out of scope:
- Changing DATE_BIN public semantics.
- Changing timestamp binning behavior.
- Changing interval/month handling.
- Changing scalar vs array error/NULL policy unless separately discussed and tested.

## Suggested Implementation

### Scalar helper

A helper can take the optional source value, source scale, output conversion, and bin context:

```rust
fn bin_time_scalar(
    value: Option<i64>,
    scale: i64,
    stride: i64,
    origin: i64,
    stride_fn: BinFunction,
) -> Result<Option<i64>> {
    match value {
        Some(value) => {
            let nanos = value_to_nanos(value, scale)?;
            Ok(stride_fn(stride, nanos, origin)
                .ok()
                .map(|binned_nanos| (binned_nanos % NANOSECONDS_IN_DAY) / scale))
        }
        None => Ok(None),
    }
}
```

Callers can cast the returned value to `i32` for `Time32` outputs where needed. If casts need checked handling, keep that explicit at the call site.

### Array helper

A generic array helper can take an Arrow primitive array and output scale. It must preserve current array behavior by mapping both scaling and `stride_fn` errors to `ArrowError::ComputeError`.

Potential shape:

```rust
fn bin_time_array<T>(
    array: &PrimitiveArray<T>,
    scale: i64,
    stride: i64,
    origin: i64,
    stride_fn: BinFunction,
) -> Result<PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
    T::Native: ...
{
    // convert native value to i64, checked-scale to nanos,
    // call stride_fn, modulo day, divide by scale,
    // convert back to native output type
}
```

If the generic conversion becomes awkward, prefer two small helpers instead of a complex generic:
- one for `i32` TIME32 arrays.
- one for `i64` TIME64 arrays.

Do not over-generalize if it hides the scalar-vs-array error policy.

## Tests

Before refactoring, add or identify characterization coverage for:
- valid `Time32Second` DATE_BIN scalar and array.
- valid `Time32Millisecond` DATE_BIN scalar and array.
- valid `Time64Microsecond` DATE_BIN scalar and array.
- valid `Time64Nanosecond` DATE_BIN scalar and array.
- null scalar and null array element behavior.
- existing overflow tests for `Time64Microsecond(i64::MAX)` still pass.
- origin type mismatch errors still match existing behavior.

Existing DATE_BIN tests may cover some of this; only add missing high-value cases.

## Acceptance Criteria

1. TIME scalar binning logic is centralized in helper(s), not repeated across every TIME unit branch.
2. TIME array binning logic is centralized in helper(s), or consciously split into simple `Time32`/`Time64` helpers.
3. Existing DATE_BIN scalar vs array error behavior is preserved.
4. Existing DATE_BIN tests pass.
5. Overflow regression tests for TIME scaling still pass.
6. No public API changes.

## Validation

Run:

```bash
cargo test -p datafusion-functions date_bin
```

Optional grep/review checks:

```bash
grep -n "DATE_BIN with Time32 source requires Time32 origin\|DATE_BIN with Time64 source requires Time64 origin" datafusion/functions/src/datetime/date_bin.rs
```

Expected: repeated compatibility checks should be reduced or isolated to helper/call-site boundaries.

```bash
grep -n "NANOSECONDS_IN_DAY" datafusion/functions/src/datetime/date_bin.rs
```

Expected: TIME modulo-by-day logic should be centralized or appear only in clearly shared helper code.

## Risk

Medium. This is a refactor of nuanced DATE_BIN behavior. The main risk is accidentally changing how scalar and array paths handle `stride_fn` errors or nulls. Keep the PR narrow, add characterization tests first, and avoid broad semantic changes.
