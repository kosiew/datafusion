source: pr-22714_a
# Replace wrapping decimal accumulation with explicit checked or widened contracts

## Summary

Decimal AVG accumulation currently uses wrapping arithmetic in multiple state-update paths. Wrapping can silently corrupt aggregate state when the intermediate sum exceeds the native decimal storage type, even if the final average is representable.

Refactor decimal accumulation helpers so every decimal sum update either:

1. uses a state type that is proven wide enough for the intended operation, or
2. performs checked arithmetic and returns an explicit overflow error.

## Motivation

`AVG(decimal)` must not silently wrap its intermediate state. The final average can fit in the output type even when the intermediate sum does not fit in the input decimal native type.

Example class of bug:

- many large Decimal32 values
- sum exceeds `i32`
- average is still the original Decimal32 value and is representable
- wrapping sum corrupts the average or triggers a misleading later overflow

The same risk exists for larger decimal types and for state merge paths. If partial aggregate states are already corrupted, final aggregation cannot recover correctness.

## Current state

Touched code in `datafusion/functions-aggregate/src/average.rs` uses `add_wrapping` in several places, including:

- `decimal_sum_as`
- scalar decimal accumulator `update_batch`
- scalar decimal accumulator `merge_batch`
- grouped decimal accumulator `update_batch`
- grouped decimal accumulator `merge_batch`

Related distinct AVG code in `datafusion/functions-aggregate-common/src/aggregate/avg_distinct/decimal.rs` uses distinct sum logic that also relies on wrapping accumulation through `DistinctSumAccumulator`.

Wrapping arithmetic is fast, but it is not self-documenting. Without a named contract, future code readers cannot tell whether wrapping is safe because the state type is wide enough, or unsafe but tolerated accidentally.

## Proposed change

Introduce explicit decimal accumulation helpers with clear behavior. Possible API shapes:

```rust
fn add_decimal_sum_checked<T>(current: T::Native, value: T::Native) -> Result<T::Native>
where
    T: DecimalType + ArrowNumericType
```

and/or:

```rust
fn add_decimal_sum_widened<I, S>(current: S::Native, value: I::Native) -> S::Native
where
    I: DecimalType + ArrowNumericType,
    S: DecimalType + ArrowNumericType,
    I::Native: Into<S::Native>
```

The important part is naming and enforcing the contract:

- `*_checked` may fail with a clear DataFusion overflow error.
- `*_widened` is only used where the chosen `S` state type is intentionally wider than `I`.
- No raw `add_wrapping` remains in decimal AVG update/merge paths unless isolated behind a helper with a comment explaining why wrapping cannot corrupt correctness.

## Acceptance criteria

- Decimal AVG update and merge paths do not call `add_wrapping` directly.
- Any remaining wrapping arithmetic is hidden behind a helper whose name/comment states the safety contract.
- Scalar and grouped decimal AVG paths use the same accumulation helper or equivalent contract.
- Partial-state merge uses the same overflow behavior as direct batch update.
- Overflow behavior is deterministic in debug and release builds.
- Errors are actionable and consistent with existing DataFusion arithmetic overflow messages.
- Tests cover both direct update and partial-state merge overflow boundaries.

## Non-goals

- Do not replace all Arrow decimal arithmetic across the whole codebase in one change.
- Do not introduce approximate decimal aggregation.
- Do not make AVG slower for common Decimal32/64 cases unnecessarily; widened native accumulation should remain cheap.
- Do not silently saturate on overflow.

## Suggested tests

Add tests for each relevant accumulator path:

1. direct scalar accumulator update with representable average and overflowing narrow sum
2. grouped accumulator update with representable average and overflowing narrow sum
3. partial states merged into final AVG where combined state would overflow narrow type
4. negative values crossing the lower bound
5. null-heavy input where only non-null values count
6. all-null / empty input still returns NULL
7. explicit overflow when no wider exact state can represent the sum

Also add SQLLogicTest coverage for SQL-visible cases, for example:

```sql
select avg(d), arrow_typeof(avg(d))
from (
  select arrow_cast(99999, 'Decimal32(5, 0)') as d
  from generate_series(1, 21476)
) t;
```

and analogous Decimal64/Decimal128 cases.

## Implementation notes

- Use checked arithmetic where the state type is not guaranteed wide enough.
- If using widened state, ensure `state_fields`, `state()`, `convert_to_state`, and `merge_batch` all agree on the widened type.
- Avoid relying on debug overflow checks; release behavior must be correct too.
- Consider colocating helpers with decimal AVG implementation first. If SUM/DISTINCT code needs the same contract, move helpers to a small shared aggregate-common module.
- Preserve existing null semantics and count handling.

## Validation

Suggested commands:

```bash
cargo test -p datafusion-functions-aggregate avg
cargo test -p datafusion-functions-aggregate decimal
cargo test -p datafusion-sqllogictest --test sqllogictests decimal
```

If the refactor touches shared aggregate-common code, also run affected aggregate-common tests.
