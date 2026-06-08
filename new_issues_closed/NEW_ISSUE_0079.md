source: pr-22714_a
# Centralize decimal AVG intermediate-state type selection

## Summary

Decimal `AVG` currently has intermediate-state type selection spread across multiple places. Recent overflow work widened some Decimal32 and Decimal64 AVG paths, but the mapping from input decimal type to sum-state decimal type is encoded manually in several branches and not shared with distinct AVG.

Refactor decimal AVG so all paths use one internal source of truth for the intermediate sum type and state field type.

## Motivation

`AVG(decimal)` has a core correctness invariant:

> The aggregate state must preserve the exact running sum needed to compute the average, or return an explicit overflow error. It must not silently wrap or let scalar/grouped/distinct execution paths diverge.

Today this invariant is hard to audit because decimal AVG behavior is split across:

- scalar accumulator construction in `datafusion/functions-aggregate/src/average.rs`
- grouped accumulator construction in `datafusion/functions-aggregate/src/average.rs`
- `Avg::state_fields` in `datafusion/functions-aggregate/src/average.rs`
- `GroupsAccumulator::convert_to_state` / `merge_batch` behavior
- distinct decimal AVG in `datafusion/functions-aggregate-common/src/aggregate/avg_distinct/decimal.rs`

When these paths choose state types independently, it is easy to fix one path and leave another with narrower/wrapping accumulation.

## Current problems

Examples of duplicated or divergent logic:

- `avg_sum_data_type` maps Decimal32 -> Decimal64 and Decimal64 -> Decimal128 for state fields.
- `Avg::accumulator` separately instantiates `DecimalAvgAccumulator<Decimal32Type, Decimal64Type>` and `DecimalAvgAccumulator<Decimal64Type, Decimal128Type>`.
- `Avg::create_groups_accumulator` separately instantiates wider grouped accumulator generics.
- `DecimalDistinctAvgAccumulator` still lives in the aggregate-common crate and uses its own same-type sum behavior.
- Decimal128/Decimal256 behavior is implicit rather than documented as part of a shared mapping.

This structure makes correctness review harder and increases the chance of regressions when new decimal sizes, state serialization paths, or aggregate modes are changed.

## Proposed change

Introduce an internal decimal AVG state-type abstraction. Possible shapes:

### Option A: internal trait

```rust
trait DecimalAvgStateType: DecimalType + ArrowNumericType {
    type Sum: DecimalType + ArrowNumericType;

    fn sum_data_type(input_precision: u8, input_scale: i8) -> DataType;
}
```

Example intended mapping:

- Decimal32 -> Decimal64 sum state
- Decimal64 -> Decimal128 sum state
- Decimal128 -> Decimal256 sum state if required for correctness
- Decimal256 -> Decimal256 with checked overflow / documented max behavior

### Option B: helper enum / match function

Centralize the runtime mapping in one helper used by:

- `Avg::state_fields`
- scalar accumulator construction
- grouped accumulator construction
- `convert_to_state`
- distinct AVG accumulator construction

The trait approach is likely cleaner for existing generic accumulator code, but either is acceptable if there is exactly one authoritative mapping.

## Acceptance criteria

- Decimal AVG state type mapping is defined in one internal location.
- Scalar AVG and grouped AVG use the same mapping.
- `Avg::state_fields` matches the actual accumulator state type for every decimal input type.
- `convert_to_state` emits the same sum type declared by `state_fields`.
- Decimal `AVG(DISTINCT)` either uses the same mapping or has a clearly documented, tested reason not to.
- Decimal32, Decimal64, Decimal128, and Decimal256 behavior is explicitly covered by unit tests.
- SQL-visible overflow/fits cases are covered by SLT where practical.

## Non-goals

- Do not change public SQL return types unless required by existing AVG semantics.
- Do not redesign all decimal aggregates in one patch.
- Do not expose a new public API unless needed.
- Do not hide overflow by returning approximate/floating results.

## Suggested tests

Add tests covering all execution paths where possible:

1. scalar/non-grouped decimal AVG
2. grouped decimal AVG
3. partial + final merge state roundtrip
4. `AVG(DISTINCT decimal)`
5. Decimal32 input with sum exceeding i32 but average fitting
6. Decimal64 input with sum exceeding i64 but average fitting
7. Decimal128 input with sum exceeding i128 but average fitting, if Decimal128 is widened
8. Decimal256 overflow behavior, explicit error if no wider exact state exists

Suggested commands:

```bash
cargo test -p datafusion-functions-aggregate avg
cargo test -p datafusion-sqllogictest --test sqllogictests decimal
```

## Implementation notes

- Keep the mapping close to `average.rs` unless distinct AVG needs it from `datafusion-functions-aggregate-common`.
- If the mapping must be shared with aggregate-common, prefer a small focused helper module rather than duplicating logic.
- Be careful with state schema compatibility: partial and final aggregate states must agree exactly on count and sum field types.
- Avoid broad generic abstractions that make the simple Decimal32/64/128/256 mapping hard to read.
