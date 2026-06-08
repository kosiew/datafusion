created #22819
source: pr-22768_a
# Centralize `approx_distinct` grouped HLL dispatch

## Summary

`approx_distinct` currently has parallel type-dispatch lists for its grouped HyperLogLog fast path. The support predicate and grouped accumulator constructor must remain in sync, but they are encoded independently. This creates a maintenance hazard when future type additions or edits update one path but not the other.

## Context

Recent work added a specialized grouped `GroupsAccumulator` for `approx_distinct` in:

- `datafusion/functions-aggregate/src/approx_distinct.rs`

The implementation now has parallel grouped-dispatch logic in at least these places:

1. `AggregateUDFImpl::create_groups_accumulator`
   - Creates the specialized `HllGroupsAccumulator<...>` implementation for grouped aggregation.
   - Defines the concrete data types that can use the grouped fast path.

2. `is_hll_groups_type`
   - Used by `groups_accumulator_supported()` to advertise whether `create_groups_accumulator` can be called.
   - Must stay exactly aligned with `create_groups_accumulator`.

## Problem

The grouped HLL type set is duplicated between the support predicate and constructor. There is no known user-visible bug in the current code, but the duplicated matches create drift risk.

The predicate and constructor have a contract: if `groups_accumulator_supported()` returns `true`, `create_groups_accumulator()` should succeed for the same arguments. Keeping those paths separate makes that contract easier to break during future edits.

Adding support for a new grouped HLL type currently requires updating independent locations. Missing one update can advertise unsupported grouped execution or fail to use the grouped fast path for a type that has a constructor.

## Desired outcome

Centralize the grouped HLL input type dispatch for `approx_distinct` so there is one source of truth for:

- whether an input type is supported by the grouped HLL fast path
- which `HllValueHasher` implementation corresponds to that input type

Keep scope focused on `groups_accumulator_supported()` and `create_groups_accumulator()`. Do not try to unify scalar accumulator dispatch unless it falls out naturally without extra complexity.

## Suggested approach

One possible design:

1. Add a helper that attempts to build the grouped HLL accumulator directly:

```rust
fn create_hll_groups_accumulator(
    data_type: &DataType,
) -> Result<Option<Box<dyn GroupsAccumulator>>> {
    let acc: Box<dyn GroupsAccumulator> = match data_type {
        DataType::UInt32 => Box::new(HllGroupsAccumulator::<NumericHasher<UInt32Type>>::new()),
        // ... exact supported set ...
        _ => return Ok(None),
    };
    Ok(Some(acc))
}
```

2. Implement support detection from that same source of truth, for example by using a shared exact `matches!` helper or by matching on a small internal enum.

3. Update `groups_accumulator_supported()` so it mirrors creation exactly.

4. Update `create_groups_accumulator()` to call the centralized helper and return the existing `not_impl_err!` if `None`.

An alternative design would introduce a small internal enum such as `HllGroupsKind` with a `try_from_data_type` function. Both support detection and accumulator construction would use that enum.

## Testing

Add a focused unit test for the support/creation contract:

- Iterate representative supported and unsupported data types.
- For every type where `groups_accumulator_supported()` returns `true`, assert `create_groups_accumulator()` succeeds.
- Include unsupported time/unit combinations as regression guards if they can be constructed:
  - `Time32(Microsecond)`
  - `Time32(Nanosecond)`
  - `Time64(Second)`
  - `Time64(Millisecond)`

Also keep existing `approx_distinct` grouped tests passing.

Suggested command:

```bash
cargo test -p datafusion-functions-aggregate approx_distinct --lib
```

## Impact

Benefits:

- Prevents advertised grouped support from diverging from actual grouped constructor support.
- Makes future grouped `approx_distinct` type additions safer.
- Reduces repeated grouped match logic in a performance-sensitive aggregate implementation.

## Scope

This is a refactor/maintainability improvement. It should not change SQL results or supported public behavior.
