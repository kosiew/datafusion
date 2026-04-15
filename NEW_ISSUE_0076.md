source: pr-22768_a
# Centralize `approx_distinct` HLL type dispatch

## Summary

`approx_distinct` currently has multiple parallel type-dispatch lists for HyperLogLog-backed input types. These lists must remain exactly in sync, but they are encoded independently in separate matches/predicates. This creates a maintenance hazard where support detection can claim a grouped accumulator exists for a type that `create_groups_accumulator` later rejects, or where future type additions update one path but not another.

## Context

Recent work added a specialized grouped `GroupsAccumulator` for `approx_distinct` in:

- `datafusion/functions-aggregate/src/approx_distinct.rs`

The implementation now has parallel dispatch logic in at least these places:

1. `AggregateUDFImpl::accumulator`
   - Creates the scalar/per-group `Accumulator` implementation.
   - Supports HLL-backed types such as `UInt32`, `UInt64`, `Int32`, `Int64`, dates/times/timestamps, strings, and binaries.

2. `AggregateUDFImpl::create_groups_accumulator`
   - Creates the specialized `HllGroupsAccumulator<...>` implementation for grouped aggregation.
   - Must support exactly the HLL-backed subset that can use the grouped fast path.

3. `is_hll_groups_type`
   - Used by `groups_accumulator_supported()` to advertise whether `create_groups_accumulator` can be called.
   - Currently broader than `create_groups_accumulator` for time types: it matches `Time32(_)` and `Time64(_)`, while creation only handles valid Arrow/DataFusion units:
     - `Time32(Second)`
     - `Time32(Millisecond)`
     - `Time64(Microsecond)`
     - `Time64(Nanosecond)`

## Problem

The supported HLL type set is duplicated. This can cause support/creation drift.

Concrete example:

- `is_hll_groups_type(DataType::Time32(TimeUnit::Microsecond))` returns `true` because it matches `DataType::Time32(_)`.
- `create_groups_accumulator` has no matching arm for `Time32(Microsecond)` and returns `not_impl_err!`.

Even if such invalid time/unit combinations are uncommon or normally blocked elsewhere, the predicate and constructor have a contract: if `groups_accumulator_supported()` returns `true`, `create_groups_accumulator()` should succeed for the same arguments. The current parallel implementation makes that contract easy to violate.

The same risk applies to future changes. Adding support for a new type requires updating several independent locations. Missing one update can silently route SQL-visible grouped aggregations to the wrong path, reject a type that was advertised as supported, or create inconsistent scalar vs grouped behavior.

## Desired outcome

Centralize the HLL-backed input type dispatch for `approx_distinct` so there is one source of truth for:

- whether an input type is supported by the grouped HLL fast path
- which `HllValueHasher` implementation corresponds to that input type
- which scalar HLL accumulator corresponds to that input type, if practical

At minimum, `groups_accumulator_supported()` and `create_groups_accumulator()` should be derived from the same helper so they cannot diverge.

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

A stronger design would introduce an internal enum such as `HllInputKind` or `HllGroupsKind` with a `try_from_data_type` function. Both support detection and accumulator construction would match on that enum.

## Testing

Add a focused unit test for the support/creation contract:

- Iterate representative data types.
- For every type where `groups_accumulator_supported()` returns `true`, assert `create_groups_accumulator()` succeeds.
- Include invalid or unsupported time/unit combinations if they can be constructed:
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

- Prevents advertised support from diverging from actual constructor support.
- Makes future `approx_distinct` type additions safer.
- Reduces repeated match logic in a performance-sensitive aggregate implementation.
- Keeps scalar and grouped HLL behavior easier to audit.

## Scope

This is a refactor/maintainability improvement. It should not change SQL results or supported public behavior, except that unsupported invalid type combinations should no longer be reported as grouped-accumulator-supported.
