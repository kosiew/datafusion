source: pr-23091_a
# Bugfix + Refactor: Centralize List-Like Nested Struct Casting Null Semantics

## Summary

Fix and centralize nested-struct casting for list-like arrays in `datafusion/common/src/nested_struct.rs`.

Current `List`, `LargeList`, `ListView`, and `LargeListView` casting recursively casts the entire child `values()` array before rebuilding the parent container with the original null buffer. This makes parent-null semantics implicit and can cause casts to fail on child values that are only reachable through null parent slots.

## Motivation

Nested schema evolution relies on planner/runtime parity:

- planning validation must allow exactly the casts runtime can perform
- runtime casting must produce arrays matching the target schema
- parent nulls must preserve Arrow semantics: child slots belonging only to null parent values are logically ignored

Today this invariant is not centralized. Each list-like helper owns its own recursive child-casting path, so the most important null-slot rule is easy to miss when extending or modifying container support.

## Current State

Relevant code paths:

- `datafusion/common/src/nested_struct.rs`
  - `cast_list_column`
  - `cast_list_view_column`
  - `validate_data_type_compatibility`
  - `requires_nested_struct_cast`

The runtime helpers currently follow this broad shape:

```rust
let source_container = downcast(...)?;
let cast_values = cast_column(
    source_container.values(),
    target_inner_field.data_type(),
    cast_options,
)?;
rebuild_container(target_inner_field, original_offsets_or_sizes, cast_values, original_nulls)
```

This casts the full child array before parent null semantics are applied to the rebuilt parent container.

## Problem

In Arrow, values covered only by null parent list entries are logically absent. Runtime nested-struct casting should not fail because those hidden child values cannot be cast.

Today, invalid hidden child values can still affect success or failure because `cast_column` sees the full child `values()` array.

The invariant should be explicit:

> Child values reachable only through null parent entries are logically absent and must not determine cast success.

Visible child values must keep the current behavior: incompatible visible values should still fail at runtime.

## Proposed Direction

Add a shared internal helper for list-like child casting that makes parent-null handling explicit.

The helper should:

- identify child positions reachable from valid parent entries
- ensure hidden child positions cannot cause recursive cast failures
- preserve original offsets/sizes/nulls when rebuilding the parent array
- avoid extra work when there are no parent nulls

`List`, `LargeList`, `ListView`, and `LargeListView` should use this shared policy. Container-specific functions should remain responsible only for geometry:

- `List` / `LargeList`: offsets define child ranges
- `ListView` / `LargeListView`: offsets + sizes define child ranges

A possible helper shape:

```rust
fn cast_list_like_values_preserving_parent_nulls(
    values: &ArrayRef,
    target_inner_field: &FieldRef,
    parent_nulls: Option<&NullBuffer>,
    visible_child_ranges: impl Iterator<Item = Range<usize>>,
    cast_options: &CastOptions,
) -> Result<ArrayRef>
```

Exact API can differ. The important part is that the helper documents and enforces the parent-null contract.

## Acceptance Criteria

- Shared helper or shared internal policy exists for list-like recursive child casting.
- The helper explicitly documents parent-null semantics.
- `List`, `LargeList`, `ListView`, and `LargeListView` use the shared policy.
- Runtime casting does not fail because of uncastable child values reachable only from null parent list entries.
- Incompatible visible child values still fail.
- Tests cover:
  - nullable parent list with invalid hidden child values
  - all-null parent list
  - additive nullable nested struct field evolution
  - incompatible visible child values still rejected
  - `List`, `LargeList`, `ListView`, and `LargeListView`

## Suggested Tests

Add tests in `datafusion/common/src/nested_struct.rs`.

### Null-parent hidden child values

Source:

- type: `List<Struct<a: Utf8>>` or equivalent list-like container
- parent null bitmap: one or more null parent rows
- child buffer under null rows contains values that cannot cast to target type, e.g. `"not_int"`

Target:

- type: `List<Struct<a: Int32>>` or equivalent list-like container

Expected:

- cast succeeds if all invalid values are hidden only by parent nulls
- resulting parent rows remain null

Repeat for:

- `List`
- `LargeList`
- `ListView`
- `LargeListView`

### Visible invalid child values still fail

Use the same source/target shape, but place invalid values in a non-null parent row.

Expected:

- validation may allow the type-level cast if Arrow can cast the declared types
- runtime returns the cast error for visible invalid data

## Risks / Considerations

- This is a behavior fix plus refactor, not a pure behavior-preserving refactor.
- Masking child buffers must preserve array lengths expected by Arrow container constructors.
- Avoid copying child arrays when there are no parent nulls.
- Avoid changing semantics for visible child values: invalid visible casts must still fail.
- For variable-size lists, offsets may include child ranges for null parent entries; those ranges must be treated as logically ignored.
- `ListView` ranges may be non-contiguous or overlapping. Hidden positions are child positions not referenced by any valid parent, not simply ranges referenced by invalid parents.
- If hidden child values are inside nested structs, marking only the parent struct row null may not be enough if recursive casting still casts the struct child arrays directly. The implementation must ensure hidden primitive leaf values cannot cause failures.

## Non-Goals

- Do not add `FixedSizeList` support in this issue.
- Do not change map-specific schema evolution semantics.
- Do not change dictionary semantics.
- Do not broaden nested adaptation to unrelated Arrow container types unless separately scoped.
- Do not weaken type compatibility validation to hide runtime errors for visible data.

## Implementation Notes

A low-risk implementation can start narrow:

1. Add failing tests that demonstrate null-parent hidden child values currently affect runtime casting.
2. Add a helper for computing visible child positions/ranges per list-like container.
3. For containers with parent nulls, ensure hidden child positions cannot cause recursive cast failures.
4. Skip the masking path when there are no parent nulls.
5. Keep container-specific rebuild code small and mechanical.

The resulting code should make future list-like nested-struct adaptation safer by putting the parent-null contract in one place instead of relying on each container implementation to rediscover it.
