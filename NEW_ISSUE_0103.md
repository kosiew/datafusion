source: pr-23091_a
# Refactor: Centralize Container Child Adaptation and Null-Slot Handling

## Summary

Centralize recursive nested-struct adaptation for list-like containers in `datafusion/common/src/nested_struct.rs`, with an explicit shared contract for parent null-slot handling before child buffers are recursively cast.

Current code has separate runtime paths for `List`, `LargeList`, `ListView`, `LargeListView`, and `FixedSizeList`. Each path recursively casts the child/value array and then rebuilds the parent container with the original offsets/sizes/nulls. This works for many cases, but it leaves parent-null semantics implicit and easy to get wrong when adding new container support.

## Motivation

Nested schema evolution relies on planner/runtime parity:

- planning validation must allow exactly the casts runtime can perform
- runtime casting must produce arrays matching the target schema
- parent nulls must preserve Arrow semantics: child slots belonging only to null parent values are logically ignored

The `FixedSizeList` support work exposed that the current implementation can recursively cast physical child values before applying parent null semantics. For fixed-size lists, null parent entries still occupy `list_size` child slots. Those child values are not logically visible, but they can still cause a cast failure if the recursive cast sees them.

This makes the invariant too easy to violate in each container-specific helper.

## Current State

Relevant code paths:

- `datafusion/common/src/nested_struct.rs`
  - `cast_list_column`
  - `cast_list_view_column`
  - `cast_fixed_size_list_column`
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

This shape does not document or enforce how child values covered only by parent null slots should be handled before recursive casts.

## Problem

The container adaptation logic duplicates the same high-level algorithm across container types, but the most important semantic rule is not centralized:

> Parent null slots define logically absent values. Runtime adaptation must not fail because of child buffer contents that are reachable only through null parent entries.

Because the rule is implicit, new container implementations can pass basic null tests while still failing on valid Arrow arrays whose null parent entries contain uncastable physical child values.

## Proposed Direction

Introduce a shared internal abstraction for recursive container-child adaptation that makes null-slot semantics explicit.

Possible design:

1. Split container adaptation into common phases:
   - downcast/source extraction
   - identify logical child ranges visible from non-null parent slots
   - prepare/mask child values as needed before recursive cast
   - recursively cast child values
   - rebuild parent container with target child field and original structural buffers

2. Add an internal helper with a contract like:

```rust
fn cast_container_values_preserving_parent_nulls(
    values: &ArrayRef,
    target_inner_field: &FieldRef,
    parent_nulls: Option<&NullBuffer>,
    child_ranges: impl Iterator<Item = Range<usize>>,
    cast_options: &CastOptions,
) -> Result<ArrayRef>
```

Exact API can differ, but the helper should encode that only child positions visible from non-null parents can affect success/failure.

3. Make each container-specific function only responsible for container geometry:
   - `List` / `LargeList`: offsets define child ranges
   - `ListView` / `LargeListView`: offsets + sizes define child ranges
   - `FixedSizeList`: `row_index * list_size..(row_index + 1) * list_size`

4. Keep validation and routing separate:
   - `requires_nested_struct_cast` decides whether to use recursive runtime adaptation
   - `validate_data_type_compatibility` decides planning compatibility
   - container runtime helpers enforce the null-slot execution contract

## Acceptance Criteria

- Shared helper or abstraction exists for recursive container child adaptation.
- The helper explicitly documents parent-null semantics.
- `List`, `LargeList`, `ListView`, `LargeListView`, and `FixedSizeList` use the shared helper or share the same central null-slot policy.
- Runtime casting does not fail because of uncastable child values reachable only from null parent container entries.
- Planner/runtime parity tests cover at least:
  - nullable parent container with invalid hidden child values
  - all-null parent container
  - additive nullable nested struct field evolution
  - incompatible visible child values still rejected
  - fixed-size-list size mismatch still rejected

## Suggested Tests

Add tests in `datafusion/common/src/nested_struct.rs`.

### FixedSizeList null-parent hidden child values

Source:

- type: `FixedSizeList<Struct<a: Utf8>, 2>`
- parent null bitmap: one or more null parent rows
- child buffer under null rows contains values that cannot cast to target type, e.g. `"not_int"`

Target:

- type: `FixedSizeList<Struct<a: Int32>, 2>`

Expected:

- cast succeeds if all invalid values are hidden only by parent nulls
- resulting parent rows remain null

### Visible invalid child values still fail

Same source/target as above, but invalid child values are in a non-null parent row.

Expected:

- validation may allow the type-level cast if Arrow can cast the declared types
- runtime returns the cast error for visible invalid data

### Parity matrix across list-like containers

Repeat the null-parent hidden-child test for:

- `List`
- `LargeList`
- `ListView`
- `LargeListView`
- `FixedSizeList`

Expected:

- all list-like containers observe the same null-slot policy

## Risks / Considerations

- Masking child buffers must preserve array lengths expected by Arrow container constructors.
- Avoid copying child arrays when there are no parent nulls.
- Avoid changing semantics for visible child values: invalid visible casts must still fail.
- For variable-size lists, offsets may include child ranges for null parent entries; those ranges must be treated as logically ignored.
- `ListView` ranges may be non-contiguous or overlapping; helper design should account for offset/size geometry rather than assuming compact offsets.

## Non-Goals

- Do not change map-specific schema evolution semantics.
- Do not broaden nested adaptation to unrelated Arrow container types unless separately scoped.
- Do not weaken type compatibility validation to hide runtime errors for visible data.

## Implementation Notes

A low-risk implementation can start narrow:

1. Add tests that demonstrate null-parent hidden child values are ignored.
2. Add a helper for computing visible child positions/ranges per container.
3. For containers with parent nulls, construct a child array where hidden positions are replaced with nulls before recursive `cast_column`.
4. Skip the masking path when there are no parent nulls.
5. Keep container-specific rebuild code small and mechanical.

The refactor should make future additions safer by putting the parent-null contract in one place instead of relying on each new container implementation to rediscover it.
