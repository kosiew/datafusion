created #22986
source: pr-22784_a
# Unify list-like row decomposition for map construction

## Summary

`datafusion/functions-nested/src/map.rs` has parallel code paths for decomposing list-like key/value inputs when building `MapArray` values. The `List`/`LargeList` path and the `FixedSizeList` path both perform the same conceptual steps: preserve input row count and null-map bitmap, decompose each non-null row into key/value arrays, skip value rows corresponding to null key maps, then call the shared `build_map_array` helper.

This duplication is a maintainability risk. A small shared abstraction for “rows of list-like arrays” would make map construction easier to reason about and reduce the chance that future fixes to null-map handling, offset accounting, or length validation are applied to one list representation but missed in another.

## Current code involved

Relevant functions in `datafusion/functions-nested/src/map.rs`:

- `make_map_array_internal<O: OffsetSizeTrait>` for `List` and `LargeList`
- `make_map_array_from_fixed_size_list` for `FixedSizeList`
- `list_to_arrays_skipping_null_rows<O: OffsetSizeTrait>`
- `fixed_size_list_to_arrays_skipping_null_rows`
- `build_map_array`

## Problem

The `List`/`LargeList` and `FixedSizeList` construction paths have nearly identical control flow:

1. Save the original key/value data types.
2. Save the original input row count.
3. Capture the key-array null bitmap, which represents null map rows.
4. Decompose key rows into `Vec<ArrayRef>`.
5. Decompose value rows while skipping rows where the corresponding key map is null.
6. Delegate to `build_map_array`.

The row-skipping helpers are also duplicated, differing mainly in how they iterate list rows (`as_list::<O>()` vs `as_fixed_size_list()`).

Because these paths encode the same invariants separately, future changes can drift. Areas at risk include:

- preserving null map rows rather than treating them as null keys
- keeping map offsets aligned with original input rows
- handling all-null maps with correct element data types
- validating matching key/value row counts consistently
- maintaining consistent behavior between `List`, `LargeList`, and `FixedSizeList`

## Proposed direction

Introduce a small shared abstraction for list-like row decomposition used by map construction.

Possible shapes:

### Option A: helper taking row iterators

Create one helper that accepts already-decomposed key rows and a value-row iterator/filter strategy, then calls `build_map_array`.

### Option B: enum-backed adapter

Add an internal enum such as `ListLikeArray<'a>` with variants for:

- `List32(&'a ArrayRef)`
- `List64(&'a ArrayRef)`
- `FixedSizeList(&'a ArrayRef)`

Expose methods like:

- `data_type()`
- `len()`
- `nulls()`
- `rows()` / `non_null_rows_skipping(nulls)`

### Option C: small trait/internal helper functions

Use a private trait or generic helper to hide the row iteration difference while keeping the current dispatch from `DataType`.

Any option should keep the public API unchanged and avoid broad architectural changes.

## Suggested implementation constraints

- Keep this refactor local to `datafusion/functions-nested/src/map.rs` unless a clearly reusable utility already belongs elsewhere.
- Preserve current error messages where practical.
- Preserve `List`, `LargeList`, and `FixedSizeList` behavior exactly.
- Do not change SQL-visible semantics.
- Keep `build_map_array` as the single place that constructs offsets, flattened arrays, struct data, and the final `MapArray`, unless the refactor makes an equally clear replacement.

## Test coverage to preserve/add

Existing unit tests around map construction should continue to pass, especially coverage for:

- null map rows
- null key rejection within a map
- `LargeList` input
- `FixedSizeList` input
- mixed scalar/array map arguments

Useful additional targeted tests if gaps remain:

1. `List` and `FixedSizeList` inputs with the same null-row pattern produce equivalent map nulls and offsets.
2. All map rows null still produce an empty child array with the expected key/value element types.
3. Mismatched key/value row counts fail consistently across `List`, `LargeList`, and `FixedSizeList`.

## Acceptance criteria

- The duplicate row-skipping logic for `List`/`LargeList` and `FixedSizeList` is unified or reduced to a thin adapter layer.
- Map construction invariants are documented in one place near the shared helper.
- Existing map unit tests pass.
- Any newly added tests pass with `cargo test -p datafusion-functions-nested --lib map::tests`.
- No public API or SQL behavior changes.

## Scope

This is a refactor only. It is not intended to add new map semantics or change validation behavior.
