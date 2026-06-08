created #23996
source: pr-23523_a
# Make RowsGroupColumn single-field extraction explicit

## Problem
`RowsGroupColumn::rows_to_array` uses a single-field `RowConverter`, then extracts the only decoded array with:

```rust
debug_assert_eq!(arrays.len(), 1, "single-field row converter");
let array = arrays.swap_remove(0);
```

The behavior is correct today, but the code does not express the invariant as clearly as it could. `swap_remove(0)` is a general vector operation, while this path expects exactly one decoded array.

## Why it matters
`rows_to_array` materializes emitted group keys for row-backed group columns. This code is small, internal, and invariant-heavy; it should make the single-field contract obvious to future maintainers.

Clearer extraction helps:
- Show that exactly one array is expected.
- Fail with a precise message if the single-field converter contract is ever broken.
- Avoid implying that element reordering is relevant.

## Invariant / desired behavior
`RowsGroupColumn::rows_to_array` must always receive exactly one decoded array because `RowsGroupColumn::try_new` builds the `RowConverter` with exactly one `SortField`.

Desired properties:
- Extraction encodes “exactly one element” directly.
- Any invariant failure reports the violated single-field contract.
- Output behavior remains identical.

## Proposed direction
Refactor only the extraction step after `convert_rows`.

Prefer an explicit single-element extraction, for example:

```rust
let [array] = arrays.try_into().unwrap_or_else(|arrays: Vec<_>| {
    panic!(
        "single-field row converter must produce exactly one array, got {}",
        arrays.len()
    )
});
```

A simpler `assert_eq!(arrays.len(), 1, ...)` plus `pop().unwrap()` is also acceptable if it reads better in context.

Keep this strictly behavior-preserving:
- No changes to row encoding or decoding.
- No changes to dictionary / nested type reconstruction.
- No changes to emitted values or schemas.

## Scope
### In
- `datafusion/physical-plan/src/aggregates/group_values/multi_group_by/row_backed.rs` cleanup in `rows_to_array`.
- Precise assertion / panic message for the single-field invariant.

### Out
- Broader `RowConverter` API changes.
- Grouping algorithm changes.
- Performance work outside this clarity refactor.

## Acceptance criteria
- [ ] `rows_to_array` no longer uses `swap_remove(0)` for single-element extraction.
- [ ] The single-field invariant is clear in code and failure messaging.
- [ ] Existing tests pass with no behavioral diffs.

## Tests / verification
- Run row-backed aggregation tests:
  - `cargo test -p datafusion-physical-plan row_backed --lib`
- No new test is required unless the implementation adds externally observable behavior.

## Notes / open questions
- Decide whether to use pattern-based extraction or `assert_eq!` plus `pop`; prefer whichever is clearest in the final code.
