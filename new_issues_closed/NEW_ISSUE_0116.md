created #23993
source: pr-23523_a
# Specialize GroupColumn storage for Dictionary and Run-End Encoded group keys

## Problem
DataFusion's column-wise aggregation path has type-specialized `GroupColumn` implementations for many native scalar group-key types. Dictionary and run-end encoded (REE) scalar group keys do not currently have equivalent specialized builders, so they are routed through row-format storage and then cast/re-encoded when emitted.

Today the effective fallback path for these encoded types is:

```text
arrow-array -> row format -> arrow-array -> cast/re-encode -> arrow-array
```

This path is correct, and must remain available for generic row-backed fallback cases. However, for top-level encoded scalar group keys it adds conversion work that native scalar group keys avoid.

Affected areas:
- `datafusion/physical-plan/src/aggregates/group_values/mod.rs`
- `datafusion/physical-plan/src/aggregates/group_values/multi_group_by/mod.rs`
- `datafusion/physical-plan/src/aggregates/group_values/multi_group_by/row_backed.rs`
- `datafusion/physical-plan/src/aggregates/group_values/row.rs`
- Existing dictionary benchmark coverage, including `dictionary_group_values`
- Possible related work in / around PR #23187

## Why it matters
Dictionary and REE encodings are commonly used to reduce memory and CPU for repeated values. Routing top-level encoded scalar group keys through row encoding and then casting/re-encoding preserves SQL semantics, but can reduce the benefit of those encodings in aggregation hot paths.

A specialized implementation could:
- Avoid row-format round trips for supported encoded scalar group keys.
- Avoid emit-time cast/re-encode work for common dictionary / REE cases.
- Keep `GroupValuesColumn` semantics clearer: supported encoded scalar keys have explicit builders; row-backed fallback remains for generic nested cases.
- Provide a focused place to optimize encoded-key equality, append, `take_n`, memory accounting, and null handling.

## Invariant / desired behavior
For supported dictionary and run-end encoded group-key columns:

- Equal logical values must receive the same group id.
- Null values must compare equal to null values and unequal to non-null values.
- Emitted group-key arrays must preserve the declared schema data type, including dictionary / REE wrappers, value type, nullability, and field metadata where applicable.
- Hashing and equality must agree for every value accepted by the specialized builder.
- `GroupValuesColumn::supported_schema` and `make_group_column` must stay in lockstep: every accepted type must have a valid `GroupColumn` implementation.
- Existing `GroupValuesRows` and row-backed nested fallback semantics must not change.

## Proposed direction
Add focused `GroupColumn` implementations for encoded scalar types. This is a behavior-preserving specialization / performance improvement, not a broad rewrite of the row fallback.

Suggested approach:

1. Start with top-level `DictionaryArray<K>` group keys.
   - Reuse or align with existing dictionary work from PR #23187 if available.
   - Store group values in a representation that preserves logical dictionary semantics without row conversion.
   - Make equality by logical value, not by dictionary key or dictionary identity.
   - Preserve null semantics and output data type on `build` / `take_n`.

2. Add `RunArray<R>` / REE support separately unless the dictionary implementation naturally exposes a small shared abstraction.
   - Treat runs as an encoding of logical values, not as group identity.
   - Ensure slicing and run boundaries do not affect grouping correctness.

3. Wire each implementation through `group_column_supported_type` and `make_group_column`.
   - Keep the existing biconditional tests updated.
   - Do not route unrelated unsupported scalar types through `GroupValuesColumn` as part of this work.

4. Keep `encode_array_if_necessary` for row-format fallback paths.
   - Do not remove schema re-encoding from `GroupValuesRows` / `RowsGroupColumn`; nested and generic row paths still need it.

## Scope
### In
- Add dictionary `GroupColumn` support for supported top-level scalar dictionary values.
- Add REE `GroupColumn` support only if it stays small and clearly shares the same design; otherwise split it into a follow-up.
- Preserve schema and logical equality semantics on `append_val`, `vectorized_append`, `equal_to`, `vectorized_equal_to`, `build`, and `take_n`.
- Update `group_column_supported_type` / `make_group_column` invariants and tests.
- Add focused unit tests and benchmark coverage for encoded group keys.

### Out
- Rewriting the generic row-backed nested fallback.
- Removing `encode_array_if_necessary` from row-format emit paths.
- Supporting unrelated unsupported scalar types such as `Float16` or `Decimal256`.
- Changing public SQL semantics for grouping dictionary or REE values.
- Large Arrow row-format changes; upstream Arrow improvements can be referenced but should not block the DataFusion-local specialization unless required.

## Acceptance criteria
- [ ] `GroupValuesColumn` can use a specialized `GroupColumn` for accepted top-level dictionary group-key columns.
- [ ] If REE is included, `GroupValuesColumn` can use a specialized `GroupColumn` for accepted top-level REE group-key columns.
- [ ] Emitted group-key arrays from `EmitTo::All` and `EmitTo::First(n)` have the declared dictionary / REE data type.
- [ ] Equal logical values with different dictionary keys or different dictionary arrays group together.
- [ ] Nulls, slices, repeated values, and multiple `intern` batches behave identically to `GroupValuesRows`.
- [ ] `group_column_supported_type_matches_make_group_column` or equivalent tests cover the newly supported encoded types.
- [ ] Benchmarks include at least one encoded-key aggregation scenario and show no broad regression.

## Tests / verification
- Unit tests for dictionary keys:
  - No nulls / with nulls.
  - Repeated values.
  - Sliced arrays.
  - Multiple dictionaries representing the same logical values.
  - `EmitTo::First(n)` followed by `EmitTo::All`.
  - Mixed schema: primitive column plus dictionary column.
- Unit tests for REE keys if implemented:
  - Runs crossing slice boundaries.
  - Null run values.
  - Duplicate logical values with different run layouts.
  - `take_n` preserving expected output type.
- Cross-check grouping against `GroupValuesRows` using canonicalized group ids where ordering is not part of the contract.
- Run targeted tests, for example:
  - `cargo test -p datafusion-physical-plan --lib aggregates::group_values::multi_group_by`
- Run relevant benchmarks:
  - Existing dictionary benchmarks.
  - At least one mixed primitive + dictionary group-by benchmark.
  - REE benchmark if REE support is included.

## Notes / open questions
- PR #23187 may already implement or prototype dictionary-specific group-column behavior; reuse it if it matches the invariants above.
- Prefer dictionary and REE as separate PRs unless the shared abstraction is obvious and small.
- Confirm exact Arrow behavior for dictionary equality and REE slicing before choosing storage representation.
- Keep reviewer context from PR #23523: the row-format-plus-cast path is acceptable for that PR but worth specializing as a follow-up.
