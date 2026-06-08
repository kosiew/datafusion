source: pr-23226_a
# Centralize Hive partition path escaping and decoding

## Problem
Listing-table Hive partition handling has two separate representations with no shared boundary:

- object-store path segment form, e.g. `category=Electronics%2FComputers`
- SQL-visible partition value form, e.g. `Electronics/Computers`

The parser path (`parse_partitions_for_path` in `datafusion/catalog-listing/src/helpers.rs`) decodes values from object-store paths. The prefix-pruning path (`evaluate_partition_prefix` in the same file) independently builds object-store prefixes from SQL literals with `format!("{p}={val}")`.

This allows the two paths to disagree. A query filter such as `category = 'Electronics/Computers'` should match files stored under `category=Electronics%2FComputers/`, but prefix pruning can construct `category=Electronics/Computers/` and skip the file before decoded partition filtering runs.

## Why it matters
This is a correctness boundary between SQL-visible partition values and object-store paths. If encode/decode rules diverge, DataFusion can silently omit matching files for partition filters. The risk grows as more Hive escaping cases are supported because each caller must remember which representation it owns.

Centralizing this boundary also makes the code easier to reason about: listing code should not hand-build encoded Hive partition path fragments in one place and decode them elsewhere with separate rules.

## Invariant / desired behavior
For Hive-style listing tables, partition value handling should have one canonical contract:

- object-store paths use encoded path segments
- DataFusion partition values exposed to SQL are decoded strings
- any conversion between these forms goes through one shared helper/API
- prefix pruning must be representation-equivalent to parsing: a SQL literal that equals a decoded partition value must list the object-store path segment that would parse back to that value

Roundtrip rule:

```text
SQL-visible value -> encoded path segment -> parsed partition value == original SQL-visible value
```

for supported UTF-8 partition values.

## Proposed direction
Create a small Hive partition value/path helper near the listing-table partition code. Keep it narrow and explicit, for example:

- parse a path segment `name=value` into `(name, decoded_value)`
- encode a SQL-visible partition value into the path-segment value used by listing prefixes
- optionally build a full partition segment `name=encoded_value`

Then update both callers to use it:

- `parse_partitions_for_path` uses the shared decode helper
- `evaluate_partition_prefix` uses the shared encode helper before constructing `Path` prefixes

Avoid adding defensive fallback checks downstream in `filter_partitioned_file`; files skipped by prefix listing never reach that code. The invariant belongs at the conversion boundary between SQL-visible values and object-store path segments.

## Scope
### In
- Add a focused helper/API for Hive partition path segment encoding and decoding.
- Update `parse_partitions_for_path` to decode through the helper.
- Update `evaluate_partition_prefix` to encode SQL literal partition values before building listing prefixes.
- Preserve existing behavior for unescaped plain values.
- Preserve invalid UTF-8 fallback behavior if that remains the intended contract.
- Add regression tests for filtered listing/prefix construction with escaped values such as `/`, space, and non-ASCII UTF-8.

### Out
- Redesigning non-Hive partition discovery.
- Changing object-store `Path` semantics globally.
- Supporting arbitrary non-UTF-8 SQL-visible partition values.
- Changing partition column type coercion rules.
- Reworking table-provider filter pushdown beyond Hive partition prefix construction.

## Acceptance criteria
- [ ] There is one shared encode/decode boundary for Hive partition path values used by both parsing and prefix construction.
- [ ] `category = 'Electronics/Computers'` can match files under `category=Electronics%2FComputers/` when prefix pruning is active.
- [ ] Plain unescaped values still use the same paths and parse to the same `ScalarValue`s as before.
- [ ] Encoded spaces and UTF-8 values roundtrip through prefix construction and parsing.
- [ ] Tests cover both direct parsing and prefix-pruned listing/filter behavior.

## Tests / verification
- Unit tests for the helper:
  - `"v1"` stays `"v1"`
  - `"v/1"` encodes to a path-safe segment and decodes back to `"v/1"`
  - `"John Doe"` roundtrips
  - `"é"` roundtrips
- Unit tests for `evaluate_partition_prefix` showing SQL-visible literals are encoded before becoming `Path` prefixes.
- Regression test at listing/pruning level using an `ObjectStore` file under an encoded partition directory and a filter on the decoded SQL-visible value.
- Run `cargo test -p datafusion-catalog-listing`.

## Notes / open questions
- Confirm exact escaping policy for path-prefix construction: encode only bytes that are unsafe in object-store path segments, or preserve existing percent-encoded spellings when users created paths manually.
- Decide whether invalid UTF-8 percent sequences should remain a raw borrowed fallback only for parsing, or be documented as unsupported for SQL-visible values.
