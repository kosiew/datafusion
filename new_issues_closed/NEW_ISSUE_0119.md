created #23994
source: pr-23523_a
# Preallocate RowsGroupColumn buffers during take_n rebuilds

## Problem
`RowsGroupColumn::take_n` rebuilds the retained rows with zero-capacity `Rows`:

```rust
let mut remaining = self.row_converter.empty_rows(0, 0);
```

At that point the retained row count is known (`self.group_values.num_rows() - n`), and Arrow `Rows` also exposes row lengths. Rebuilding from zero capacity can force avoidable growth and copying while pushing the remaining rows.

`RowsGroupColumn::try_new` also initializes with `empty_rows(0, 0)`, but construction currently has no reliable row-count hint. Treat constructor preallocation as optional follow-up, not required scope.

## Why it matters
`take_n` runs on emit paths for row-backed group columns. In high-cardinality or repeated partial-emit workloads, rebuilding retained row buffers from zero capacity can add avoidable CPU and allocator pressure without changing semantics.

## Invariant / desired behavior
`RowsGroupColumn` should preserve exact grouping semantics while avoiding known reallocations in rebuild paths:

- No change to equality or group-id behavior.
- No change to emitted values, output type reconstruction, or emit ordering.
- `take_n` should allocate enough capacity up front when retained row count and byte size are known.

## Proposed direction
Refactor only the `take_n` rebuild path first.

1. Compute exact retained capacity before rebuilding:
   - `remaining_rows = self.group_values.num_rows() - n`
   - `remaining_bytes = self.group_values.lengths().skip(n).sum()`

2. Create the replacement buffer with those capacities:
   - `self.row_converter.empty_rows(remaining_rows, remaining_bytes)`

3. Push the same retained rows in the same order.

4. Leave constructor capacity hinting out unless a trustworthy local hint already exists.
   - Do not add speculative planner/batch-size plumbing just to pass an estimate.

## Scope
### In
- `datafusion/physical-plan/src/aggregates/group_values/multi_group_by/row_backed.rs` `take_n` preallocation.
- Focused test coverage proving unchanged outputs across `take_n`.
- Targeted perf/allocation comparison if practical.

### Out
- Rewriting `Rows` internals or Arrow row implementation.
- Changing grouping semantics, output schema behavior, dictionary re-encoding, or fallback selection logic.
- Adding broad constructor/planner capacity plumbing without an existing reliable hint.
- Broader aggregation architecture changes unrelated to `RowsGroupColumn::take_n`.

## Acceptance criteria
- [ ] `RowsGroupColumn::take_n` no longer rebuilds retained rows with zero-capacity `Rows`.
- [ ] Retained row and byte capacities are derived from existing `Rows` data, not guesses.
- [ ] Existing row-backed group-value tests pass unchanged.
- [ ] Added or updated tests exercise `take_n` and validate unchanged emitted arrays / remaining rows.
- [ ] If benchmarked, before/after results show no regression and expected allocation or throughput improvement on representative row-backed workloads.

## Tests / verification
- Run existing row-backed group-column tests:
  - `cargo test -p datafusion-physical-plan row_backed --lib`
- Add or update a focused unit test that:
  - Appends multiple row-backed group values.
  - Calls `take_n`.
  - Verifies emitted output and later `build` output are unchanged.
- Optional perf check:
  - Compare before/after on a row-backed aggregation case with high and medium cardinality inputs.

## Notes / open questions
- Constructor preallocation may be useful later, but only with a trustworthy capacity source.
- Exact byte preallocation is available from `Rows::lengths()` for the current `take_n` rebuild path.
