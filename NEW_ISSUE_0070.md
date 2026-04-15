source: filtering-logic-01-22665a
# Refactor Issue 02: Add SQL-Level Coverage for Nullable Aggregate FILTER Predicates

## Summary

Add end-to-end SQLLogicTest coverage for nullable aggregate `FILTER` predicates, especially grouped `first_value` and `last_value`, if a stable SQL reproducer can exercise the grouped aggregate path.

The low-level Rust tests cover Arrow buffers where the filter value bit is true but validity is NULL. SQL tests should verify the user-facing contract: aggregate `FILTER` accepts only `TRUE`; both `FALSE` and `NULL` reject the row.

## Core Invariant

From SQL, aggregate `FILTER` must behave as:

```sql
FILTER (WHERE predicate)
```

passes a row only when `predicate IS TRUE`.

Equivalent behavior table:

| SQL predicate result | Aggregate row included? |
| --- | --- |
| `TRUE` | yes |
| `FALSE` | no |
| `NULL` | no |

This issue is about end-to-end SQL behavior, not low-level Arrow value bits under nulls.

## Problem

The direct Rust tests for grouped `first_value` / `last_value` can construct this precise edge case:

```rust
let filter = BooleanArray::new(
    BooleanBuffer::from(vec![true]),
    Some(NullBuffer::from(vec![false])),
);
```

That is the strongest unit-level regression for the original bug. However, SQL users observe aggregate FILTER through SQL expressions, not hand-built Arrow arrays. The PR would be better protected by SQLLogicTest coverage showing nullable FILTER predicates reject rows for grouped first/last aggregates.

Caveat: SQL expression evaluation may not preserve arbitrary true value bits under NULL validity. Therefore SQL tests should assert semantic behavior (`NULL` predicate rejects row), not the exact Arrow-buffer shape from the Rust reproducer.

## Evidence

Relevant implementation and tests:

- `datafusion/functions-aggregate/src/first_last.rs`
  - grouped `first_value` / `last_value` now use shared `filter_to_validity`.
  - Rust tests cover nullable filter arrays with true value bits under NULL validity.
- `datafusion/functions-aggregate-common/src/aggregate/groups_accumulator/nulls.rs`
  - `filter_to_validity` defines `Some(true)` semantics.
- `datafusion/sqllogictest/test_files/aggregate.slt`
  - existing SQLLogicTest file for aggregate behavior, including `first_value` and `last_value` cases.

Existing SQL test areas found during review:

- `aggregate.slt` has `first_value(...)` / `last_value(...)` aggregate coverage.
- `window.slt` has window-function `FILTER` rejection coverage, but this issue is about aggregate FILTER semantics.

## Scope

In scope:

- Add SQLLogicTest cases for nullable aggregate FILTER predicates.
- Prefer grouped `first_value` / `last_value` with `ORDER BY` inside the aggregate.
- Include cases where a row would otherwise be chosen by ordering but has a NULL filter predicate.
- Include a case where all rows are rejected and result is NULL.
- Keep tests deterministic and small.

Out of scope:

- Reproducing Arrow value-bit-under-null internals through SQL.
- Changing execution code.
- Broad aggregate FILTER test matrix across every aggregate function.
- Window function FILTER behavior.
- Planner or optimizer rewrites.

## Candidate SQLLogicTest Shape

Add to `datafusion/sqllogictest/test_files/aggregate.slt` near existing `first_value` / `last_value` aggregate tests.

Example using inline `VALUES` and nullable boolean predicates:

```sql
query IT
SELECT
  g,
  first_value(v ORDER BY sort_key) FILTER (WHERE pred) AS first_v
FROM (VALUES
  (1, 10, 1, CAST(NULL AS BOOLEAN)),
  (1, 20, 2, FALSE),
  (2, 30, 1, CAST(NULL AS BOOLEAN)),
  (2, 40, 2, TRUE)
) AS t(g, v, sort_key, pred)
GROUP BY g
ORDER BY g;
----
1 NULL
2 40
```

Possible `last_value` companion:

```sql
query IT
SELECT
  g,
  last_value(v ORDER BY sort_key) FILTER (WHERE pred) AS last_v
FROM (VALUES
  (1, 10, 1, TRUE),
  (1, 20, 2, CAST(NULL AS BOOLEAN)),
  (1, 30, 3, FALSE),
  (2, 40, 1, CAST(NULL AS BOOLEAN)),
  (2, 50, 2, TRUE)
) AS t(g, v, sort_key, pred)
GROUP BY g
ORDER BY g;
----
1 10
2 50
```

Adjust expected type markers and NULL rendering to match nearby `aggregate.slt` conventions.

## Implementation Plan

1. Inspect nearby `aggregate.slt` style.
   - Match query type markers (`query I`, `query IT`, etc.).
   - Match NULL output spelling.
   - Place tests near existing `first_value` / `last_value` aggregate coverage.

2. Add minimal grouped SQL cases.
   - One `first_value` case where the earliest ordered row has `NULL` predicate and must be skipped.
   - One `last_value` case where a later ordered row has `NULL` or `FALSE` predicate and must be skipped.
   - One all-rejected group that emits NULL.

3. Verify the tests use aggregate syntax, not window syntax.
   - Use `GROUP BY`.
   - Do not use `OVER (...)`.

4. Confirm the physical path is relevant if possible.
   - If the test only hits a non-grouped/single accumulator path, adjust query to group by a column.
   - If needed, use `EXPLAIN` locally to confirm grouped aggregate execution.

5. If SQL cannot reliably expose the intended path, document that in the PR and keep Rust tests as authoritative.

## Validation

Run the target SQLLogicTest file:

```bash
cargo test -p datafusion-sqllogictest --test sqllogictests -- aggregate.slt
```

Also run the focused Rust tests that protect the low-level edge:

```bash
cargo test -p datafusion-functions-aggregate first_last --lib
cargo test -p datafusion-functions-aggregate-common test_accumulate_indices_with_null_filter --lib
```

Before PR ready:

```bash
cargo fmt --all
```

If only SLT files are changed, full clippy is likely unnecessary unless CI policy requires it.

## Risks

Correctness risk: low. These tests should only assert existing SQL semantics.

Coverage risk: medium. SQL may not reproduce true value bits under NULL validity, so this should complement—not replace—the direct Rust tests.

Stability risk: low if tests use inline `VALUES` and explicit `CAST(NULL AS BOOLEAN)`.

Scope risk: low. Keep this as test-only unless the SQL test exposes an additional bug.

## Acceptance Criteria

- SQLLogicTest coverage exists for grouped aggregate `first_value` and/or `last_value` with nullable FILTER predicates.
- Tests prove `NULL` predicate rows are rejected.
- At least one group where all rows are rejected emits NULL.
- At least one group where only a later/earlier `TRUE` row passes returns that row.
- Tests are placed in the aggregate SLT suite and follow local output conventions.
- If SQL cannot exercise the intended path, the limitation is documented and no misleading test is added.
