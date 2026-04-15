created #22666
source: pr-22068_a
# Grouped first_value/last_value FILTER Incorrectly Includes NULL Predicate Rows

## Summary
Grouped `first_value` and `last_value` currently apply aggregate `FILTER` using only `BooleanArray::value(idx)`, without checking predicate validity. This allows rows where the filter predicate is `NULL` to be treated as passing when the underlying value bit is set.

Under SQL aggregate `FILTER` semantics, rows pass only when the filter evaluates to `TRUE` (`Some(true)`), and must be excluded when it is `FALSE` or `NULL`.

## Affected Area
- File: `datafusion/functions-aggregate/src/first_last.rs`
- Function: `FirstLastGroupsAccumulator::get_filtered_extreme_of_each_group`
- Current predicate:
  - `let passed_filter = opt_filter.is_none_or(|x| x.value(idx_in_val));`

## Problem Statement
The grouped path for `first_value`/`last_value` computes `passed_filter` from `BooleanArray::value()` only. For nullable boolean arrays, this can treat `NULL` rows as passing if the value bitmap bit is `1`, even though the row should be excluded.

This violates SQL semantics for aggregate `FILTER` and can produce incorrect non-NULL aggregate results when all rows should be filtered out.

## Reproduction
### SQL
```sql
SELECT
  g,
  first_value(a ORDER BY a) FILTER (WHERE b < 1) AS fv
FROM (
  VALUES
    (0, 10, CAST(NULL AS INT)),
    (0, 20, 2)
) AS t(g, a, b)
GROUP BY g;
```

### Observed Result
- `fv = 10`

### Expected Result
- `fv = NULL`

Reason:
- Row `(0, 10, NULL)` has `b < 1` = `NULL` and must not pass `FILTER`.
- Row `(0, 20, 2)` has `b < 1` = `FALSE` and must not pass `FILTER`.
- No rows satisfy `Some(true)`, so grouped `first_value` should return `NULL`.

## Root Cause
In grouped `first_last` processing, filter pass logic does not encode the invariant:
- row passes aggregate filter iff predicate is `Some(true)`

Instead, it checks only `value(idx)`, which ignores nullability at that row.

## Proposed Fix
Update grouped filter evaluation in `get_filtered_extreme_of_each_group` to require both validity and value (or equivalent `Some(true)` check), for example:
- `x.is_valid(idx_in_val) && x.value(idx_in_val)`
- or `x.iter().nth(idx_in_val) == Some(Some(true))`
- or a shared helper encapsulating `Some(true)` semantics.

A helper-based approach is preferable to avoid semantic drift with other grouped aggregate paths.

## Testing Plan
1. Add SQLLogicTest coverage for grouped `first_value` with nullable `FILTER` predicate returning no `TRUE` rows.
2. Add the equivalent grouped `last_value` case.
3. Add at least one mixed case where only some rows satisfy `Some(true)` to verify rows with `NULL` predicates are excluded.
4. Run:
   - `cargo test -p datafusion-functions-aggregate --lib first_last`
   - `cargo test -p datafusion-sqllogictest --test sqllogictests aggregate`

## Acceptance Criteria
1. Grouped `first_value` and `last_value` exclude rows where `FILTER` evaluates to `NULL`.
2. Reproducer query returns `NULL` as expected.
3. New regression tests fail before the fix and pass after.
4. No behavior regressions for non-null filter predicates.

## Impact
- Correctness issue in SQL-visible results.
- Affects grouped ordered aggregates (`first_value` / `last_value`) with nullable filter predicates.
- Can silently return non-NULL values when SQL semantics require `NULL`.

## Scope and Priority
- Scope: targeted bug fix in grouped `first_last` filter predicate handling.
- Priority: medium-high (correctness bug with user-visible query results).

## Additional Notes
This appears adjacent to, but separate from, prior `accumulate_multiple` / `accumulate_indices` fixes. It should be tracked and fixed as a dedicated follow-up to close NULL `FILTER` semantic gaps in grouped aggregate paths.

## GitHub-Ready Snippet
### Suggested Title
Grouped first_value/last_value incorrectly treat NULL FILTER predicate rows as passing

### Suggested Labels
- bug
- sql
- aggregates

### Suggested Body
Grouped `first_value` / `last_value` currently evaluate aggregate `FILTER` in `first_last.rs` using `BooleanArray::value(idx)` only, which can treat `NULL` predicate rows as passing when value bits are set.

This violates SQL aggregate `FILTER` semantics (row passes iff predicate is `Some(true)`).

Reproducer:
```sql
SELECT
  g,
  first_value(a ORDER BY a) FILTER (WHERE b < 1) AS fv
FROM (
  VALUES
    (0, 10, CAST(NULL AS INT)),
    (0, 20, 2)
) AS t(g, a, b)
GROUP BY g;
```
Observed: `fv = 10`
Expected: `fv = NULL`

Proposed fix: in grouped `first_last` path, require `is_valid && value` (or equivalent `Some(true)` helper) when evaluating `FILTER`, and add SLT coverage for grouped `first_value` and `last_value` with nullable predicates.
