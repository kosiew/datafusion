source: pr-22444_a
# Refactor eliminate_outer_join null-rejection analysis to track join sides directly

## Summary

`datafusion/optimizer/src/eliminate_outer_join.rs` currently tracks null-rejection evidence as `Vec<Column>` and later reduces those columns to two booleans: whether the predicate rejects NULLs from the join's left side, right side, or both.

This is more indirect than the optimizer needs. The join-elimination decision does not depend on which specific column was null-rejecting; it only depends on which join side was null-rejecting. Refactor the analysis to represent that invariant directly.

## Current code

In `try_simplify_join`:

- `null_rejecting_cols: Vec<Column>` is built.
- `extract_null_rejecting_columns(...)` appends column references into that vector.
- The caller loops over those columns and calls `join.left.schema().has_column(col)` / `join.right.schema().has_column(col)` to derive `left_non_nullable` and `right_non_nullable`.
- `eliminate_outer(...)` only consumes those side-level booleans.

In `extract_null_rejecting_columns`:

- `Expr::Column` pushes the column into the vector.
- Top-level `AND` appends evidence from both branches.
- `OR` and nested `AND` compute branch-local column vectors, then push one representative column for each side where both branches reject NULLs.

The current logic works, but the data model is column-driven while the decision is side-driven.

## Refactor goal

Replace the intermediate `Vec<Column>` contract with explicit side-level state, for example:

```rust
struct NullRejectingSides {
    left: bool,
    right: bool,
}
```

The helper should return or update `NullRejectingSides` directly. `try_simplify_join` should then pass those booleans to `eliminate_outer(...)` without re-scanning collected columns.

## Why this is a good refactor

- **Clearer invariant:** code states directly that outer-join elimination depends on null-rejection of join sides, not individual column identities.
- **Less indirection:** removes the collect-columns-then-reclassify-columns step.
- **Less repeated work:** avoids repeated `has_column` scans after extraction.
- **Clearer OR handling:** branch combination can be expressed as side-level intersection instead of selecting representative columns.
- **Easier maintenance:** future changes can reason about `left` / `right` null-rejection directly.

## Required behavior preservation

This must be a behavior-preserving cleanup. Do not change SQL-visible semantics or expected optimized plans.

Preserve these rules:

- Top-level `AND` in a `WHERE` predicate unions null-rejection evidence from both conjuncts.
- `OR` credits a side only when both branches independently reject NULLs from that side.
- Nested `AND` follows the same side-intersection behavior as `OR`, matching current conservative handling.
- `IS NOT NULL`, `IS TRUE`, `IS FALSE`, and `IS NOT UNKNOWN` only recurse when allowed by the current `top_level` guard.
- Operators using `returns_null_on_null()` continue to recurse as they do today.
- `IS NULL`, `IS NOT TRUE`, `IS NOT FALSE`, `IS UNKNOWN`, function calls, literals, subqueries, and other currently skipped expressions remain skipped.
- Qualified or ambiguous column handling remains equivalent to the current `DFSchema::has_column`-based side-membership checks.

## Suggested implementation shape

Possible helper methods:

```rust
impl NullRejectingSides {
    fn union(self, other: Self) -> Self { ... }
    fn intersection(self, other: Self) -> Self { ... }
}
```

Possible extraction flow:

- `Expr::Column(col)` returns side membership based on `left_schema.has_column(col)` and `right_schema.has_column(col)`.
- Top-level `AND` returns `left_sides.union(right_sides)`.
- `OR` and nested `AND` return `left_sides.intersection(right_sides)`.
- Null-propagating operators recurse and union child evidence as current column collection does.

Keep the refactor local to `eliminate_outer_join.rs` unless a small helper extraction clearly improves readability.

## Non-goals

- No optimizer behavior change.
- No new outer-join elimination cases.
- No changes outside `eliminate_outer_join` unless mechanically necessary.
- No public API changes.

## Acceptance criteria

- `try_simplify_join` no longer builds or consumes `Vec<Column>` for null-rejection analysis.
- Side-level null-rejection state is explicit in the code.
- Existing eliminate_outer_join behavior and expected plans are unchanged.
- OR / nested-AND side-intersection behavior remains intact.
- Current top-level guard behavior remains intact.
- Code is simpler to read than the current collect-and-reduce flow.

## Test strategy

Run the existing focused optimizer tests:

```bash
cargo test -p datafusion-optimizer eliminate_outer_join --lib
```

Ensure existing coverage still exercises:

- LEFT / RIGHT / FULL outer join conversion under null-rejecting predicates.
- No-conversion cases under null-accepting predicates.
- OR predicates that reject one side or both sides.
- Top-level guard behavior for `IS NOT NULL`, `IS TRUE`, `IS FALSE`, and `IS NOT UNKNOWN`.

If the helper becomes easy to test directly, optionally add focused unit tests for side-level union/intersection behavior.

## Impact

This refactor should reduce conceptual complexity without changing optimizer output. It encodes the true invariant directly: outer-join elimination is driven by whether the filter rejects null-padded rows from the left side, the right side, or both.

## Origin

Pre-existing design opportunity identified during PR review for apache/datafusion PR #22444.
