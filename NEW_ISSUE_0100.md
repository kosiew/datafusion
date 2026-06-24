source: pr-23002_a
# Refactor: Centralize join-input normalization before SQL unparse recursion

## Summary

Join unparsing currently handles left and right inputs inline in `datafusion/sql/src/unparser/plan.rs`. Both sides perform table-scan filter extraction, but already-projected qualified passthrough `Projection(Join)` inputs intentionally diverge:

- the left input can be unwrapped before recursive unparsing;
- the right input may need to be converted to a nested join relation so SQL preserves required join grouping.

Refactor the duplicated common setup into a small helper, while keeping the side-specific passthrough-projection behavior explicit.

## Motivation

The SQL unparser must preserve alias scope when optimized logical plans contain join inputs that have been rewritten by optimizer passes. In particular, optimized plans can contain a qualified passthrough `Projection` over a `Join`, while the outer join condition still references aliases from inside that join input.

The invariant is side-aware:

> Before unparsing a join input in an already-projected context, avoid creating a derived-table boundary that hides aliases needed by an outer join condition. For the left input this can mean unwrapping a transparent passthrough `Projection(Join)`. For the right input this can mean preserving the nested join grouping with `qualified_passthrough_join_projection_to_nested_relation`.

## Current code shape

Observed in `datafusion/sql/src/unparser/plan.rs`, inside `LogicalPlan::Join` handling:

- Left input is transformed with `try_transform_to_simple_table_scan_with_filters`.
- Right input repeats the same table-scan filter extraction logic.
- Additional passthrough projection unwrapping is applied only to the left input when `already_projected` is true.
- The right input has a different, intentional path: `qualified_passthrough_join_projection_to_nested_relation`, which can emit a nested join relation instead of recursively unparsing through a derived-table boundary.

This is a maintainability risk because the common table-scan extraction is duplicated, while the side-specific alias-scope behavior is subtle and easy to accidentally “simplify” into an incorrect symmetric rewrite.

## Proposed refactor

Add a local helper near the join unparsing code for the shared table-scan-filter extraction, for example:

```rust
fn extract_join_input_table_scan_filters(
    plan: &Arc<LogicalPlan>,
    table_scan_filters: &mut Vec<Expr>,
) -> Result<Arc<LogicalPlan>> {
    match try_transform_to_simple_table_scan_with_filters(plan)? {
        Some((plan, filters)) => {
            table_scan_filters.extend(filters);
            Ok(Arc::new(plan))
        }
        None => Ok(Arc::clone(plan)),
    }
}
```

Then keep the side-specific handling visible:

```rust
let left_plan = Self::extract_join_input_table_scan_filters(
    left_plan,
    &mut table_scan_filters,
)?;
let left_plan = if already_projected {
    Self::unwrap_qualified_passthrough_join_projection(left_plan)
} else {
    left_plan
};

let right_plan = Self::extract_join_input_table_scan_filters(
    right_plan,
    &mut table_scan_filters,
)?;

let mut right_relation = RelationBuilder::default();
if already_projected
    && let Some(nested_relation) = self
        .qualified_passthrough_join_projection_to_nested_relation(
            right_plan.as_ref(),
            query,
        )?
{
    right_relation = nested_relation;
} else {
    // existing recursive right-input unparsing
}
```

A side-aware helper is also acceptable if it preserves the distinction between left unwrapping and right nested-relation construction. Avoid a symmetric helper that blindly unwraps both sides.

## Expected benefits

- Removes duplicated table-scan filter extraction.
- Documents the side-specific alias-scope invariant near the code that enforces it.
- Makes future join-unparse fixes easier to review.
- Reduces inline control-flow in the already-large `LogicalPlan::Join` arm without hiding important left/right differences.
- Reduces risk of an incorrect symmetric passthrough-projection rewrite.

## Scope

In scope:

- Local refactor under `datafusion/sql/src/unparser/plan.rs`.
- Preserve existing behavior.
- Apply shared table-scan-filter extraction helper to both left and right join inputs.
- Preserve left-only passthrough projection unwrapping.
- Preserve right-side nested-relation handling for qualified passthrough `Projection(Join)` inputs.
- Keep helper private unless broader reuse is found.
- Add or retain tests covering nested join inputs on both sides.

Out of scope:

- Redesigning the SQL unparser.
- Changing logical optimizer output.
- Changing SQL alias naming policy.
- Adding public APIs.

## Suggested tests

Add or retain targeted regression coverage for both sides of the invariant:

1. Optimized plan where the left join input is a qualified passthrough `Projection(Join)` and the outer join condition references aliases from inside it.
2. Right-input qualified passthrough `Projection(Join)` coverage; current coverage includes `test_unparse_projected_join_unwraps_right_nested_passthrough_projection` in `datafusion/sql/tests/cases/plan_to_sql.rs`.
3. Verify generated DuckDB SQL does not create a derived-table boundary that hides aliases used by the outer join condition.
4. Existing unparser roundtrip or DuckDB dialect tests should continue passing.

A narrow command during development:

```bash
cargo test -p datafusion --test core_integration optimized_duckdb_unparse_preserves_derived_table_scope -- --nocapture
```

Consider adding a left-input counterpart if missing, rather than widening existing assertion-only tests too much.

## Risks and cautions

- Do not unwrap projections that are not transparent passthrough projections.
- Do not drop projection expressions that rename, compute, or reorder columns in a way required by SQL output.
- Preserve table-scan filter collection semantics and order.
- Ensure semi/anti/mark join handling still uses the correct input after normalization.

## Acceptance criteria

- Shared table-scan-filter extraction is implemented through one private helper.
- Both left and right inputs use that helper.
- Left/right passthrough-projection behavior remains intentionally distinct.
- Existing behavior is preserved for non-join projections and non-passthrough projections.
- Regression coverage includes both left and right nested join-input alias-scope cases.
- Targeted unparser tests pass.
