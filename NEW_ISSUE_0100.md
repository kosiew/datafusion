source: pr-23002_a
# Refactor: Centralize join-input normalization before SQL unparse recursion

## Summary

Join unparsing currently normalizes the left and right inputs inline in `datafusion/sql/src/unparser/plan.rs`. Recent alias-scope fixes added more normalization logic around join inputs, including table-scan filter extraction and passthrough projection unwrapping. Keeping this logic duplicated or asymmetric makes it easy for one join side to diverge from the other.

Refactor join input handling into a small shared helper that normalizes either side before recursive unparsing.

## Motivation

The SQL unparser must preserve alias scope when optimized logical plans contain join inputs that have been rewritten by optimizer passes. In particular, optimized plans can contain a qualified passthrough `Projection` over a `Join`, while the outer join condition still references aliases from inside that join input.

When this normalization is applied only on one side, right-nested or right-input join shapes can still emit invalid SQL. Centralizing the normalization makes the invariant explicit:

> Before recursively unparsing a join input in an already-projected context, remove transparent join-input wrappers that would create a derived-table boundary while leaving outer join conditions referencing aliases inside that boundary.

## Current code shape

Observed in `datafusion/sql/src/unparser/plan.rs`, inside `LogicalPlan::Join` handling:

- Left input is transformed with `try_transform_to_simple_table_scan_with_filters`.
- Additional passthrough projection unwrapping is applied to the left input when `already_projected` is true.
- Right input has nearby, similar table-scan filter extraction logic.
- The join-input normalization rule can drift between left and right inputs.

This is a maintainability risk because join semantics are symmetric for many of these structural unparse rewrites, even when the later SQL generation treats left/right differently.

## Proposed refactor

Add a local helper near the join unparsing code, for example:

```rust
fn normalize_join_input_for_unparse(
    plan: &Arc<LogicalPlan>,
    already_projected: bool,
    table_scan_filters: &mut Vec<Expr>,
) -> Result<Arc<LogicalPlan>> {
    let normalized = match try_transform_to_simple_table_scan_with_filters(plan)? {
        Some((plan, filters)) => {
            table_scan_filters.extend(filters);
            Arc::new(plan)
        }
        None => Arc::clone(plan),
    };

    if already_projected {
        Ok(Self::unwrap_qualified_passthrough_join_projection(normalized))
    } else {
        Ok(normalized)
    }
}
```

Then use it for both sides:

```rust
let left_plan = Self::normalize_join_input_for_unparse(
    left_plan,
    already_projected,
    &mut table_scan_filters,
)?;

let right_plan = Self::normalize_join_input_for_unparse(
    right_plan,
    already_projected,
    &mut table_scan_filters,
)?;
```

Exact signatures may need adjustment depending on ownership and helper visibility.

## Expected benefits

- Encodes the join-input alias-scope invariant once.
- Prevents left/right behavior drift.
- Makes future join-unparse fixes easier to review.
- Reduces inline control-flow in the already-large `LogicalPlan::Join` arm.
- Makes regression tests map to a single normalization point.

## Scope

In scope:

- Local refactor under `datafusion/sql/src/unparser/plan.rs`.
- Preserve existing behavior.
- Apply shared helper to both left and right join inputs.
- Keep helper private unless broader reuse is found.
- Add or retain tests covering nested join inputs on both sides.

Out of scope:

- Redesigning the SQL unparser.
- Changing logical optimizer output.
- Changing SQL alias naming policy.
- Adding public APIs.

## Suggested tests

Add targeted regression coverage for both sides of the invariant:

1. Optimized plan where the left join input is a qualified passthrough `Projection(Join)` and the outer join condition references aliases from inside it.
2. Same shape on the right join input.
3. Verify generated DuckDB SQL does not create a derived-table boundary that hides aliases used by the outer join condition.
4. Existing unparser roundtrip or DuckDB dialect tests should continue passing.

A narrow command during development:

```bash
cargo test -p datafusion --test core_integration optimized_duckdb_unparse_preserves_derived_table_scope -- --nocapture
```

Consider adding a second targeted test for the right-input case rather than widening the existing assertion-only test too much.

## Risks and cautions

- Do not unwrap projections that are not transparent passthrough projections.
- Do not drop projection expressions that rename, compute, or reorder columns in a way required by SQL output.
- Preserve table-scan filter collection semantics and order.
- Ensure semi/anti/mark join handling still uses the correct input after normalization.

## Acceptance criteria

- Join input normalization is implemented through one shared private helper.
- Both left and right inputs use the helper.
- Existing behavior is preserved for non-join projections and non-passthrough projections.
- Regression coverage includes both left and right nested join-input alias-scope cases.
- Targeted unparser tests pass.
