# PR Feedback

## Summary

- The change unwraps nested `Expr::Alias` layers during aggregate planning so user-provided aliases like `count_all().alias("TOTAL_COUNT")` correctly reach the underlying aggregate function.
- A regression test ensures `count_all()` can be aliased without panicking.

## Suggestions

- **Potential refactor**: the alias-unwrapping loop added to `create_aggregate_expr_and_maybe_filter` could be extracted into a small helper to avoid duplication and possibly reused by `create_window_expr`, which currently removes only a single alias layer.
- **Naming**: consider using `String::new()` instead of `String::default()` for clarity when initializing strings.
- **Commit message**: "pr" is terse; a descriptive message referencing the issue or bug would aid future traceability.

## Validation

- `cargo test -p datafusion-core --test physical_planner test_aggregate_count_all_with_alias`
