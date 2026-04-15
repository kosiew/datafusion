created #22769
source: pr-22620_a
# Make `LogicalPlan::Unnest` expression/rebuild contracts explicit

## Summary

`LogicalPlan::Unnest` currently has an inconsistent contract across the logical plan expression APIs:

- `LogicalPlan::expressions()` / `apply_expressions()` expose the `Unnest::exec_columns` as `Expr::Column` values.
- `LogicalPlan::with_new_exprs()` rejects expressions for `LogicalPlan::Unnest` via `assert_no_expressions(expr)?`.

This means generic optimizer code cannot safely assume that `node.with_new_exprs(node.expressions(), new_inputs)` is valid for all logical plan nodes. `PushDownLeafProjections` now has to special-case `Unnest` as a barrier to avoid an internal error.

## Motivation

The immediate regression fixed in PR #22620 was an optimizer failure:

```text
Internal error: Assertion failed: expr.is_empty(): Unnest ... should have no exprs, got [Column(...)]
```

The failure happened because `PushDownLeafProjections` rebuilt a node using:

```rust
node.with_new_exprs(node.expressions(), new_inputs)?
```

For `Unnest`, `expressions()` returns `exec_columns`, but `with_new_exprs()` requires the expression list to be empty. This is surprising for generic logical plan transforms and may cause similar bugs in other optimizer rules.

## Current behavior

Relevant locations:

- `datafusion/expr/src/logical_plan/tree_node.rs`
  - `LogicalPlan::apply_expressions()` maps `Unnest::exec_columns` into `Expr::Column` values.
- `datafusion/expr/src/logical_plan/plan.rs`
  - `LogicalPlan::with_new_exprs()` handles `LogicalPlan::Unnest` by calling `assert_no_expressions(expr)?`, then rebuilding from the old `exec_columns` and new input.
- `datafusion/optimizer/src/extract_leaf_expressions.rs`
  - `try_push_into_inputs()` now explicitly returns `Ok(None)` for `LogicalPlan::Unnest` to avoid this mismatch.

## Why this matters

Generic plan transforms often treat `expressions()` and `with_new_exprs()` as paired APIs: read expressions, transform or preserve them, then rebuild the node with new inputs and expressions. `Unnest` breaks that assumption.

This increases risk that future optimizer changes will:

1. hit internal errors when rebuilding `Unnest`, or
2. incorrectly treat `Unnest::exec_columns` as normal evaluable expressions, even though they are structural operator parameters with special input/output column identity semantics.

`Unnest` also has a column-identity hazard: an unnested output column may share a name with the input list/struct column but have different type and values. Name-based routing is insufficient to decide whether an expression can be pushed below `Unnest`.

## Proposed direction

Make the `Unnest` contract explicit at the logical-plan API layer. Possible approaches:

1. **Do not expose `Unnest::exec_columns` through generic expression APIs**
   - Treat them as structural parameters, not normal expressions.
   - `expressions()` would return empty for `Unnest`, matching `with_new_exprs()`.

2. **Allow `with_new_exprs()` to accept and validate `Unnest::exec_columns`**
   - Keep exposing them through `expressions()`.
   - Rebuild `Unnest` from the provided columns after validating they are plain columns and semantically valid for the new input.

3. **Split structural parameters from evaluable expressions**
   - Add or document a separate API for operator parameters such as `Unnest::exec_columns`.
   - Keep optimizer expression rewrites focused on expressions that can be transformed and rebound safely.

The best option should preserve current optimizer semantics while making invalid generic rewrites hard to write.

## Acceptance criteria

- The contract between `expressions()` / `apply_expressions()` and `with_new_exprs()` is documented and consistent for `LogicalPlan::Unnest`.
- Generic rebuild patterns either work for `Unnest` or are clearly prevented by API shape/documentation.
- Optimizer rules no longer need to rely on undocumented knowledge that `Unnest` exposes expressions but rejects them during rebuild.
- Existing Unnest behavior and schemas remain unchanged.
- Regression coverage includes a generic transform/rebuild path involving `Unnest`.

## Suggested tests

- Unit test showing the chosen contract for `LogicalPlan::Unnest`:
  - if `expressions()` returns empty, assert that explicitly;
  - if it returns `exec_columns`, assert `with_new_exprs(expressions(), new_inputs)` succeeds for valid input.
- Optimizer regression test with `PushDownLeafProjections` and `Unnest` remains passing.
- SQLLogicTest coverage for expressions above `Unnest`, including:
  - expression references a sibling input column preserved through `Unnest`;
  - expression references the unnested output column itself.

## Notes

This is out of scope for PR #22620. The PR's conservative barrier for `Unnest` is correct, but the underlying API mismatch remains and can affect future optimizer work.
