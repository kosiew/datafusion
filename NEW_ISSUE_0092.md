source: pr-22903_a
# Refactor: Centralize schema-aware logical-plan child rewrites

## Summary

DataFusion has multiple logical-plan rewrite paths that can replace child plans while preserving parent nodes that cache derived schema. When a child rewrite changes its output schema, the parent node must refresh its own cached schema before later optimizer rules read it.

Today this invariant is handled in ad hoc places. PR #22903 adds explicit child-schema change detection to `rewrite_plan_in_place` and calls `LogicalPlan::recompute_schema()` when needed. That fixes the immediate stale-schema bug, but the invariant still lives in one traversal path rather than in a shared schema-aware rewrite abstraction.

## Problem

Some logical plan nodes cache schema derived from their children, for example joins, projections, aggregates, aliases, windows, unions, recursive queries, and other wrapper nodes. Optimizer rules can rewrite a child so its schema changes, such as by adding/removing helper projections or pruning extracted expressions.

If the parent keeps its old cached schema after such a child rewrite:

- later projection pruning can request fields that no longer exist,
- column indices can be computed against stale fields,
- schema/name/qualifier contracts can diverge from the actual child output,
- failures surface later as missing-column or internal planner errors, far from the rewrite that caused them.

The current fix addresses this in `datafusion/optimizer/src/optimizer.rs` for the in-place optimizer traversal, but similar schema-refresh reasoning remains distributed across traversal helpers and optimizer rules.

## Current state

Relevant code paths:

- `datafusion/optimizer/src/optimizer.rs`
  - `rewrite_plan_in_place` now detects child schema changes and calls `LogicalPlan::recompute_schema()`.
- `datafusion/expr/src/logical_plan/tree_node.rs`
  - `LogicalPlan::map_children` rebuilds nodes but generally preserves cached parent schemas.
- `datafusion/expr/src/logical_plan/plan.rs`
  - `LogicalPlan::recompute_schema()` contains node-specific schema refresh logic.
- Optimizer rules sometimes call `recompute_schema()` manually after child rewrites.

This means the core invariant is not represented by a single API contract.

## Desired invariant

After any optimizer traversal replaces a child plan with a schema-different child, the parent plan exposed to later rules must either:

1. have its cached schema recomputed from its current children, or
2. deliberately preserve an explicit schema contract and validate that the new child schema is compatible with that contract.

No later optimizer rule should observe a parent whose cached schema is inconsistent with its rewritten children.

## Proposed approach

Introduce or refactor toward a shared schema-aware child rewrite helper for `LogicalPlan` traversal. Possible shape:

```rust
fn map_children_and_refresh_schema<F>(
    plan: LogicalPlan,
    f: F,
) -> Result<Transformed<LogicalPlan>>
where
    F: FnMut(LogicalPlan) -> Result<Transformed<LogicalPlan>>;
```

or an in-place equivalent:

```rust
fn map_children_mut_and_refresh_schema<F>(
    plan: &mut LogicalPlan,
    f: F,
) -> Result<bool>
where
    F: FnMut(&mut LogicalPlan) -> Result<bool>;
```

The helper should:

1. capture each child schema before rewrite,
2. rewrite children,
3. detect whether any rewritten child schema changed,
4. rebuild/recompute the parent schema if needed,
5. preserve existing traversal semantics, including stop/jump behavior where applicable,
6. make exceptions explicit for nodes whose output schema is intentionally independent of child schemas.

## Important design questions

- Should this live in `datafusion-expr` alongside `LogicalPlan::map_children`, or only in `datafusion-optimizer`?
- Should `LogicalPlan::map_children` itself become schema-aware, or should a new helper avoid changing existing semantics?
- Which nodes should recompute schema vs preserve an explicit contract?
- How should extension nodes participate, given they use `UserDefinedLogicalNode::with_exprs_and_inputs`?
- Should schema comparison include metadata and nullability, or match the existing logical optimizer invariant that ignores some schema dimensions?

## Suggested implementation plan

1. Inventory all uses of `map_children`, `rewrite`, `rewrite_with_subqueries`, and manual `recompute_schema()` in optimizer code.
2. Categorize logical plan nodes by schema behavior:
   - child-schema-derived,
   - expression-derived,
   - explicit fixed output schema,
   - schema-independent.
3. Add a schema-aware child rewrite helper with clear docs and tests.
4. Migrate `rewrite_plan_in_place` to use the helper.
5. Migrate optimizer rules that manually call `recompute_schema()` after child rewrites where the helper fits.
6. Keep manual recompute only where a rule has special schema semantics.
7. Add regression tests that exercise both in-place and ownership-based traversal paths.

## Test coverage

Add tests for:

- join parent schema refresh when left/right child projection width changes,
- filter/sort/limit wrapper schema refresh when child schema changes,
- projection/aggregate/window expression-derived schemas still recompute correctly,
- union schema behavior, including label preservation where required,
- subquery-containing plans using `rewrite_with_subqueries`,
- extension logical nodes, if feasible with a test node,
- no-op child rewrites do not force unnecessary schema changes.

Existing regression context:

- PR #22903 fixed stale parent schemas after join-key/leaf-expression helper projections changed child output schema.

## Acceptance criteria

- There is one documented API/helper for schema-aware child rewrites.
- Optimizer traversal code no longer duplicates child-schema change detection logic.
- Parent cached schemas are refreshed consistently after child schema changes.
- Existing optimizer behavior and schema contracts are preserved.
- Regression tests cover both direct children and subquery traversal where applicable.
- No meaningful optimizer-time regression on common no-op rewrites.

## Scope / non-goals

- Do not change SQL-visible behavior except fixing stale-schema inconsistencies.
- Do not broaden logical optimization schema invariants beyond the existing compatibility rules unless separately justified.
- Do not rewrite all optimizer rules mechanically if a staged migration is safer.

## Risk

Medium. This touches shared optimizer traversal behavior and may expose latent schema contract mismatches. The work should be staged with focused tests and careful review of logical-plan node-specific schema semantics.
