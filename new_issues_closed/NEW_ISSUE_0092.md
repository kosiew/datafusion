created: #23666
source: pr-22903_a
# Refactor: Add an optimizer-local helper for schema-aware child rewrites

## Problem

Some logical plan nodes cache schemas derived from their children. When an optimizer rule rewrites a child and the child's output schema changes, the parent must refresh its cached schema before later optimizer rules inspect it.

The current in-place optimizer traversal handles this directly in `rewrite_plan_in_place` by comparing child schemas before/after recursion and calling `LogicalPlan::recompute_schema()` when needed. That fixes the known stale-schema path, but the contract is still local to one traversal implementation.

Other optimizer code still uses `LogicalPlan::map_children(...)` and then manually decides whether to call `recompute_schema()`. This keeps the invariant easy to miss when adding or changing optimizer rules.

Relevant current code:

- `datafusion/optimizer/src/optimizer.rs`
  - private `map_children_mut(...)`
  - `rewrite_plan_in_place(...)` detects child schema changes and recomputes the parent schema
- `datafusion/expr/src/logical_plan/tree_node.rs`
  - `LogicalPlan::map_children(...)` rebuilds nodes while preserving cached parent schemas
- `datafusion/expr/src/logical_plan/plan.rs`
  - `LogicalPlan::recompute_schema()` contains node-specific schema refresh logic
- optimizer rules with manual recompute-after-child-rewrite logic, for example:
  - `datafusion/optimizer/src/optimize_projections/mod.rs`
  - `datafusion/optimizer/src/eliminate_cross_join.rs`

## Why it matters

The invariant is correctness-sensitive but currently implicit:

> If an optimizer child rewrite changes a child schema, the parent plan observed by later optimizer rules must not expose a stale cached schema.

A stale parent schema can lead to projection pruning, column resolution, or later rule logic using field positions/names that no longer match the rewritten children. Failures then surface far from the rewrite that introduced the inconsistency.

This issue is a maintainability refactor, not a claim that the current in-place optimizer path is broken.

## Invariant / desired behavior

After an optimizer traversal rewrites direct children of a logical plan node:

- if no child schema changed, preserve the current behavior and avoid unnecessary schema recomputation;
- if any child schema changed, recompute the parent schema before returning it to later optimizer code;
- if a node intentionally has an explicit/fixed schema contract, that behavior remains encoded in `LogicalPlan::recompute_schema()` or documented at the call site;
- traversal semantics, including transformed/no-op status and recursion control, are preserved.

## Proposed direction

Add a small optimizer-local helper rather than changing `LogicalPlan::map_children` globally.

Possible shape for owned-plan optimizer rewrites:

```rust
fn map_children_recompute_schema_if_needed<F>(
    plan: LogicalPlan,
    f: F,
) -> Result<Transformed<LogicalPlan>>
where
    F: FnMut(LogicalPlan) -> Result<Transformed<LogicalPlan>>;
```

The helper should:

1. record each direct child schema before rewrite;
2. rewrite direct children using the existing `map_children` behavior;
3. detect whether any changed child now has a different schema;
4. call `LogicalPlan::recompute_schema()` only when needed;
5. preserve `Transformed` state and `TreeNodeRecursion` behavior from the child rewrite.

If useful, add a mutating variant for the in-place optimizer traversal later, but avoid broad API changes until the owned helper proves useful.

## Scope

### In

- Add a private helper under `datafusion/optimizer/src/` for schema-aware direct-child rewrites.
- Use the helper in one or two existing optimizer rules that currently do `map_children(...)` followed by manual `recompute_schema()`.
- Keep `rewrite_plan_in_place` behavior unchanged or migrate it only if the helper clearly fits without making the traversal harder to read.
- Add focused unit coverage for the helper through an optimizer rule or small test-only rule.

### Out

- Do not change `LogicalPlan::map_children` semantics globally.
- Do not redesign `TreeNode` traversal APIs.
- Do not require all optimizer rules to migrate in one patch.
- Do not change SQL-visible schemas except where existing stale-schema behavior would already be considered a bug.
- Do not broaden schema equality rules beyond the current `DFSchema` equality checks used by optimizer code.
- Do not redesign extension-node APIs as part of this issue.

## Acceptance criteria

- [ ] There is one documented optimizer-local helper for direct-child rewrites that refreshes parent schema when a changed child schema differs.
- [ ] At least one manual `map_children(...)` + `recompute_schema()` optimizer site is migrated to the helper.
- [ ] No-op child rewrites do not force unnecessary parent schema recomputation.
- [ ] Parent schemas are refreshed when changed children produce different output schemas.
- [ ] Existing in-place optimizer stale-schema protection remains intact.
- [ ] Existing optimizer tests pass.

## Tests / verification

Add focused tests that cover:

- a child rewrite that changes schema and causes the parent schema to refresh;
- a child rewrite that changes the plan but not the child schema and does not require parent schema refresh;
- an existing rule migrated to the helper still preserves its expected optimized plan.

Run targeted tests first:

```bash
cargo test -p datafusion-optimizer
```

If the migrated rule has focused tests, run those specifically as well, for example:

```bash
cargo test -p datafusion-optimizer optimize_projections
cargo test -p datafusion-optimizer eliminate_cross_join
```

## Notes / open questions

- Extension nodes remain subtle because they rebuild via `UserDefinedLogicalNode::with_exprs_and_inputs`. Keep their behavior unchanged unless the helper naturally covers them through `map_children`.
- Subquery traversal is separate from direct child traversal. A follow-up can consider a schema-aware helper for `rewrite_with_subqueries` if duplicated recompute logic remains there.
- If future work wants this behavior in `datafusion-expr`, that should be a separate design discussion because changing `LogicalPlan::map_children` could affect many callers that currently rely on schema-preserving rebuilds.
