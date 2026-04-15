created #22670
source: pr-22313_a
# Tighten `ScalarSubqueryToJoin` Projection Rewrite State and Coverage

## Summary
`ScalarSubqueryToJoin` projection rewrite logic already uses slot-indexed ownership (`alias_to_index` + `rewrite_exprs`) and no longer depends on `Expr -> Expr` rewrite maps.

However, state is still distributed across parallel containers and tested mostly through a small set of projection scenarios. This issue proposes a follow-up cleanup to make ownership/rewrite progression more explicit and add targeted regression coverage.

Goal: reduce maintenance risk in a historically fragile path without changing behavior.

## Context
In `datafusion/optimizer/src/scalar_subquery_to_join.rs` (projection branch), correlated scalar subqueries are extracted, then each subquery is converted to a join and projection expressions are rewritten iteratively.

Current shape (conceptually):

- collect all extracted `(subquery, alias)` pairs
- map each alias to a projection slot index
- keep per-slot rewritten expression state in a vector
- for each built join, locate owning expression via alias, then rewrite it

This is functionally correct today, but ownership and rewrite progression are still spread across loosely coupled containers (`all_subqueries`, `alias_to_index`, `rewrite_exprs`), which makes future changes harder to reason about.

## Problem
The current implementation has maintainability risk:

1. Ownership of subquery aliases and per-slot rewrite state is represented indirectly via parallel vectors/maps.
2. The projection branch performs iterative in-place rewrites where subtle ordering/name-preservation regressions are possible.
3. Test coverage does not clearly stress multi-subquery-per-slot and repeated structurally similar projection expressions.

This is the same class of fragility surfaced by recent fixes in this area.

## Proposed Design
Keep behavior unchanged, but make projection rewrite state explicit in one slot-owned model.

Example model:

```rust
struct ProjectionRewriteState {
    rewritten_expr: Expr,
    subquery_aliases: Vec<String>,
}
```

Drive rewrite logic from:

- `Vec<ProjectionRewriteState>` indexed by projection slot
- `HashMap<String, usize>` mapping extracted alias to owning slot
- extraction-order queue of `(Subquery, String)` (or equivalent)

High-level flow:

1. Initialize per-slot state from projection expressions.
2. During extraction, append aliases into the owning slot and populate `alias -> slot_index`.
3. For each `(subquery, alias)` join conversion:
   - resolve owning slot index from `alias -> slot_index`
    - rewrite `states[slot_index].rewritten_expr` in place using `expr_check_map`
4. Build final projection expressions from per-slot rewritten expressions, preserving output names/aliases exactly as today.

## Why This Helps
1. Encodes ownership and rewrite progression directly in slot state.
2. Reduces cognitive overhead from parallel-container coupling.
3. Makes it easier to audit extraction/rewrite order guarantees.
4. Improves readability and lowers regression risk in future optimizer work.

## Scope
In scope:

- Refactor projection arm of `ScalarSubqueryToJoin` to explicit slot-owned state (without behavioral change).
- Keep existing behavior and output schema/name preservation unchanged.
- Update/add tests as needed.

Out of scope:

- Broader redesign of scalar-subquery rewrite outside projection branch.
- Behavioral changes to which queries are optimized.

## Acceptance Criteria
1. No behavior regression for correlated scalar subquery rewrites in projection context.
2. Projection rewrite bookkeeping is modeled through explicit slot-owned rewrite state.
3. Alias ownership remains represented by slot index (or equivalent direct ownership model).
4. Existing tests pass and targeted projection tests are added for this refactor-sensitive area.

## Test Plan
Add/adjust tests to cover:

1. Multiple projection expressions, each with correlated scalar subqueries.
2. A projection expression containing multiple correlated scalar subqueries.
3. Repeated/structurally-similar expressions in projection list (ensure correct slot ownership rewrite).
4. Existing regression scenario from the recent fix.

Prefer SQL-visible behavior checks via sqllogictest where appropriate, plus focused optimizer/unit tests near the rule implementation.

## Risks and Mitigations
Risk: subtle rewrite-order or alias-preservation regression.

Mitigation:

- Preserve extraction and rewrite order semantics.
- Validate output expression naming behavior against current logic.
- Add targeted tests for mixed projection/subquery layouts.

## Effort Estimate
Medium.

Expected to be localized to projection handling in `datafusion/optimizer/src/scalar_subquery_to_join.rs` plus related tests/SLT updates.

## References
- PR review note identifying this refactor opportunity (pre-existing design risk, non-blocking).
- Current implementation in `datafusion/optimizer/src/scalar_subquery_to_join.rs` (projection rewrite path).
