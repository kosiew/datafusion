created #22056
source: pr-21667_a
# GitHub Issue Draft

## Title
[Optimizer] Consolidate repeated filter-rebuild patterns in PushDownFilter

## Body
```markdown
# [Optimizer] Consolidate repeated filter-rebuild patterns in PushDownFilter

## Summary
`PushDownFilter` rebuilds filter nodes in many branch-specific paths. This duplication increases maintenance cost and makes rule invariants harder to keep consistent.

## Background and Motivation
Recent optimizer work introduced `make_filter` to centralize filter construction, but push/keep/reinsert logic remains duplicated across plan variants.

Today, each branch tends to repeat the same shape:
1. split predicates into pushable/keep sets
2. rebuild one or more child filters
3. reinsert kept predicates above
4. return transformed plan

This pattern appears in `Sort`, `Distinct`, `Repartition`, `Projection`, `Union`, `Extension`, `Aggregate`, `Window`, `Unnest`, and join-related paths.

## Problem Statement
Filter reconstruction mechanics are duplicated across many branches in `datafusion/optimizer/src/push_down_filter.rs`.

Concrete symptoms:
- repeated `make_filter(...)` and `Arc::new(...)` call patterns
- repeated split/push/keep control flow
- repeated branch-local child replacement logic

Representative references:
- `datafusion/optimizer/src/push_down_filter.rs:837` (Repartition)
- `datafusion/optimizer/src/push_down_filter.rs:875` (Projection)
- `datafusion/optimizer/src/push_down_filter.rs:892` (Unnest)
- `datafusion/optimizer/src/push_down_filter.rs:968` (Union)
- `datafusion/optimizer/src/push_down_filter.rs:993` (Aggregate)
- `datafusion/optimizer/src/push_down_filter.rs:1060` (Window)
- `datafusion/optimizer/src/push_down_filter.rs:1215` (Extension)
- `datafusion/optimizer/src/push_down_filter.rs:1275` (Extension child rebuild)

## Why This Matters
- Correctness risk: duplicated rewrite logic makes subtle behavior divergence more likely across plan-node branches.
- Invariant drift: filter reconstruction/reinsertion conventions are harder to enforce when spread across many sites.
- Review burden: future changes require auditing many branches for equivalent behavior.
- Evolvability: adding new single-input node rewrites repeats boilerplate and invites copy-paste defects.

## Proposed Direction
Introduce a small, internal unary-node helper for filter reconstruction within `PushDownFilter`.

Candidate design (illustrative):
- helper that takes:
  - the current plan node (single-input)
  - push predicate (optional)
  - keep predicate (optional)
  - optional expression replacement closure (for projection-like rewrites)
- helper returns transformed plan with consistent semantics:
  - push predicate below node when allowed
  - keep predicate above node when needed
  - preserve no-op behavior when neither exists

Potential decomposition:
- `push_predicate_below_unary_node(...) -> Result<Transformed<LogicalPlan>>`
- `rebuild_filter_if_needed(predicate, input) -> LogicalPlan` (or `Result<LogicalPlan>` when fallible)
- `split_push_keep(...)` utility for repeated partition logic

## Scope
In scope:
- refactor `PushDownFilter` internals to reduce duplicated filter-rebuild logic
- preserve current behavior and rule ordering
- keep filter-construction behavior centralized and documented

Out of scope:
- changing optimizer semantics
- broad optimizer framework refactors outside `push_down_filter.rs`
- unrelated performance tuning in other rules

## Acceptance Criteria
- duplicated filter reconstruction logic in `push_down_filter.rs` is materially reduced
- existing behavior remains unchanged for all covered plan-node branches
- helper API clearly documents assumptions and semantics for filter reconstruction and reinsertion
- code readability improves (shorter per-branch rewrite blocks, less boilerplate)
- existing tests pass

## Validation Plan
1. Run focused optimizer tests for push-down behavior.
2. Run sqllogictests that cover predicate pushdown-sensitive plans.
3. Verify no plan-shape regressions in existing push-down test cases.
4. (Optional) run planner benchmark `sql_planner_extended` to ensure no performance regression.

## Risks and Mitigations
- Risk: helper abstraction hides branch-specific edge cases.
  - Mitigation: keep helper narrowly scoped; retain branch-specific logic only where semantics differ.
- Risk: accidental behavior change in predicate placement.
  - Mitigation: preserve existing tests and add targeted assertions for push/keep outcomes.

## Implementation Notes
- Prefer mechanical refactor steps and small commits.
- Keep temporary refactor helpers private to the module.
- Follow existing DataFusion optimizer style and error-handling patterns.
```
