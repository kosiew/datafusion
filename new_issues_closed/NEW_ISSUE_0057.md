stale
source: pr-22239_a
# Issue Proposal: Centralize GetFieldFunc UDF Construction in simplify

## Summary
`GetFieldFunc` UDF construction is duplicated across simplify paths in `datafusion/functions/src/core/getfield.rs`. The same pattern appears in multiple locations via `Arc::new(ScalarUDF::new_from_impl(GetFieldFunc::new()))`.

## Background
Current code has at least two simplify-time construction sites for the same UDF object:
- Re-wrap path when remaining field path elements exist.
- Flattened-call reassembly path.

This is functionally correct, but duplication makes construction semantics harder to maintain consistently.

## Problem Statement
There is no single canonical helper for constructing the `get_field` UDF in simplify logic, increasing maintenance overhead and making future changes (for example cached singleton construction) harder to apply consistently.

## Why This Matters
Primary benefit today:
- Better readability and consistency.
- Clear single construction path.

Secondary potential benefit (optional follow-up):
- Reduced allocation churn if a cached singleton strategy is adopted (`OnceLock` or `thread_local`) and proven safe/desirable.

## Scope
In scope:
- Extract a local helper (for example `fn get_field_udf() -> Arc<ScalarUDF>`) and route all simplify-time construction through it.
- Update call sites in `getfield.rs` simplify flow.
- Preserve behavior exactly.

Optional in scope (if accepted by maintainers):
- Introduce cached construction in helper using `OnceLock` or `thread_local`.

Out of scope:
- Broad refactors of unrelated scalar UDF construction patterns.
- Behavior changes to simplify decisions.

## Proposed Approach
Phase 1 (safe, low-risk):
1. Add helper function local to module/test-visible scope as needed.
2. Replace duplicate inline constructors at simplify call sites.
3. Keep helper implementation equivalent to current semantics.

Phase 2 (optional optimization):
1. Evaluate cached singleton design.
2. Confirm `ScalarUDF` and dependencies are safe for shared static reuse.
3. Add guardrail tests and benchmarks if caching is introduced.

## Acceptance Criteria
- All previous call sites use the helper.
- Behavior and output plans remain unchanged.
- Existing tests pass without modification except for stylistic updates.
- If caching is implemented, add explicit tests/documentation showing no semantic change and thread-safe behavior assumptions.

## Validation Plan
- Run targeted tests for `getfield.rs` simplification behavior.
- Run crate-scoped tests for `datafusion/functions`.
- If caching is enabled, add stress/concurrency-oriented validation where practical.

## Risks and Mitigations
Risk: Overstating performance gains from helper extraction alone.
Mitigation: Document helper refactor as maintainability-first; treat caching as separate optional step.

Risk: Cached singleton introduces subtle lifecycle/threading assumptions.
Mitigation: Keep caching optional and gated by explicit safety review.

## Notes
This issue is intentionally small and suitable for a standalone cleanup PR. A caching optimization can be proposed afterward with dedicated evidence.