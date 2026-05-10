# PR Review
owner: apache
repo: datafusion
pr_number: issue-22034

## Decision
- [ ] Approve
- [ ] Approve with suggestions
- [x] Request changes

## Blocking findings
1. file: datafusion/expr/src/logical_plan/plan.rs
   line: 2288
   side: RIGHT
   body: `recursive_query_schema` zips static and recursive fields without validating field count or type compatibility. `RecursiveQuery::try_new` is now used outside `LogicalPlanBuilder::to_recursive_query` (for example proto deserialization and tree rewrites), so callers can construct a recursive query whose stored schema silently truncates to the shorter input or advertises static field types even when the recursive child has different types. That breaks the logical plan schema contract. Please add count/type validation here (matching the existing builder/physical checks) and regression tests for direct `RecursiveQuery::try_new` / proto-invalid inputs.

2. file: datafusion/physical-plan/src/recursive_query.rs
   line: 407
   side: RIGHT
   body: The logical and physical recursive CTE schema derivations now disagree on metadata. Logical `recursive_query_schema` intersects field/schema metadata across static and recursive terms, but physical `recursive_query_output_schema` always keeps static field and schema metadata. With conflicting non-empty metadata, the logical plan can expose the intersected metadata while `RecursiveQueryExec` advertises the static metadata (or rejects aligning the recursive child), reintroducing logical-vs-physical schema mismatch at a schema boundary. Please make the physical derivation mirror the logical contract or pass the expected logical schema into physical planning, and add metadata coverage for recursive CTEs.

## Non-blocking suggestions
1. file: datafusion/sql/src/cte.rs
   line: 171
   side: RIGHT
   body: The replan logic is correct but subtle. Consider extracting the widened-schema replan into a small helper (for example `replan_recursive_term_with_schema`) or adding a short comment that the first plan is only used to discover the fixed output schema. This would make the two-pass planning invariant easier to maintain.

2. file: datafusion/physical-plan/src/common.rs
   line: 127
   side: RIGHT
   body: `align_plan_to_schema` intentionally ignores `project_plan_to_schema` errors and retries with `SchemaAlignExec`. Consider documenting that the projection error is only a path-selection signal, not the final diagnostic. Otherwise future maintainers may mistake the swallowed error for accidental broad error handling.

## High-impact refactor opportunities (out of scope)
1. title: Unify recursive CTE schema derivation contract
   context: Recursive output schemas are now derived separately in `datafusion/expr/src/logical_plan/plan.rs` and `datafusion/physical-plan/src/recursive_query.rs`.
   origin: introduced in this PR
   benefit: Reduces risk that metadata, field count, and type rules diverge between logical and physical layers. The current diff already shows different metadata behavior, which is a schema-contract risk.
   effort: Medium. Likely requires either passing the logical expected schema into physical planning or introducing a shared contract/helper at an appropriate lower-level crate boundary without adding an undesirable dependency.

## Follow-up actions
- Keep the UNION-like nullability direction from `PR_RESPOND_01.md`: anchor/static names, widened nullability, and both physical children aligned to the widened logical schema.
- Add direct logical/proto tests for mismatched recursive CTE column counts and types.
- Add recursive CTE metadata tests covering logical-vs-physical schema equality.
