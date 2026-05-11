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
   line: 2263
   side: RIGHT
   body: The core invariant is not implemented: this constructor now widens the recursive CTE logical/static schema (`static_nullable || recursive_nullable`) and wraps the static term in a Projection with that widened schema. The issue requires preserving the declared/static schema, including non-nullability from `0 AS level`, and aligning the recursive term to that schema. This also makes the restored SQL pass by changing the logical schema rather than fixing physical child alignment.

2. file: datafusion/physical-plan/src/recursive_query.rs
   line: 141
   side: RIGHT
   body: `try_new_with_schema` still uses `project_plan_to_schema` for the recursive child. That helper explicitly rejects nullability narrowing, so it cannot satisfy the required nullable-recursive-input -> non-null-static-output case. The requested plan-time adapter (`align_plan_to_schema` / `SchemaAlignExec`) is still missing; instead the code relies on the logical layer widening the expected schema, which masks the original bug.

3. file: datafusion/sqllogictest/test_files/cte.slt
   line: 1303
   side: RIGHT
   body: The new regression test codifies the opposite contract: it states recursive CTE nullability is union-like and expects a nullable output (`NULL`). This does not protect the reported regression where the recursive CTE must keep the anchor/static non-null schema while accepting a nullable recursive physical child.

## Non-blocking suggestions
No non-blocking suggestions; the blockers are contract-level and should be fixed before polish.

## High-impact refactor opportunities (out of scope)
1. title: Centralize recursive CTE schema compatibility checks
   context: Similar count/type/nullability/metadata schema construction now exists in logical `recursive_query_schema` and physical `recursive_query_output_schema`.
   origin: introduced in this PR
   benefit: Avoids logical/physical drift for the same recursive CTE contract and makes future schema contract changes testable in one place.
   effort: Medium: extract a shared helper or clearly split validation vs physical alignment responsibilities.

2. title: Remove schema-discovery replanning from SQL CTE planning
   context: `sql/src/cte.rs` now builds an initial recursive query only to discover a widened schema, then replans the recursive term with a different work-table schema.
   origin: introduced in this PR
   benefit: Reduces planner complexity and avoids making optimizer behavior depend on a speculative first plan. A single authoritative declared schema plus physical alignment is easier to reason about.
   effort: Medium after the adapter-based fix exists.

## Follow-up actions
- Preserve `RecursiveQuery` logical schema as the static/anchor schema.
- Add a plan-time alignment adapter that can advertise the expected schema for nullable-input -> non-null-expected output while validating count, type, field metadata, and schema metadata.
- Update tests to assert `RecursiveQueryExec::schema()` and `recursive_term().schema()` equal the static/logical schema for the `0 AS level` case.
