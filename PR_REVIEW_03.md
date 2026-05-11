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
   line: 2313
   side: RIGHT
   body: This still violates the stated core invariant. The PR summary/non-goals say recursive CTE output must preserve the static/anchor schema, including nullability, and must not widen output nullability to match recursive expressions. However `recursive_query_schema` stores `static_nullable || recursive_nullable` as the logical recursive CTE schema, and physical planning now treats that logical schema as authoritative. Thus `SELECT 0 AS level ... recursive nullable level` still declares a nullable recursive CTE output instead of the static non-null schema. Please make `RecursiveQuery` expose the static/anchor schema as the declared CTE schema for this PR direction, and only adapt children to that schema.

2. file: datafusion/physical-plan/src/recursive_query.rs
   line: 514
   side: RIGHT
   body: The local adapter rejects the nullable-recursive-input to non-null-static-output case that this PR is supposed to fix. If `RecursiveQueryExec` is changed to use the static schema as required, a recursive child with nullable `MIN(rs.level) + 1` aligned to non-null `0 AS level` will hit this `input_field.is_nullable() && !output_field.is_nullable()` error. The requested behavior/tests explicitly include "nullable recursive input aligned to non-null static/logical output". The adapter must allow this contract when the recursive CTE output schema is the declared static schema, while still validating count and data type.

3. file: datafusion/physical-plan/src/recursive_query.rs
   line: 506
   side: RIGHT
   body: `RecursiveSchemaRebindExec` silently allows field and schema metadata changes. `project_plan_to_schema` rejects metadata mismatches, but `align_recursive_plan_to_schema` falls back to this adapter for any projection error, and `validate_recursive_schema_rebind` only checks column count, type, and nullability. That contradicts the adapter requirements to reject unsafe field metadata and schema metadata changes, and can mask logical/physical schema metadata drift. Please either preserve the static metadata contract so projection suffices, or explicitly validate/reject metadata mismatches in the rebind path.

4. file: datafusion/sqllogictest/test_files/cte.slt
   line: 1303
   side: RIGHT
   body: The added regression test codifies the opposite semantics from this PR description: it states recursive CTE nullability is union-like and expects a nullable recursive output from `SELECT 0 AS n UNION ALL SELECT CAST(NULL AS INT)`. Under the requested invariant, this is not the behavior being fixed; the key regression is restoring `0 AS level` without widening the declared CTE schema. Please replace or move this test to match the chosen contract, and add the missing physical test that `RecursiveQueryExec::schema() == static_term.schema()` while the recursive child is aligned to that schema.

## Non-blocking suggestions
1. file: datafusion/physical-plan/src/recursive_query.rs
   line: 377
   side: RIGHT
   body: Once the contract is corrected, consider making the fallback name more explicit about the recursive-only invariant, e.g. `align_recursive_child_to_declared_schema`, and avoid swallowing the `project_plan_to_schema` error unless the fallback is intentionally handling that specific incompatibility. This would make the control flow and diagnostics easier to maintain.

## High-impact refactor opportunities (out of scope)
1. title: Centralize the recursive CTE schema contract
   context: Schema behavior is described in SQL planning comments, `RecursiveQuery::try_new`, physical child alignment, and SLT/physical tests.
   origin: introduced in this PR
   benefit: The current diff shows the contract can drift between "static schema" and "union-like nullability". A single helper or clearly documented constructor contract would reduce future regressions and make tests assert one invariant end-to-end.
   effort: Medium. Requires choosing the contract, updating logical schema construction, physical planning, and regression tests consistently.

## Follow-up actions
- Decide and enforce one contract: this PR text says static/anchor schema including nullability.
- Update `RecursiveQuery` logical schema derivation to match that contract.
- Make `RecursiveQueryExec` align recursive children to the declared static schema and allow nullable input to non-null declared output where required.
- Add/adjust tests for `RecursiveQueryExec::schema() == static_term.schema()`, recursive child schema alignment, metadata rejection, and restored `0 AS level` SLT coverage.
