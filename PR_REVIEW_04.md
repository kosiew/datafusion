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
   line: 2332
   side: RIGHT
   body: This changes the logical recursive CTE schema to `static_nullable || recursive_nullable`, which is the opposite of the stated invariant for this issue. The reported regression is that the physical recursive CTE output was widened to nullable while the logical/static schema stayed non-null (`0 AS level`), causing the logical-vs-physical schema check to fail. This fix should preserve the declared/static schema and align physical children to it; instead it makes the logical/static term nullable by wrapping the static term in a Projection. Please keep the recursive query schema anchored to the static term and do the nullable-recursive-child adaptation in the physical alignment layer.

2. file: datafusion/physical-plan/src/recursive_query.rs
   line: 141
   side: RIGHT
   body: `try_new_with_schema` aligns children with `project_plan_to_schema`, but that helper rejects nullable input -> non-null expected output. That is the exact required case for this issue (`0 AS level` static schema, nullable recursive expression). The PR does not add/use the requested adapter path that can advertise the expected static schema while rebinding batches after count/type/metadata validation, so the physical-layer invariant `recursive_term().schema() == static_term.schema()` is still not enforceable for the target case without widening the expected schema.

3. file: datafusion/sqllogictest/test_files/cte.slt
   line: 1303
   side: RIGHT
   body: The new regression test codifies the opposite behavior from the issue: it says recursive CTE nullability is union-like and expects `NULL` to be emitted from a CTE whose anchor is `SELECT 0 AS n`. This does not test that `0 AS level` remains valid with the declared/static non-null schema preserved; it instead validates the schema-widening workaround. Please replace/add coverage for nullable recursive physical input aligned to the non-null static/logical output schema.

## Non-blocking suggestions
No non-blocking suggestions. The issues above are contract-level blockers, not polish.

## High-impact refactor opportunities (out of scope)
No separate high-impact refactor opportunities identified. The main maintainability risk is the same as the blocker: schema ownership is split between logical widening and physical projection rather than a single plan-time alignment contract. That should be addressed as part of the fix, not deferred as an out-of-scope refactor.

## Follow-up actions
- Preserve recursive CTE logical output schema as the static/anchor schema for this issue.
- Add/use a physical alignment adapter for nullable recursive child -> non-null expected schema, with count/type/metadata/schema-metadata validation.
- Update tests to assert `RecursiveQueryExec::schema()` and `recursive_term().schema()` equal the static schema for the target `0 AS level` case.

Focused validation run:
- `cargo test -p datafusion-physical-plan recursive_query_exec` passed
- `cargo test -p datafusion-physical-plan project_plan_to_schema` passed
- `cargo test -p datafusion-sqllogictest --test sqllogictests -- cte` passed
