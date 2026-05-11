# PR Review
owner: apache
repo: datafusion
pr_number: issue-22034

## Decision
- [ ] Approve
- [ ] Approve with suggestions
- [x] Request changes

## Blocking findings
1. file: datafusion/physical-plan/src/recursive_query.rs
   line: 506
   side: RIGHT
   body: `RecursiveSchemaRebindExec` silently allows field and schema metadata changes. `project_plan_to_schema` rejects metadata mismatches, but `align_recursive_plan_to_schema` falls back to this adapter for any projection error, and `validate_recursive_schema_rebind` only checks column count, data type, and nullability. Under the current accepted contract (anchor names, widened nullability, logical schema authoritative), metadata should still be consistent between logical and physical schemas. This fallback can mask metadata drift and make physical execution advertise the logical schema while accepting children with incompatible metadata. Please validate/reject field and schema metadata mismatches in the rebind path, or avoid the fallback for metadata-related projection errors.

## Non-blocking suggestions
1. file: datafusion/physical-plan/src/recursive_query.rs
   line: 377
   side: RIGHT
   body: `align_recursive_plan_to_schema` swallows every `project_plan_to_schema` error before trying the recursive-specific fallback. Consider only falling back for the cases the local rebind is meant to handle, or preserve the projection error in the final diagnostic. That would avoid hiding more actionable errors from `project_plan_to_schema`.

2. file: datafusion/physical-plan/src/recursive_query.rs
   line: 377
   side: RIGHT
   body: The helper name is generic relative to the invariant it enforces. A name like `align_recursive_child_to_logical_schema` would make it clearer that the logical recursive CTE schema is authoritative and this is not a general physical-plan adapter.

## High-impact refactor opportunities (out of scope)
1. title: Centralize recursive CTE schema contract documentation
   context: Schema behavior is described across `RecursiveQuery::try_new`, SQL two-pass recursive planning comments, physical child alignment, and SLT/physical tests.
   origin: introduced in this PR
   benefit: The PR history has moved from static-output nullability to UNION-like nullability. Consolidating the contract text in one helper/module-level comment would reduce future regressions and make it easier to verify that logical planning, physical planning, and tests assert the same invariant.
   effort: Small to medium. Mostly documentation/comment consolidation plus possibly renaming the physical helper.

## Follow-up actions
- Keep the current accepted contract explicit: anchor/static names, widened nullability, logical recursive CTE schema authoritative, physical children aligned to that logical schema.
- Add metadata validation/rejection to `RecursiveSchemaRebindExec` or constrain fallback use so metadata mismatches are not silently accepted.
- Consider improving fallback diagnostics and helper naming.
