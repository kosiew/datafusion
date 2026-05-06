# Justification for Deviating from `PR_RESPOND_01.md`

The newer recursive CTE issue changes the technical requirement from the narrower plan in `PR_RESPOND_01.md`.

## Why the deviation is justified

- Recursive CTE execution must preserve the static/anchor term schema as the declared CTE output schema.
- Existing valid SQL such as `0 AS level` must remain valid and should not need to be rewritten as `SUM(0) AS level` merely to widen nullability.
- The recursive term can be planned with nullable expressions even when the anchor/static term is non-null.
- Therefore, rejecting nullable recursive input when the expected static schema is non-null would preserve the earlier feedback but fail the newer issue requirement.
- A higher-level `align_plan_to_schema(input, expected_schema)` helper is justified because it guarantees `aligned_plan.schema() == expected_schema`.
- `SchemaAlignExec` is justified for cases `ProjectionExec` cannot express exactly, such as nullable input aligned to a non-null expected field.
- Batch schema rebinding is acceptable when contained inside the explicit adapter plan node, because the alignment remains visible in the physical plan and is not hidden inside `RecursiveQueryStream`.
- Metadata and type safety remain important: column count, data type, field metadata, and schema metadata mismatches should still error.

## What should still hold

- `RecursiveQueryExec::schema()` should equal `static_term.schema()`.
- `RecursiveQueryExec::recursive_term().schema()` after construction should equal `static_term.schema()`.
- Rename-only alignment can use `ProjectionExec`.
- Nullable input to non-null expected output should use the explicit adapter.
- Unsafe metadata or type changes should be rejected.
- The recursive stream itself should not perform hidden schema patching.

## SLT restoration

Restoring the recursive CTE SQLLogicTest from `SUM(0) AS level` back to `0 AS level` is a valid goal.

It proves the fix preserves the declared/static recursive CTE schema and removes the SQL workaround introduced by the regression. If the current PR is scoped only to the reusable helper, this restoration may belong in the recursive CTE follow-up PR, but it is valid and important coverage for the behavior fix.
