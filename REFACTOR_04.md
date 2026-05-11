# REFACTOR_04: Simpler Recursive CTE Schema Fix

## Summary

Yes, there is a simpler way to fix issue #22034.

The current diff derives recursive CTE schema in both logical and physical layers and adds a broad `SchemaAlignExec` adapter. This works toward the desired behavior, but it is more complex than necessary and leaves room for logical/physical schema drift.

## Main findings

### 1. Physical planner still ignores declared logical schema

File: `datafusion/core/src/physical_planner.rs`

`RecursiveQuery` now stores the desired logical schema, but physical planning drops it:

```rust
RecursiveQueryExec::try_new(name, static_term, recursive_term, is_distinct)
```

Then `RecursiveQueryExec` recomputes schema from physical children. That can diverge from the logical schema if physical planning changes nullability or metadata.

Better: pass `RecursiveQuery.schema` into `RecursiveQueryExec` and make it the single physical output contract.

### 2. `SchemaAlignExec` is too broad for this issue

File: `datafusion/physical-plan/src/common.rs`

The new global adapter can rebind field names, nullability, and metadata. That is bigger than needed for recursive CTEs and introduces more contract surface area, especially around nullability narrowing.

This issue only needs recursive CTE output to use:

- anchor/static field names
- compatible data types
- widened nullability
- consistent metadata

It does not require a general-purpose physical plan adapter that can re-advertise arbitrary schema differences.

### 3. Duplicate schema derivation

Schema derivation exists in two places:

- `recursive_query_schema` in `datafusion/expr/src/logical_plan/plan.rs`
- `recursive_query_output_schema` in `datafusion/physical-plan/src/recursive_query.rs`

Even if they currently match, this duplication can drift. The logical recursive CTE schema should be authoritative.

## Simpler fix

Keep:

- `RecursiveQuery { schema }`
- `RecursiveQuery::try_new(...)`
- SQL two-pass replan in `datafusion/sql/src/cte.rs`
  - This is needed to avoid stale work-table nullability assumptions, e.g. `WHERE n IS NOT NULL` being optimized using anchor-only non-null schema.

Simplify physical planning:

### 1. Pass declared logical schema to physical exec

In `datafusion/core/src/physical_planner.rs`:

```rust
LogicalPlan::RecursiveQuery(RecursiveQuery {
    name,
    schema,
    is_distinct,
    ..
}) => {
    let [static_term, recursive_term] = children.two()?;
    Arc::new(RecursiveQueryExec::try_new(
        name.clone(),
        static_term,
        recursive_term,
        Arc::clone(schema.inner()),
        *is_distinct,
    )?)
}
```

### 2. Change `RecursiveQueryExec::try_new`

Accept the declared output schema:

```rust
pub fn try_new(
    name: String,
    static_term: Arc<dyn ExecutionPlan>,
    recursive_term: Arc<dyn ExecutionPlan>,
    output_schema: SchemaRef,
    is_distinct: bool,
) -> Result<Self>
```

### 3. Remove physical schema recomputation

Remove `recursive_query_output_schema` from `datafusion/physical-plan/src/recursive_query.rs`.

The logical schema becomes the source of truth.

### 4. Align children to the declared schema

```rust
let recursive_term = assign_work_table(recursive_term, &work_table)?;
let static_term = project_plan_to_schema(static_term, &output_schema)?;
let recursive_term = project_plan_to_schema(recursive_term, &output_schema)?;
```

If metadata intersection makes `project_plan_to_schema` fail, choose one small fix:

- simplest: preserve static metadata in logical recursive CTE schema too; or
- add a recursive-CTE-local schema rebinder rather than a new global `SchemaAlignExec`.

## Recommendation

Do not add broad `SchemaAlignExec` in `common.rs` for this bug.

Make the logical recursive CTE schema authoritative and pass it into physical planning. This reduces complexity, removes duplicate schema derivation, and enforces the desired invariant directly.
