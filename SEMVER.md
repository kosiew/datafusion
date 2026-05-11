# Semver fix instructions

`cargo-semver-checks` is correct. Fix the API break without changing the accepted recursive CTE contract:

- anchor/static field names
- nullability widened across static and recursive terms
- logical recursive CTE schema is authoritative
- physical children align to that logical schema

## 1. Fix `RecursiveQuery` semver break

Problem:

```text
constructible_struct_adds_field: RecursiveQuery.schema
```

Do **not** add any field to public struct `RecursiveQuery`. It is externally constructible with struct literals, so adding either a public or private field is breaking.

Required change:

- Remove `pub schema: DFSchemaRef` from `datafusion/expr/src/logical_plan/plan.rs`.
- Restore `LogicalPlan::schema()` for `LogicalPlan::RecursiveQuery` to return the static term schema, e.g. `static_term.schema()`.
- Preserve widened recursive CTE schema by making the stored `static_term` itself have the widened schema before constructing/storing `RecursiveQuery`.

Suggested approach:

- Keep `RecursiveQuery::try_new(...)` as a semver-safe helper with the same 4 args.
- Inside `try_new`, compute the reconciled recursive CTE schema with current `recursive_query_schema(static, recursive)`.
- If either child schema differs from that reconciled schema, wrap that child in a logical projection/alias/cast-equivalent plan so its schema becomes the reconciled schema.
  - Static term: preserve anchor/static field names, qualifiers, metadata; widen nullability as needed.
  - Recursive term: align names/qualifiers/nullability to the reconciled schema.
- Then return `RecursiveQuery { name, static_term, recursive_term, is_distinct }` with no stored schema field.

Important:

- Do not revert to anchor-only nullability.
- Do not remove SQL recursive-term replanning unless tests prove the stale work-table nullability hang cannot return. The replan was needed because recursive term optimization must see widened work-table nullability.

## 2. Fix `RecursiveQueryExec::try_new` semver break

Problem:

```text
method_parameter_count_changed: RecursiveQueryExec::try_new takes 4 parameters, now takes 5
```

Restore the old public constructor signature and add a new constructor for the new behavior.

Required change in `datafusion/physical-plan/src/recursive_query.rs`:

```rust
impl RecursiveQueryExec {
    /// Backward-compatible constructor. Uses the static term schema as output schema.
    pub fn try_new(
        name: String,
        static_term: Arc<dyn ExecutionPlan>,
        recursive_term: Arc<dyn ExecutionPlan>,
        is_distinct: bool,
    ) -> Result<Self> {
        let output_schema = static_term.schema();
        Self::try_new_with_schema(
            name,
            static_term,
            recursive_term,
            output_schema,
            is_distinct,
        )
    }

    /// Constructor used by the planner when the logical recursive CTE schema is known.
    pub fn try_new_with_schema(
        name: String,
        static_term: Arc<dyn ExecutionPlan>,
        recursive_term: Arc<dyn ExecutionPlan>,
        output_schema: SchemaRef,
        is_distinct: bool,
    ) -> Result<Self> {
        // current 5-arg implementation body
    }
}
```

Then update call sites that know the logical schema:

- `datafusion/core/src/physical_planner.rs` should call `RecursiveQueryExec::try_new_with_schema(...)` and pass `Arc::clone(schema.inner())`.
- Tests in `recursive_query.rs` that need explicit output schema should call `try_new_with_schema(...)`.
- Any legacy tests expecting old behavior can keep `try_new(...)`.

Use owned `SchemaRef` for `try_new_with_schema` unless there is a strong reason to take `&SchemaRef`; owned `Arc` is simpler and avoids borrowing a temporary.

## 3. Keep the local recursive adapter

Do **not** make `RecursiveSchemaRebindExec` generic for this semver fix.

Keep the current `REFACTOR_04.md` direction:

- no broad `SchemaAlignExec`
- no global `align_plan_to_schema`
- recursive CTE-local adapter only
- validate count/type/metadata
- preserve `project_plan_to_schema` diagnostics when fallback cannot handle the case

## 4. Tests to run

After changes:

```bash
cargo fmt -- datafusion/expr/src/logical_plan/plan.rs datafusion/physical-plan/src/recursive_query.rs datafusion/core/src/physical_planner.rs datafusion/sql/src/cte.rs
cargo test -p datafusion-expr recursive_query --quiet
cargo test -p datafusion-physical-plan recursive_query_exec --quiet
cargo test -p datafusion-physical-plan common::tests:: --quiet
cargo check -p datafusion --quiet
cargo test -p datafusion-sqllogictest --test sqllogictests cte
```

Then rerun the semver job/check that failed.
