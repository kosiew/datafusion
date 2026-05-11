# Semver-safe recursive CTE fix path

`cargo-semver-checks` is correct. The recursive CTE fix should not introduce public API breaks.

Keep the accepted recursive CTE contract:

- anchor/static field names and qualifiers define exposed output names
- output nullability is widened across static and recursive terms
- logical recursive CTE schema is authoritative for SQL planning
- physical children align to that logical schema
- metadata stays consistent across logical and physical schemas

## Verdict

The semver breaks are **not justified** for issue #22034.

This issue needs a schema-contract fix, not a breaking public API change.

Fix by preserving existing public APIs and adding any new capability as additive API with documented defaults.

## 1. Fix `RecursiveQuery` semver break

Problem:

```text
constructible_struct_adds_field: RecursiveQuery.schema
```

Do **not** add any field to public struct `RecursiveQuery`. It is externally constructible with struct literals, so adding either a public or private field is breaking.

Required change:

- Remove `pub schema: DFSchemaRef` from `datafusion/expr/src/logical_plan/plan.rs`.
- Keep the public struct shape unchanged:

```rust
pub struct RecursiveQuery {
    pub name: String,
    pub static_term: Arc<LogicalPlan>,
    pub recursive_term: Arc<LogicalPlan>,
    pub is_distinct: bool,
}
```

- Preserve widened recursive CTE schema by ensuring the stored children are aligned before constructing/storing `RecursiveQuery`.
- `LogicalPlan::schema()` for `LogicalPlan::RecursiveQuery` should continue to use the stored static term schema, e.g. `static_term.schema()`.

Suggested approach:

- Keep `RecursiveQuery::try_new(...)` with the same existing public signature.
- Inside `try_new`, compute the reconciled recursive CTE schema:
  - names / qualifiers from static term
  - compatible data types across static and recursive terms
  - nullability = `static_nullable || recursive_nullable`
  - metadata per the accepted contract
- If either child schema differs from the reconciled schema, wrap that child in a logical projection / alias / cast-equivalent plan so its schema becomes the reconciled schema.
  - Static term: preserve anchor/static names, qualifiers, and metadata; widen nullability as needed.
  - Recursive term: align names, qualifiers, nullability, and metadata to the reconciled schema.
- Return `RecursiveQuery { name, static_term, recursive_term, is_distinct }` with no stored `schema` field.

Important:

- Do not revert to anchor-only nullability.
- Do not remove SQL recursive-term replanning unless tests prove stale work-table nullability cannot regress. The replan was needed because recursive term optimization must see widened work-table nullability.

## 2. Fix `RecursiveQueryExec::try_new` semver break

Problem:

```text
method_parameter_count_changed: RecursiveQueryExec::try_new takes 4 parameters, now takes 5
```

Restore the old public constructor signature. Add the new schema-aware behavior as an additive constructor.

Required shape in `datafusion/physical-plan/src/recursive_query.rs`:

```rust
impl RecursiveQueryExec {
    /// Backward-compatible constructor.
    ///
    /// Uses a default output schema derived from the provided children. Prefer
    /// [`Self::try_new_with_schema`] when the logical recursive CTE schema is
    /// known, such as from the DataFusion physical planner.
    pub fn try_new(
        name: String,
        static_term: Arc<dyn ExecutionPlan>,
        recursive_term: Arc<dyn ExecutionPlan>,
        is_distinct: bool,
    ) -> Result<Self> {
        // Preferred default if physical derivation remains available:
        // let output_schema = recursive_query_output_schema(
        //     static_term.schema().as_ref(),
        //     recursive_term.schema().as_ref(),
        // )?;
        //
        // If REFACTOR_04 removes physical schema derivation, use the old
        // backward-compatible default instead:
        let output_schema = static_term.schema();

        Self::try_new_with_schema(
            name,
            static_term,
            recursive_term,
            output_schema,
            is_distinct,
        )
    }

    /// Creates a recursive query with an explicit output schema.
    ///
    /// This is the preferred constructor for planner-created recursive CTEs.
    /// The supplied `output_schema` is authoritative: both static and recursive
    /// children are aligned to it before execution.
    ///
    /// Recursive CTE schema contract:
    ///
    /// - names / qualifiers come from the anchor/static term
    /// - data types must be compatible across static and recursive terms
    /// - nullability is widened across both terms
    /// - metadata must remain consistent with the logical schema
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

Recommendation for the old `try_new` default:

1. If physical recursive schema derivation is still present and exactly matches the logical contract, use it to derive widened output schema from both children.
2. If following `REFACTOR_04.md` and removing duplicate physical derivation, use `static_term.schema()` for strict backward compatibility and document that callers with a known logical schema should call `try_new_with_schema`.

Do **not** change `try_new` arity.

Update call sites:

- `datafusion/core/src/physical_planner.rs` should call `RecursiveQueryExec::try_new_with_schema(...)` and pass the authoritative logical schema.
- Tests that need explicit widened output schema should call `try_new_with_schema(...)`.
- Legacy tests or external-style construction coverage can keep `try_new(...)`.

Use owned `SchemaRef` for `try_new_with_schema`; owned `Arc` is simple and avoids borrowing temporaries.

## 3. Keep the recursive-specific physical adapter

Do **not** add or keep broad public API solely for this bug.

Preferred direction from `REFACTOR_04.md`:

- no broad `SchemaAlignExec` for this issue
- no global `align_plan_to_schema` public helper for this issue
- keep any schema-rebind adapter local to recursive CTE execution
- validate count, type, nullability, field metadata, and schema metadata
- preserve `project_plan_to_schema` diagnostics when fallback cannot handle the mismatch

Rationale:

- A global schema adapter increases public contract surface.
- Recursive CTE only needs a narrow alignment rule.
- Logical schema should be the source of truth; physical planning should not recompute a second independent contract if avoidable.

## 4. Documentation to add

Add rustdoc near the constructors and/or helper:

- `try_new`: backward-compatible convenience/default constructor.
- `try_new_with_schema`: schema-aware constructor used by the planner.
- State that `output_schema` is authoritative for `try_new_with_schema`.
- State recursive CTE contract explicitly:
  - static term supplies output names / qualifiers
  - nullability is widened across static and recursive terms
  - both physical children are aligned to the logical schema
  - metadata mismatches are rejected or reconciled per the documented contract

Also add a short comment in SQL recursive CTE planning explaining the two-pass replan:

- first pass discovers recursive term schema
- planner computes widened work-table schema
- second pass replans recursive term so optimization sees the widened nullability

## 5. Tests to keep/add

Required regression coverage:

- recursive CTE with non-null anchor and nullable recursive term:

```sql
WITH RECURSIVE t AS (
  SELECT 0 AS n
  UNION ALL
  SELECT CAST(NULL AS INT) AS n FROM t WHERE n IS NOT NULL
)
SELECT * FROM t;
```

- existing SLT should use `0 AS level`, not `SUM(0) AS level`.
- direct `RecursiveQuery::try_new` tests for:
  - widened nullability
  - count mismatch rejected
  - type mismatch rejected
  - metadata contract preserved/rejected consistently
- `RecursiveQueryExec::try_new_with_schema` tests for:
  - explicit schema is authoritative
  - both children align to it
  - metadata mismatch is not silently accepted
- backward-compatible `RecursiveQueryExec::try_new` test still compiles and documents default behavior.

## 6. Validation commands

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
