created #23155
source: pr-22917_a
# Refactor: Make column-list aliasing a schema-position transform

## Summary

Column-list aliasing for table aliases and subqueries should be implemented as a positional schema transform, not by reconstructing column references from field display names.

Today `SqlToRel::apply_expr_alias` builds a projection by iterating over the current output fields and parsing each `field.name()` back into a `Column` with `Column::from_qualified_name_ignore_case`. This couples aliasing to SQL-name parsing and can misinterpret valid output field names that are not safe to reparse as SQL identifiers.

## Current behavior

Relevant code: `datafusion/sql/src/planner.rs`, `SqlToRel::apply_expr_alias`.

The current flow is:

1. Check the number of supplied aliases against `plan.schema().fields().len()`.
2. Clone `plan.schema().fields()`.
3. Build a projection with one expression per field.
4. For each field, create `Expr::Column(Column::from_qualified_name_ignore_case(field.name()))`.
5. Apply the requested alias with `.alias(...)`.

This treats `field.name()` as if it were parseable SQL column-reference text. That is not always true. A field name is an Arrow/DataFusion schema name, not necessarily an unambiguous SQL column reference.

## Problem

Column-list aliasing is semantically positional: the first alias renames the first output column, the second alias renames the second output column, and so on. It should not perform fresh name resolution based on a stringified field name.

Reparsing `field.name()` can break when an existing field name is valid as a schema field but ambiguous or differently meaningful as SQL column-reference text, such as:

- quoted identifiers containing separators, for example `"A.B"`;
- field names containing dots or other characters that `Column::from_qualified_name_ignore_case` treats as qualified-name syntax;
- duplicate unqualified names that are only distinguishable by relation or schema position;
- case-sensitive names where normalization and reparsing can change lookup behavior;
- future fields produced by expressions whose display name is not intended to be a resolvable input column expression.

Example shape:

```sql
CREATE TABLE t ("A.B" INT);
SELECT * FROM (SELECT * FROM t) AS q(x);
```

The output field named `A.B` should be aliased positionally to `x`. Instead, reparsing `A.B` can produce a qualified column reference and resolve as relation `A`, column `B`, or fail because no such qualified field exists.

## Desired behavior

Column-list aliasing should:

- preserve the resolved identity and order of the input plan's output columns;
- alias each output slot by position;
- avoid parsing field display names as SQL text;
- preserve relation / qualifier information when present;
- keep the existing alias-count validation and error message behavior;
- continue to normalize alias identifiers with the existing `ident_normalizer`.

## Proposed refactor

Introduce a small helper that converts a `DFSchema`'s output positions into `Expr::Column` expressions using resolved schema metadata, then zips those expressions with the supplied aliases.

Possible shape:

```rust
fn schema_columns_for_projection(schema: &DFSchema) -> Vec<Expr> {
    schema
        .iter()
        .map(|(qualifier, field)| {
            Expr::Column(Column {
                relation: qualifier.cloned(),
                name: field.name().clone(),
            })
        })
        .collect()
}
```

Use the actual DataFusion schema iterator / column helper that preserves the existing `TableReference` and field name exactly. If there is already an equivalent method such as `schema.columns()` or qualified-field accessors, prefer that over adding new public API.

Then `apply_expr_alias` becomes conceptually:

```rust
let columns = columns_from_schema_positions(plan.schema());
LogicalPlanBuilder::from(plan)
    .project(columns.into_iter().zip(idents).map(|(expr, ident)| {
        expr.alias(self.ident_normalizer.normalize(ident))
    }))?
    .build()
```

Keep the helper private to `datafusion/sql` unless another crate already has the right abstraction. This is a planner-internal refactor and should not widen public API without clear reuse.

## Scope

In scope:

- `SqlToRel::apply_expr_alias` in `datafusion/sql/src/planner.rs`;
- table alias column lists, for example `FROM t AS alias(c1, c2)`;
- subquery alias column lists, for example `FROM (SELECT ...) AS alias(c1, c2)`;
- tests for names that cannot safely be reparsed from `field.name()`.

Out of scope:

- changing general SQL identifier parsing;
- changing alias normalization rules;
- changing projection output naming outside column-list aliasing;
- broad `DFSchema` public API redesign unless an existing helper is insufficient.

## Test plan

Add SQLLogicTest coverage under `datafusion/sqllogictest/test_files/` for SQL-visible behavior.

Suggested cases:

```sql
CREATE TABLE t ("A.B" INT) AS VALUES (1);
SELECT * FROM (SELECT * FROM t) AS q(x);
----
1
```

Also cover direct table aliasing:

```sql
SELECT * FROM t AS q(x);
```

Additional useful cases:

- quoted mixed-case source name aliased by column list;
- two input columns with duplicate unqualified names after a join or projection, then alias by position;
- alias-count mismatch still returns the existing clear planner error.

If there are nearby Rust unit tests for `SqlToRel` alias planning, add a narrow unit test for the generated logical plan as well. Prefer SLT for end-to-end regression coverage because this is SQL-visible behavior.

Targeted validation:

```bash
cargo test -p datafusion-sqllogictest --test sqllogictests -- <new-or-existing-slt-file>
cargo test -p datafusion-sql
```

Adjust exact commands to the final test file location and existing test harness conventions.

## Risks and review notes

- Ensure the projection references the pre-alias input schema, not the post-table-alias relation name.
- Preserve relation qualifiers for fields that need qualified lookup.
- Do not use `field.name()` with `Column::from_qualified_name*` in the new path.
- Avoid changing behavior when no column aliases are supplied; `idents.is_empty()` should continue returning the original plan.
- Keep the alias-count mismatch path unchanged except for incidental formatting if needed.

## Expected outcome

Column-list aliasing becomes independent of string parsing and robust for all schema field names DataFusion can produce, including quoted names containing `.`. The planner code better matches the SQL semantics: aliases are applied by output position, not by resolving names a second time.

## Labels

- `sql`
- `planner`
- `refactor`
- `good first issue` if the maintainer agrees the helper path is clear
- `tests`

## Related context

This was identified as a high-impact, out-of-scope refactor during review of `apply_expr_alias`. The blocking bug was that reparsing field names still fails for quoted identifiers containing SQL identifier separators. The broader refactor is to make the implementation enforce the positional aliasing invariant directly.
