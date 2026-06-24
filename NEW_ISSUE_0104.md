source: pr-21944_a
# Centralize information_schema catalog traversal

## Problem
`datafusion/catalog/src/information_schema.rs` repeats the same traversal shape in several virtual-table builders:

- `make_tables`
- `make_schemata`
- `make_views`
- `make_columns`

Each method separately walks:

```text
catalog_list.catalog_names()
  -> catalog_list.catalog(catalog_name)
  -> catalog.schema_names()
  -> catalog.schema(schema_name)
  -> schema.table_names()
  -> schema.table(...) or schema.table_type(...)
```

The traversal also repeats policy decisions, such as skipping the `information_schema` schema and tolerating schemas that disappear between `schema_names()` and `schema()`.

This duplication became more visible after catalog/schema lookup methods became fallible. Every call site now has to independently propagate errors with `?`, which makes future API or behavior changes easy to apply inconsistently.

## Why it matters
- Catalog traversal is a contract boundary: providers may be dynamic, remote, or fallible.
- Repeated traversal logic increases the chance that one information schema table handles missing schemas, errors, or `information_schema` filtering differently from another.
- Future catalog API changes will require mechanical edits across multiple methods, increasing regression risk.
- The current shape obscures the actual difference between builders: what row each virtual table emits.

## Invariant / desired behavior
All `information_schema` virtual tables should use one consistent traversal policy:

- catalog names come from `CatalogProviderList::catalog_names()`;
- catalog lookup failures for names returned by the list are treated consistently;
- `INFORMATION_SCHEMA` is skipped for user-schema traversal;
- schemas that disappear between `schema_names()` and `schema()` are skipped consistently;
- fallible `schema_names`, `schema`, `table_names`, `table`, and `table_type` calls propagate their errors without being hidden;
- each builder only owns row-emission logic specific to its virtual table.

## Proposed direction
Introduce one or more private traversal helpers on `InformationSchemaConfig` that encode the shared traversal policy and leave virtual-table-specific row construction to callbacks or small helper methods.

Possible shape:

- helper for catalog + non-information-schema schema traversal;
- helper for table-name traversal where the caller chooses whether to fetch `table_type` or full `table`;
- keep the helper private to `information_schema.rs` unless another module has a clear reuse need.

Avoid over-generalizing into a public abstraction. The goal is to centralize the information-schema traversal invariant, not create a new catalog iteration API.

## Scope
### In
- Refactor `InformationSchemaConfig` traversal used by `make_tables`, `make_schemata`, `make_views`, and `make_columns`.
- Preserve current behavior for missing schemas returned by `schema_names()` but not found by `schema()`.
- Preserve current error propagation from fallible catalog/schema/table provider methods.
- Preserve explicit addition of `INFORMATION_SCHEMA_TABLES` rows for `information_schema.tables`.
- Add or update focused tests if the helper changes observable error or missing-schema behavior.

### Out
- Changing public `CatalogProvider`, `SchemaProvider`, or `CatalogProviderList` APIs.
- Changing `information_schema` output columns, row ordering, or table definitions.
- Adding caching, concurrent traversal, or streaming behavior changes.
- Broad cleanup of builder structs unrelated to traversal.

## Acceptance criteria
- [ ] Shared catalog/schema traversal policy exists in one private helper or small helper family in `information_schema.rs`.
- [ ] `make_tables`, `make_schemata`, `make_views`, and `make_columns` no longer duplicate the full catalog -> schema -> table traversal loop.
- [ ] Fallible provider calls still use `?` at the traversal boundary and are not converted to silent skips except for the existing `Ok(None)` schema/table disappearance cases.
- [ ] Existing information schema tests pass without output changes.
- [ ] Any new helper has a narrow name that describes the traversal invariant, not a vague generic name such as `process` or `handle`.

## Tests / verification
- Run targeted catalog tests:
  - `cargo test -p datafusion-catalog information_schema`
- Run a focused core query test if available for information schema SQL output.
- If behavior around disappearing schemas/tables is clarified by the refactor, add a unit test with a custom provider that returns a name from `schema_names()` / `table_names()` and then returns `Ok(None)` from lookup.
- If error propagation is touched, add a unit test with a provider whose `schema_names`, `schema`, `table_names`, or `table` returns `Err`, and assert the information schema query returns that error.

## Notes / open questions
- The helper should probably distinguish schema-only traversal from table traversal. `make_schemata` does not need `table_names()`, while `make_tables`, `make_views`, and `make_columns` do.
- `make_tables` can use `table_type()` without fetching full table providers; keep that performance property.
- Preserve current row ordering unless a separate issue explicitly changes or documents ordering.
