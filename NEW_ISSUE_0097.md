source: pr-22960_a
# Refactor: Build a reusable struct-access path tree for Parquet row-filter planning

## Summary

Parquet row-filter planning currently handles struct field access paths in multiple independent shapes. `build_filter_schema` / `prune_struct_type` now group struct paths while constructing the projected Arrow schema, but `resolve_struct_field_leaves` still handles each `StructFieldAccess` separately by building a prefix and scanning every Parquet leaf column.

Introduce a reusable struct-access path tree (trie) for row-filter planning, and use it to drive both:

1. Parquet projection-mask leaf selection.
2. Arrow projected-schema pruning.

This would encode the core invariant once: the set of struct paths required by a row-filter must produce a projection mask and projected schema that describe the same pruned struct shape.

## Current behavior

Relevant file: `datafusion/datasource-parquet/src/row_filter.rs`

Current flow:

- `PushdownChecker` collects flat `StructFieldAccess { root_index, field_path }` entries.
- `resolve_struct_field_leaves` iterates every access path, builds a string prefix, then scans all Parquet leaf columns to find matching paths.
- `build_filter_schema` groups paths by root column.
- `prune_struct_type` groups paths by next struct-field name at each schema level.

After PR `2024b71258`, schema pruning avoids repeated scans, but leaf resolution still has the older flat-path scan pattern.

## Problem

The same logical information — required struct access paths — is interpreted in separate places with separate algorithms:

- Projection-mask construction scans Parquet leaves per access path.
- Projected-schema construction recursively groups paths by Arrow struct fields.

This causes avoidable repeated work for predicates with many struct accesses, especially with shared nested prefixes, for example:

```sql
WHERE s['outer']['a'] > 10
  AND s['outer']['b'] < 20
  AND s['outer']['inner']['c'] IS NOT NULL
```

It also increases maintenance risk: projection-mask selection and schema pruning can drift, even though they must stay contract-compatible for row-filter evaluation.

## Proposed refactor

Create a private local representation such as:

```rust
struct StructAccessTree {
    roots: BTreeMap<usize, StructAccessNode>,
}

struct StructAccessNode {
    children: BTreeMap<String, StructAccessNode>,
    selected_here: bool,
}
```

Exact shape can differ; key goal is to represent shared prefixes once.

Use the tree to:

1. Build grouped paths once from `Vec<StructFieldAccess>`.
2. Resolve Parquet leaves by walking/comparing against the tree instead of scanning all leaves for each access path.
3. Prune Arrow `DataType::Struct` fields by walking the same tree.
4. Keep whole-column references as an override: if a root appears in `regular_indices`, preserve the full field type and include all root leaves.

## Expected benefits

- Less repeated iterator work and fewer temporary path/prefix allocations.
- One shared abstraction for the row-filter struct-access invariant.
- Lower risk that projection mask and projected schema disagree.
- Easier tests for nested shared-prefix access patterns.
- Cleaner future extensions for struct/map/list pruning behavior.

## Correctness invariant

For any pushed-down filter expression with struct field accesses:

- The Parquet projection mask must include exactly the leaf columns needed to evaluate those accesses.
- The projected Arrow schema passed to row-filter evaluation must match the schema produced by the Parquet reader for that projection.
- If a struct root is also referenced as a whole column, pruning must be disabled for that root and the full struct must be preserved.

## Suggested implementation steps

1. Add a private helper in `row_filter.rs` to build the access tree from `&[StructFieldAccess]`.
2. Replace `paths_by_root` / `paths_by_field` ad hoc grouping with tree traversal in `build_filter_schema` and `prune_struct_type`.
3. Update `resolve_struct_field_leaves` to use the same tree when matching Parquet column paths.
4. Keep behavior unchanged for non-struct columns, missing columns, map columns, and list predicates.
5. Add focused tests for shared-prefix nested struct paths.

## Test coverage to add

Add unit tests in `datafusion/datasource-parquet/src/row_filter.rs` covering:

1. Multiple fields under the same struct root:
   - `s['a']`
   - `s['b']`
   - projection mask includes only `s.a` and `s.b` leaves.
   - projected schema is `Struct{a, b}`.

2. Multiple nested paths sharing a prefix:
   - `s['outer']['a']`
   - `s['outer']['b']`
   - projection mask includes only those nested leaves.
   - projected schema is `Struct{outer: Struct{a, b}}`.

3. Mixed whole-root and nested access:
   - `s` plus `s['outer']['a']`.
   - full `s` schema and all `s` leaves are preserved.

4. End-to-end row-filter evaluation for nested shared-prefix predicates, asserting returned row count and pushdown metrics.

## Scope

Local to `datafusion/datasource-parquet/src/row_filter.rs` unless a more general nested-path helper already exists and is clearly reusable.

Avoid widening public API. Keep helper private unless another module has an immediate, concrete need.

## Effort

Medium.

The refactor is conceptually local, but it touches both projection-mask and projected-schema construction, so tests should verify end-to-end row-filter behavior and exact schema/projection compatibility.
