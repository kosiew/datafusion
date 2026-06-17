source: pr-22950_a
# Make file-statistics cache keys schema-aware

## Summary

DataFusion's file-statistics cache currently keys cached Parquet statistics by table scope and object path. However, the cached value is also dependent on the schema used to read the file. When the same physical Parquet file is read with different explicit schemas, cached statistics can be reused for an incompatible logical/file schema.

PR #22950 avoids one manifestation by keeping anonymous `ListingTable` statistics in a table-local cache instead of the shared session cache. That fixes the immediate regression, but the deeper invariant is still implicit: **file statistics are only reusable when the schema contract used to compute them is compatible with the schema contract of the later scan**.

## Current behavior

Relevant area:

- `datafusion/catalog-listing/src/table.rs`
- `datafusion/execution/src/cache/list_files_cache.rs` (`TableScopedPath`)
- `datafusion/execution/src/cache/cache_manager.rs` (`FileStatisticsCache`)
- `datafusion/execution/src/cache/file_statistics_cache.rs`

The statistics cache key is effectively:

```rust
TableScopedPath {
    table: Option<TableReference>,
    path: object_store::path::Path,
}
```

The cached value stores `CachedFileMetadata`, including `Statistics` and optional ordering. The key does not include the file/read schema used by `infer_stats_and_ordering`.

This is unsafe when a single Parquet path is read multiple ways, for example:

1. Read `data.parquet` with inferred schema: `(id, population)`.
2. Read the same path with an explicit wider schema: `(id, population, extra)`.
3. The old cached statistics can have too few column statistics or mismatched column positions for the second scan.

PR #22950 prevents shared-cache reuse for anonymous tables by using a local per-`ListingTable` statistics cache. That avoids cross-DataFrame contamination for `SessionContext::read_parquet`, but the cache key still does not encode the actual compatibility rule.

## Desired invariant

A cached file-statistics entry may be reused only if all schema-dependent attributes that affect statistics interpretation are compatible with the current scan.

At minimum, compatibility should account for:

- field count/order used by the statistics object
- field names and positions
- data types
- nullability when statistics semantics depend on injected/null-filled missing columns
- columns physically present vs columns synthesized due to an explicit wider schema
- any future schema adapter behavior that can alter the produced scan schema

The invariant should be explicit at the cache-key or cache-validation layer, not enforced indirectly by whether a table has a stable table reference.

## Why this matters

The current workaround is correct for anonymous reads, but it sacrifices cache reuse and leaves the schema dependency easy to miss in future code paths. Schema-aware cache keys would:

- preserve safe session-level reuse for anonymous reads with the same schema
- prevent accidental reuse across incompatible explicit schemas
- make the correctness contract visible where cache lookup happens
- reduce reliance on table-local cache scoping as a proxy for schema compatibility
- make future schema-adapter/statistics changes easier to review

## Proposed approach

Introduce a schema-aware component into the file-statistics cache key or validation path.

Possible design:

1. Define a compact schema fingerprint for statistics compatibility.
   - Include field names, data types, nullability, and order.
   - Consider whether field/schema metadata should be included. If metadata cannot affect statistics today, document that exclusion explicitly.
   - Consider distinguishing file-only schema from full table schema if partition/virtual/default columns are appended elsewhere.

2. Use the fingerprint when looking up and storing file statistics.
   - Either extend `TableScopedPath` for file-statistics use, or introduce a new key type specific to file statistics.
   - Avoid changing list-files cache keys if list-file results are not schema-dependent.

3. Keep table-reference scoping for invalidation.
   - Registered tables should still be invalidatable by table reference.
   - If a new key type is introduced, ensure `drop_table_entries` can still remove all entries for a table.

4. Revisit anonymous-table local caching.
   - With schema-aware keys, anonymous reads with identical schema could safely use the shared session cache again.
   - If local caching is retained, document why it is still needed beyond schema compatibility.

## Alternatives

### Keep PR #22950 behavior only

Pros:
- Simple and already fixes the known panic.
- Avoids changing cache API/key types.

Cons:
- Does not encode the true schema compatibility invariant.
- Loses shared-cache benefit for anonymous reads.
- Future contributors may reintroduce similar bugs in registered-table or schema-adapter paths.

### Validate cached statistics against current schema before reuse

Instead of changing the key, store the schema/fingerprint inside `CachedFileMetadata` and reject cache hits when incompatible.

Pros:
- Avoids widening the key type.
- Can preserve existing invalidation APIs.

Cons:
- Requires every cache hit path to remember validation.
- Cache may store multiple incompatible entries under the same path poorly unless the cache can hold variants per path.

### Disable statistics cache whenever an explicit schema is provided

Pros:
- Very simple.

Cons:
- Too conservative.
- Penalizes common safe cases where explicit schema is identical to inferred/file schema.
- Still does not make the underlying invariant explicit.

## Suggested tests

Add focused coverage around the file-statistics cache behavior:

1. Anonymous same-path reads with different explicit schemas do not reuse incompatible stats.
   - First read inferred schema.
   - Second read wider explicit schema with a missing nullable column.
   - Assert stats for projected missing column are correct, e.g. exact null count.

2. Anonymous same-path reads with the same explicit schema can reuse shared cache, if the implementation restores shared-cache reuse.
   - Assert cache entry count or use a test cache that records hits.

3. Registered table behavior remains safe.
   - Register a listing table with a stable table reference and explicit schema.
   - Replace/re-register with different schema over same path.
   - Assert invalidation or schema-aware keying prevents stale-stat reuse.

4. Cache invalidation still removes all entries for a table.
   - If key type changes, verify `drop_table_entries(Some(table_ref))` removes all schema variants.

5. Projection-sensitive stats remain aligned.
   - Project columns after a wider-schema read and assert `column_statistics.len()` and per-column stats match the projected schema.

## Risks and considerations

- Schema fingerprinting must be stable and cheap enough for scan planning paths.
- Including too much schema detail could reduce useful cache hits; including too little can preserve the bug.
- Key type changes may affect cache memory accounting via `DFHeapSize`.
- Public/internal API changes to `FileStatisticsCache` may require updates across custom cache implementations.
- Registered table invalidation must remain table-scoped even if multiple schema variants exist for the same table/path.

## Acceptance criteria

- File-statistics cache reuse is impossible across schema-incompatible reads of the same object path.
- Safe reuse remains possible for schema-compatible reads.
- The schema compatibility rule is documented near the cache key or cache lookup logic.
- Regression tests cover inferred schema followed by explicit wider schema with a missing nullable column.
- Existing table-reference cache invalidation semantics continue to work.
