created #23072
source: pr-22950_a
# Make file-statistics cache keys schema-aware

## Summary

DataFusion's file-statistics cache should encode the schema compatibility invariant used to interpret cached statistics. Today the cache is keyed by table scope and object path, but statistics are also dependent on the schema used by the scan. This means the same physical Parquet file can produce incompatible statistics when read with different explicit schemas.

The immediate PR fix should likely bypass file-statistics caching for anonymous `SchemaSource::Specified` scans. That is the narrow safe behavior. However, the more complete refactor is to make the file-statistics cache schema-aware so safe cache reuse remains possible even for anonymous explicit-schema reads.

## Background

The motivating case is schema evolution / CDC:

- an old Parquet file physically contains columns such as `city` and `population`
- a new logical schema adds an optional column such as `census_url`
- `SessionContext::read_parquet` is called with the new schema explicitly
- the missing optional column is expected to be produced as all NULLs

If statistics for the same file path were previously cached using the inferred old schema, reusing those stats for the wider explicit schema can misalign column statistics. The cached stats may have fewer columns or different positions than the current projected scan schema expects.

## Current state

Relevant files:

- `datafusion/catalog-listing/src/table.rs`
- `datafusion/catalog-listing/src/config.rs`
- `datafusion/execution/src/cache/list_files_cache.rs`
- `datafusion/execution/src/cache/cache_manager.rs`
- `datafusion/execution/src/cache/file_statistics_cache.rs`

Current cache key shape is path/table scoped via `TableScopedPath`:

```rust
pub struct TableScopedPath {
    pub table: Option<TableReference>,
    pub path: Path,
}
```

This is sufficient to distinguish registered tables and file paths, but not sufficient to distinguish scans of the same path using different schema contracts.

The narrow PR-level fix should avoid cache reuse for anonymous specified-schema scans. That prevents stale stats from being reused but sacrifices potentially safe cache reuse for repeated anonymous reads using the same explicit schema.

## Problem

The real invariant is not "anonymous tables need a separate cache" or "explicit schemas can never cache".

The real invariant is:

> Cached file statistics may be reused only when the schema-dependent meaning of the statistics is compatible with the current scan schema.

Without encoding this invariant in the cache key or cache entry validation, future changes can accidentally reintroduce stale-stat reuse across incompatible schemas.

Examples of unsafe reuse:

1. Same path, inferred schema first, explicit wider schema second.
2. Same path, two different explicit schemas with different field counts.
3. Same path, same field count but different field order.
4. Same path, same names but incompatible data types.
5. Same path, schema adapter behavior changes how missing columns are synthesized.

## Proposed refactor

Introduce a schema-aware file-statistics cache key or validation layer.

### Option A: new file-statistics-specific key

Create a key specific to file statistics, for example:

```rust
struct FileStatisticsCacheKey {
    table: Option<TableReference>,
    path: Path,
    schema_fingerprint: SchemaFingerprint,
}
```

`SchemaFingerprint` should include the scan/file schema attributes that determine the layout and interpretation of `Statistics.column_statistics`.

Likely fields to include:

- field count
- field order
- field names
- data types
- nullability

Fields to evaluate/document:

- field metadata
- schema metadata
- extension metadata
- partition/virtual/default columns vs physically stored file columns

The key should remain table-scoped so registered-table invalidation can remove all variants for a table.

### Option B: store schema fingerprint in cached metadata

Keep the existing external key, but store a schema fingerprint in `CachedFileMetadata` and reject cache hits when the current scan schema is incompatible.

This is less ideal if multiple schema variants need to coexist for the same path, because a single path key cannot naturally hold multiple valid entries unless the cached value becomes a variant map.

### Recommended direction

Prefer Option A unless API compatibility makes it too disruptive. A statistics-specific key makes the invariant explicit and avoids overloading `TableScopedPath`, which is also used by list-files caching where schema is not part of the result.

## Design questions

1. Should the fingerprint be based on `file_schema`, full `table_schema`, or projected scan schema?
   - The statistics are initially inferred for file columns, then combined/projected later.
   - The key should cover the schema used by `infer_stats_and_ordering` and any missing-column/null-fill behavior that affects stats.

2. Should field metadata be included?
   - If metadata cannot affect statistics today, excluding it may improve cache hit rate.
   - If excluded, document why it is safe.

3. How should partition columns be handled?
   - Partition statistics are derived from path values and appended separately.
   - Avoid mixing partition-derived stats with file-column stats in the fingerprint unless required.

4. How should registered-table invalidation work?
   - `drop_table_entries(Some(table_ref))` should remove all schema variants for that table.

5. Can anonymous specified-schema reads safely use the shared cache once schema-aware keying exists?
   - Yes, if the key fully captures the schema compatibility invariant.

## Testing plan

Add regression tests covering both safety and reuse.

### Safety tests

1. Inferred schema then explicit wider schema for same anonymous Parquet path.
   - First read: infer old schema and collect stats.
   - Second read: use explicit wider schema with a missing nullable column.
   - Assert projected stats include the missing column and exact null count.

2. Two explicit schemas for same path with different field counts.
   - Assert no incompatible stats reuse.

3. Same field count but different field order.
   - Assert stats align with output schema positions.

4. Type mismatch or coercion case, if supported.
   - Assert either safe recomputation or clear rejection.

### Reuse tests

1. Same anonymous explicit schema read twice.
   - Assert cache hit or stable cache entry count if test hooks allow.

2. Registered table with same schema read repeatedly.
   - Assert existing shared-cache behavior remains.

3. Registered table invalidation.
   - Register/drop or replace a listing table.
   - Assert all schema-key variants for that table are removed.

### Resource tests

1. Cache memory accounting includes schema key size.
2. Cache limit remains global; repeated anonymous reads do not create per-table caches that bypass `file_statistics_cache_limit`.

## Risks

- Too broad a fingerprint may reduce cache hits unnecessarily.
- Too narrow a fingerprint can preserve correctness bugs.
- Changing cache key types may affect custom `FileStatisticsCache` implementations.
- Memory accounting must include the schema fingerprint to keep eviction accurate.
- Invalidation APIs must continue to support table-scoped removal.

## Acceptance criteria

- File-statistics cache reuse is impossible across schema-incompatible scans of the same path.
- Anonymous explicit-schema reads can safely use the shared cache when the schema fingerprint matches.
- Global `datafusion.runtime.file_statistics_cache_limit` remains meaningful; no per-anonymous-table cache workaround is needed.
- Registered-table cache invalidation removes all schema variants for that table.
- Tests cover inferred old schema followed by explicit wider evolved schema with missing nullable column stats.
- The schema compatibility rule is documented near the cache key or cache lookup code.
