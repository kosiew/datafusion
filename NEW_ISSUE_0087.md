source: pr-22857_a
# Centralize Parquet page-index loading policy

## Summary

DataFusion currently has more than one place that can decide whether Parquet page indexes (`ColumnIndex` + `OffsetIndex`) are loaded. This makes it easy for one path to preserve the intended pruning order while another path bypasses it.

The page-index loading decision should be centralized behind a single explicit policy boundary used by the Parquet opener, cached reader, and metadata cache code.

## Current problem

The Parquet opener wants this ordering:

1. Load footer metadata without page indexes.
2. Build row-group pruning predicates.
3. Prune row groups with row-group statistics.
4. Load page indexes only if at least one surviving row group is not fully matched by row-group statistics.
5. Apply page-index pruning.

However, the default cached-reader path can independently request page indexes during metadata fetch:

- `datafusion/datasource-parquet/src/opener/mod.rs`
  - `PreparedParquetOpen::load()` creates `ArrowReaderOptions` with `PageIndexPolicy::Skip`.
  - Later, `should_load_page_index()` decides whether page indexes are needed after row-group pruning.
- `datafusion/datasource-parquet/src/reader.rs`
  - `CachedParquetFileReader::get_metadata()` receives `ArrowReaderOptions`, but does not propagate the requested page-index policy to `DFParquetMetadata`.
- `datafusion/datasource-parquet/src/metadata.rs`
  - `DFParquetMetadata::fetch_metadata()` forces `PageIndexPolicy::Optional` when a metadata cache exists, to make cached metadata complete.

This means the opener can say “skip page indexes for now,” while the cached metadata layer can still load them immediately.

## Why this matters

The page index is not inline in the Parquet footer and may require extra I/O. If row-group statistics prove all surviving row groups fully match the predicate, page-index pruning cannot remove more rows or pages. Loading page indexes in that case is wasted I/O and can hurt scan latency, especially on remote object stores.

Keeping this policy split across layers also makes future changes risky:

- cache behavior can accidentally override pruning-order invariants;
- tests that cover only the opener state machine may miss default `ParquetSource` behavior;
- new metadata readers or cache modes may repeat the same bypass.

## Proposed refactor

Introduce one explicit page-index metadata loading policy and thread it through all metadata-loading paths.

Possible shape:

```rust
enum ParquetMetadataLoadMode {
    FooterOnly,
    FooterWithOptionalPageIndex,
}
```

or reuse/thread `PageIndexPolicy` directly if that is sufficient.

Apply it consistently through:

- `PreparedParquetOpen::load()`;
- `CachedParquetFileReader::get_metadata()`;
- `DFParquetMetadata::fetch_metadata()`;
- metadata cache insert/get logic, if cache completeness depends on whether page indexes were loaded.

The key invariant should be explicit:

> The initial metadata load for scans must not fetch page indexes unless the caller explicitly requested a metadata mode that permits it. Page indexes are loaded only after row-group statistics pruning proves they may still prune surviving row groups.

## Design considerations

### Cache completeness

The current cache path loads full metadata, including page indexes, so cached metadata can satisfy later page-index needs without another object-store read. A centralized policy should make cache entry completeness explicit.

Possible approaches:

1. Store whether cached metadata includes page indexes.
2. Permit footer-only cache entries and later upgrade them with page indexes.
3. Keep a separate cache key or cache metadata flag for page-index-inclusive entries.
4. Keep full-metadata caching only for metadata-only APIs, but scan opener initial loads use footer-only metadata.

Avoid a design where the presence of a cache silently changes scan I/O behavior.

### API surface

Prefer a small internal abstraction local to `datafusion-datasource-parquet` unless broader datasource APIs need to express this. Avoid exposing parquet-rs internals more widely than needed.

### Backwards compatibility

This should be a behavior/performance refactor, not a user-visible semantic change. Query results must stay identical.

## Suggested implementation steps

1. Add an internal metadata load policy type, or thread `PageIndexPolicy` through `DFParquetMetadata`.
2. Update `CachedParquetFileReader::get_metadata()` to read the requested policy from `ArrowReaderOptions` and pass it to `DFParquetMetadata`.
3. Update `DFParquetMetadata::fetch_metadata()` so cache presence does not unconditionally force `PageIndexPolicy::Optional` for scan opener loads.
4. Make cache entries record whether page indexes are present, or otherwise prevent footer-only entries from being mistaken for page-index-complete metadata.
5. Update opener code to call the shared policy boundary instead of relying on local comments/invariants.
6. Add tests that exercise the default cached-reader / `ParquetSource` path, not only direct opener helper paths.

## Test plan

Add regression tests that observe page-index I/O directly.

Suggested tests:

1. **Initial scan metadata load skips page index**
   - Use a Parquet file with page indexes.
   - Enable page index and metadata cache.
   - Use a predicate such as `IS NOT NULL` on a non-null column, so row-group statistics fully match all surviving row groups.
   - Assert that no page-index ranges are requested, using a counting object store or test reader.

2. **Page index still loads when needed**
   - Use a predicate that row-group stats cannot fully prove for all surviving row groups but page indexes can prune.
   - Assert page-index I/O occurs and pruning still works.

3. **Cache does not mask policy**
   - Run a scan once with footer-only metadata.
   - Run a second scan with the same cache.
   - Assert the cache does not cause eager page-index loading unless the scan reaches the explicit page-index loading phase.

## Acceptance criteria

- There is one obvious internal policy path controlling whether page indexes are fetched.
- `CachedParquetFileReader` respects the caller's page-index loading policy.
- Metadata cache behavior cannot silently override scan pruning order.
- Tests fail if page-index I/O occurs before row-group statistics pruning in the fully-matched case.
- Query results and existing page-index pruning behavior are unchanged.

## Impact

- Performance: avoids unnecessary page-index I/O on scans fully resolved by row-group statistics.
- Maintainability: reduces duplicated policy decisions across opener, reader, and metadata cache layers.
- Correctness confidence: makes the pruning-order invariant testable end to end.
