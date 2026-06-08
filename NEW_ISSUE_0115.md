source: schema_caching-01-21554a
# Add generation-keyed Parquet pruning setup cache reuse for dynamic predicates

## Problem

`ParquetPruningSetupCache` currently disables reuse whenever the scan predicate contains a `DynamicFilterPhysicalExpr`.

That conservative rule is necessary today because row-group pruning setup stores a `PruningPredicate`, and `PruningPredicate::try_new` snapshots dynamic filter state when it is built. If a cached setup were reused after the dynamic filter changed, later files could prune using an obsolete predicate snapshot and return wrong results.

The safe bypass fixes correctness, but it also leaves performance on the table. Dynamic predicates whose snapshot generation has not changed still rebuild the same adapted projection, adapted predicate, and row-group pruning predicate for each same-schema file.

## Why it matters

Dynamic filters appear in important scan paths such as top-k / limit-style pruning and join-derived runtime filters. These scans can touch many Parquet files with identical physical schemas. Rebuilding the pruning setup for each file adds avoidable CPU cost exactly in the path this cache is meant to optimize.

The system should not choose between correctness and reuse. Reuse is valid when the cache key represents the dynamic-filter snapshot used to build the cached `PruningPredicate`.

## Invariant / desired behavior

A cached Parquet pruning setup may be reused only when all inputs that affect the adapted projection, adapted predicate, and row-group `PruningPredicate` are equivalent.

For dynamic predicates:

- Cached setup must not be reused across different dynamic-filter snapshot generations.
- Cached setup may be reused when predicate identity, projection identity, logical schema, physical schema, adapter cache-safety, and dynamic-filter generation are unchanged.
- If the generation cannot be represented safely, the code must fall back to the existing no-cache behavior.
- `input_file_name()` projections and literal-column rewrites remain per-file and must not become cacheable through this change.

## Proposed direction

Add a cache-key-safe dynamic generation component at the pruning setup cache boundary.

Suggested implementation shape:

1. Replace the current boolean-only dynamic predicate check in `is_pruning_setup_reusable` with a small private classification result, for example:
   - reusable static predicate
   - reusable dynamic predicate with generation key
   - not reusable
2. Extend `ParquetPruningSetupCacheKey` with the dynamic generation key when present.
3. Derive that key from existing dynamic-filter tracking / `PhysicalExpr::snapshot_generation` mechanisms rather than adding a second traversal model.
4. Keep the default fallback conservative: if any dynamic subexpression cannot provide a clear generation value, bypass the cache.
5. Keep this localized to `datafusion/datasource-parquet/src/opener/mod.rs` unless a tiny helper is needed in dynamic-filter tracking APIs.

The safety rule should be visible where the cache key is built. Avoid scattered checks around later pruning execution.

## Scope

### In

- Dynamic-predicate cache eligibility and key construction for `ParquetPruningSetupCache`
- Tests for unchanged dynamic generation reuse
- Tests for changed dynamic generation isolation
- Documentation comments on why generation is part of the cache key

### Out

- Page pruning predicate caching
- Row-filter split/support-analysis caching
- Public scan API changes
- New metrics/counters
- Broad dynamic-filter architecture redesign
- Making custom `PhysicalExprAdapterFactory` implementations reusable by default

## Acceptance criteria

- [ ] Same-schema files with a dynamic predicate and unchanged generation reuse the cached pruning setup.
- [ ] Updating a dynamic filter between files causes a cache miss or bypass, so the later file uses the current predicate snapshot.
- [ ] Static predicate cache behavior remains unchanged.
- [ ] Non-cache-safe adapter factories still bypass the cache.
- [ ] Literal-column replacement still disables reuse.
- [ ] `input_file_name()` projections still disable reuse.
- [ ] The generation-key logic is documented at the cache-key boundary.

## Tests / verification

- Add opener tests under `datafusion/datasource-parquet/src/opener/mod.rs`:
  - dynamic predicate, same physical schema, unchanged generation: assert adapter/setup creation count shows reuse
  - dynamic predicate update between files: assert correct returned rows and cache miss/isolation
  - unsupported/unclear dynamic generation: assert no cache entry is populated, if such a case exists
- Run:
  - `cargo test -p datafusion-datasource-parquet test_pruning_setup_cache --features parquet_encryption`
  - `cargo test -p datafusion-datasource-parquet opener::test:: --features parquet_encryption`

## Notes / open questions

- Confirm whether the existing dynamic-filter generation value covers all dynamic subexpressions in a compound predicate, or whether the cache key needs a combined generation vector/hash.
- Confirm generation wraparound behavior is acceptable within one scan-local cache lifetime.
- If generation lookup requires walking the expression tree per file, verify the traversal cost is smaller than the setup work being cached.
