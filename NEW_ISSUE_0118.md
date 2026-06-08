source: pr-23651_a
# Decide canonical cache representation for extended statistics

## Problem

`StatisticsContext` computes both core `Statistics` and provider-supplied `ExtendedStatistics`, but its per-walk cache stores them in two separate maps:

- `statistics: HashMap<CacheKey, Arc<Statistics>>`
- `extensions: HashMap<CacheKey, Extensions>`

The current split is intentional: the core statistics cache is the hot path, and the extensions cache is only touched when statistics providers exist and return non-empty extensions. That keeps the no-provider walk cheap.

However, the split means one logical result is represented as two coordinated cache entries. The traversal has to maintain coherence across `cached_statistics`, `store_statistics`, `cached_extensions`, `store_extensions`, and `child_extended_stats`. `compute()` returns the base statistics directly, while `compute_extended()` reconstructs `ExtendedStatistics` by reading the base statistics plus any cached extensions.

This may be the right performance tradeoff, but the invariant is subtle and deserves an explicit decision backed by measurement.

## Why it matters

Statistics are used by optimizer rules and by plan display. Incorrect or incoherent cached statistics can change optimization choices or mislead users reading `EXPLAIN` output.

The current two-cache design creates maintainability risk:

- future changes may update base stats and forget matching extension behavior
- extension propagation depends on cached side data rather than one canonical value
- code readers must reason about when extensions exist, when they are absent, and whether absence means “no provider result” or “not cached yet”
- simplifying to one canonical cached value may improve correctness and comprehension, but could regress the no-provider hot path

The issue is not that the current implementation is wrong. The issue is that the cache representation encodes a performance/design tradeoff that should be deliberately validated and documented.

## Invariant / desired behavior

For each `(plan node, partition)` within one `StatisticsContext` walk, there should be one coherent cached logical statistics result.

If the system keeps separate base-statistics and extension caches, the code and tests should prove that the split cannot produce mismatched base stats and extensions.

If the system uses `Arc<ExtendedStatistics>` as the canonical cached value, `compute()` should return the same base stats from that canonical value without changing no-provider behavior beyond an accepted, measured cost.

Either design is acceptable if it has:

- a clear ownership boundary for cached results
- no stale or mismatched base/extension state
- documented performance rationale
- tests covering provider and no-provider paths

## Proposed direction

Treat this as a design decision with a small benchmark/profiling step before refactoring.

1. Measure current no-provider statistics walk overhead.
   - Use representative physical plans with no registered `StatisticsProvider`.
   - Include shallow and deep plans, and plans with shared subtrees if available.
   - Record allocations and runtime if possible.

2. Prototype a single canonical cache representation:

```rust,ignore
struct StatsCache {
    extended: HashMap<CacheKey, Arc<ExtendedStatistics>>,
}
```

Then have:

```rust,ignore
StatisticsContext::compute(...) -> Arc<Statistics> // from extended.base_arc()
StatisticsContext::compute_extended(...) -> Arc<ExtendedStatistics>
```

3. Compare complexity and performance against the current split-cache design.

4. Choose one:

- If the single-cache design has negligible overhead, prefer it for comprehension and coherence.
- If the split-cache design is measurably better for the no-provider hot path, keep it and document the invariant and benchmark rationale near `StatsCache`.

Avoid a purely aesthetic refactor that slows common statistics computation without evidence.

## Scope

### In

- Evaluate whether `StatisticsContext` should cache `Arc<ExtendedStatistics>` as the canonical value.
- Benchmark or otherwise measure no-provider overhead before changing representation.
- If changing representation, update `compute`, `compute_extended`, child stats assembly, and cache reset paths together.
- If keeping split caches, strengthen comments/tests that describe and prove cache coherence.
- Preserve current extension propagation semantics: extensions only exist for nodes where a provider computed them, and built-in fallback nodes do not automatically propagate child extensions.

### Out

- No changes to statistics estimation formulas.
- No changes to `StatisticsProvider` chain ordering.
- No changes to `ExecutionPlan::statistics_from_inputs` or `child_stats_requests` contracts unless required by the cache decision.
- No new provider features.
- No broad optimizer behavior changes.

## Acceptance criteria

- [ ] The PR includes a clear decision: single canonical `ExtendedStatistics` cache, or retained split cache with rationale.
- [ ] The chosen representation has one documented coherence invariant for `(plan node, partition)` cache entries.
- [ ] No-provider statistics walks remain within an accepted performance/allocation budget, documented in the PR.
- [ ] Provider-supplied extensions still reach parent providers only through the existing intended propagation path.
- [ ] `StatisticsContext::compute()` and `StatisticsContext::compute_extended()` return consistent base statistics for the same plan and partition.
- [ ] Cache reset clears all state needed to safely reuse a context after plan rewrites.

## Tests / verification

- Add or keep unit tests in `datafusion/physical-plan/src/statistics.rs` for:
  - no-provider walk uses built-in `statistics_from_inputs`
  - provider-computed base stats are visible through both `compute()` and `compute_extended()`
  - provider extensions are present for provider-handled nodes
  - extensions do not accidentally propagate through built-in fallback nodes
  - partition-specific cache entries do not collide with overall entries
  - `reset_cache()` removes both base and extension state

- Run focused tests:

```bash
cargo test -p datafusion-physical-plan statistics::tests --lib
```

- Run at least one broader optimizer/statistics check if behavior changes:

```bash
cargo test -p datafusion-physical-optimizer --lib
cargo test -p datafusion --lib statistics
```

Adjust filters to existing test names.

## Notes / open questions

- What overhead is acceptable for the default no-provider path?
- Is `Extensions` cloning cheap enough for a single-cache representation in common plans?
- Are there existing benchmarks that exercise `StatisticsContext`, or should a small targeted benchmark be added before changing the cache shape?
