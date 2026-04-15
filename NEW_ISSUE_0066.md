source: unify-avg-22638a
# Shared null-aware Avg merge helper

## Summary
Spark Avg and built-in Avg currently implement merge semantics through different code paths. Spark Avg has custom merge logic in `datafusion/spark/src/function/aggregate/avg.rs`, while built-in Avg relies on shared null-aware accumulation behavior (via `NullState::accumulate`) in `datafusion/functions-aggregate/src/average.rs`.

This split increases the risk of semantic drift, especially around null and filter handling during `merge_batch`. We should introduce a shared helper for Avg state merging that enforces null-aware and filter-aware behavior consistently, while preserving caller-controlled state layout and count type.

## Problem Statement
The `GroupsAccumulator::merge_batch` contract requires `opt_filter` semantics to match `update_batch` semantics. Today:
- Built-in Avg path already centralizes null/filter-aware behavior.
- Spark Avg path uses custom merge loop logic.

When these paths diverge, correctness can regress in one implementation without immediate visibility in the other. This is already a known risk area in the current review context.

## Motivation
A shared merge helper would:
- Encode the null/filter invariant once.
- Reduce duplicate merge-loop logic.
- Lower maintenance burden for Avg behavior updates.
- Keep Spark and built-in Avg aligned under the same contract-level semantics.

## Scope
### In scope
- Add a reusable helper for null-aware, filter-aware Avg merge behavior.
- Use the helper in Spark Avg merge path and/or built-in Avg merge path where appropriate.
- Preserve state ordering and caller conventions:
  - Spark state order: `[sum, count]`
  - Spark count type: `Int64`
  - Built-in count type: `UInt64`
- Add targeted tests covering null/filter handling and state conversion + merge round-trips.

### Out of scope
- Redesigning the full aggregate trait surface.
- Changing Spark Avg output/state schema contracts.
- Broad refactors unrelated to Avg merge semantics.

## Proposed Design
Introduce a helper with responsibilities limited to the merge loop:
- Inputs:
  - sum array + count array for partial states
  - group indices
  - optional filter
  - mutable output state vectors
- Behavior:
  - Skip rows where state is null/invalid for merge semantics.
  - Apply filter semantics exactly like `update_batch` (`true` merges, `false` and `NULL` skip).
  - Merge count and sum atomically into destination group.
- Generic constraints:
  - Support different count physical types (`Int64` and `UInt64`).
  - Keep state slot mapping caller-controlled to avoid hardcoding layout assumptions outside local call sites.

Likely placement options:
1. Shared aggregate-common helper crate (preferred if no layering issues).
2. Existing average module helper extracted and reused by Spark.

Selection criteria:
- Minimal layering friction.
- Clear ownership and discoverability.
- No circular dependency introduction.

## Acceptance Criteria
- Spark Avg and built-in Avg both honor identical null/filter semantics for `merge_batch`.
- Shared helper is used so merge-loop invariant is centralized.
- Spark keeps `[sum, count]` state order and `Int64` count type.
- Built-in Avg keeps `UInt64` count type behavior unchanged.
- Regression tests pass for:
  - `opt_filter = Some(BooleanArray::from(vec![Some(true), Some(false), None]))`
  - Null values in partial state inputs
  - Convert-to-state then merge round-trip cases for both Spark and built-in contracts

## Test Plan
1. Add Spark Avg merge test with mixed filter values (`true`, `false`, `NULL`) and validate only `true` rows merge.
2. Add Spark state round-trip test: input -> `convert_to_state` -> `merge_batch` and validate sum/count.
3. Add/extend built-in Avg tests to ensure helper adoption preserves current behavior.
4. Run targeted crates:
   - `cargo test -p datafusion-spark`
   - `cargo test -p datafusion-functions-aggregate`
   - `cargo test -p datafusion-functions-aggregate-common`

## Risks and Mitigations
- Risk: Type-generic helper becomes over-abstracted.
  - Mitigation: Keep helper narrow and focused on merge semantics only.
- Risk: State-order assumptions leak into helper.
  - Mitigation: Pass explicit array/state references from callers; avoid implicit slot indices in shared code.
- Risk: Behavioral regressions in one path during migration.
  - Mitigation: Add mirrored tests for Spark and built-in contracts before/with refactor.

## Related Context
- Review context identified this as a high-impact refactor opportunity to reduce divergence and enforce `merge_batch` contract consistency.
- The issue intersects correctness, maintainability, and regression prevention for aggregate behavior.
