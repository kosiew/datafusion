# Tasks to Resolve `MinMaxBytesState` Performance Regression

## Root Cause Summary

`MinMaxBytesState::update_batch` enables the dense scratch table lazily based on
a per-batch density heuristic. Each time the loop discovers a new group id, it
invokes `enable_dense_for_batch`, which walks every `scratch_group_ids` entry to
copy sparse locations into the dense table.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L537-L658】【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L761-L815】

On dense workloads the heuristic fires repeatedly—once for almost every new
group—so `scratch_group_ids` is rescanned O(n²) times in a single batch. The
quadratic churn dwarfs the savings from dense scratch reuse, causing the massive
regressions reported by the dense Criterion benchmarks.

## Remediation Tasks

1. **Prevent repeated dense migrations within a batch.**
   * Track whether the dense scratch table has already been activated for the
     current epoch and skip subsequent `enable_dense_for_batch` calls, or defer
     the activation until after the loop so `scratch_group_ids` is walked only
     once.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L537-L658】【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L761-L815】
   * Ensure the fix still honours the sparse-path heuristic so we avoid eager
     dense allocations for sparse workloads.
2. **Cover dense-first workloads in tests/benchmarks.**
   * Extend the unit tests in `min_max_bytes.rs` (or add a new one) to assert
     that a single dense batch only triggers one dense activation and leaves the
     sparse map empty afterwards.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L861-L937】
   * Re-run the Criterion suite for sparse, dense, and monotonic cases to verify
     that dense regressions are eliminated without harming sparse gains.
3. **Document the heuristic contract.**
   * Update inline comments around the density toggle to explain the
     single-activation expectation and how the scratch vectors are reused so
     future refactors avoid reintroducing quadratic scans.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L522-L736】
