created #22638
source: pr-21548_a
# [Refactor] Unify Spark Avg and Built-in Avg Group-Accumulator State Handling

## Summary

Spark Avg and built-in Avg currently maintain parallel group-accumulator logic for state conversion and merge behavior. Recent regressions showed that these code paths can drift, especially around null-state semantics used by convert_to_state and merge_batch. This issue proposes consolidating shared behavior so null/filter handling, state layout, and merge semantics are implemented once and reused by both implementations.

## Problem

The Spark implementation in datafusion/spark/src/function/aggregate/avg.rs now includes convert_to_state support that mirrors behavior already present in datafusion/functions-aggregate/src/average.rs. However, duplicate logic increases the chance of semantic divergence:

- null-state rows encoded by convert_to_state may be handled differently by merge paths
- shared edge-case fixes can land in one implementation but not the other
- reviewer burden grows because behavior must be compared manually across two files

This drift already surfaced in the supports_convert_to_state gap during Comet migration and again around null-aware merge correctness.

## Motivation

- Correctness: eliminate divergence risks in null/filter/state semantics for avg aggregation.
- Maintainability: centralize logic so bug fixes and optimizer-related behavior changes are implemented once.
- Reviewability: make Spark Avg behavior easy to verify against built-in Avg.
- Extensibility: create a reusable primitive for future aggregate implementations that need similar state conversion patterns.

## Goals

1. Share core group-accumulator state handling between Spark Avg and built-in Avg.
2. Ensure convert_to_state and merge_batch have identical null/filter semantics where intended.
3. Keep state field ordering and types explicit and consistent with each implementation's contracts.
4. Preserve public behavior for Spark compatibility and built-in SQL behavior.

## Non-Goals

- Reworking all aggregate UDF implementations in one pass.
- Changing Spark SQL semantic compatibility guarantees.
- Performing broad optimizer rewrites unrelated to Avg state handling.

## Proposed Design

### 1. Extract a shared helper/primitive

Introduce shared utilities in a common aggregate-support location (likely datafusion/functions-aggregate-common) that encapsulate:

- null-mask derivation from input nulls plus optional filter
- applying null masks consistently to state arrays (sum/count)
- null-aware merge accumulation over group indices

The helper should be generic over Arrow primitive type and count type where practical.

### 2. Rewire built-in Avg and Spark Avg to use shared logic

- Update convert_to_state in Spark Avg to reuse the same helper path currently used in built-in Avg (filtered_null_mask and set_nulls or a shared wrapper around them).
- Update merge_batch in Spark Avg to use null-aware accumulation semantics equivalent to built-in Avg groups accumulator behavior.
- Keep implementation-specific differences (type aliases, return field wiring, Spark-specific constraints) isolated around the shared core.

### 3. Keep state contracts explicit

Document and enforce expected state tuple ordering in both implementations:

- Spark Avg today: [sum, count]
- Built-in Avg today: [count, sum]

If ordering remains intentionally different, codify this in helper interfaces and tests to avoid accidental cross-use bugs.

## Detailed Work Items

1. Design and add a shared state-conversion helper API.
2. Design and add a shared null-aware group merge helper API.
3. Refactor Spark Avg to use helpers for convert_to_state and merge_batch.
4. Optionally align built-in Avg to the same abstraction boundary (without behavior change).
5. Add regression tests that round-trip convert_to_state -> merge_batch -> evaluate for:
   - nullable input rows
   - filtered input rows
   - mixed nullable+filtered rows
6. Add tests validating state ordering assumptions and merge correctness.
7. Add micro-benchmark comparison (if needed) between helper-based and previous custom code path for Spark Avg.

## Testing Strategy

### Unit tests

- Spark Avg groups accumulator:
  - null input rows are ignored in merge when represented as null state entries
  - filtered rows are ignored in merge when represented as null state entries
  - all-non-null path unchanged

- Built-in Avg groups accumulator:
  - no behavior regression for convert_to_state and merge_batch

### SQL logic tests

Add or extend sqllogictest coverage for avg over nullable inputs and filter-sensitive scenarios to validate end-to-end query behavior.

### Performance checks

- Run crate-scoped tests first:
  - cargo test -p datafusion-spark -- <relevant test name>
  - cargo test -p datafusion-functions-aggregate -- <relevant test name>
- If helper indirection changes hot-path behavior, run targeted benchmark comparison for avg group aggregation workloads.

## Risks and Mitigations

- Risk: abstraction introduces overhead in hot loops.
  - Mitigation: keep helper APIs monomorphizable/generic and benchmark before/after.

- Risk: accidental state-order mismatch between implementations.
  - Mitigation: explicit helper signatures and invariant tests for state tuple ordering.

- Risk: subtle behavior change in Spark compatibility semantics.
  - Mitigation: preserve Spark-specific behavior in adapter layer and add compatibility-focused tests.

## Acceptance Criteria

1. Spark Avg and built-in Avg both pass convert_to_state/merge/evaluate regression tests for null and filtered inputs.
2. Spark Avg no longer open-codes null-mask + merge logic where shared helper applies.
3. No behavior regression in existing Avg tests across datafusion-spark and functions-aggregate crates.
4. Code reviewers can trace shared null/filter/state logic to a single reusable implementation.
5. Optional benchmark evidence shows no material performance regression for representative avg group workloads.

## References

- Source review: PR_REVIEW_01.md
- Spark implementation: datafusion/spark/src/function/aggregate/avg.rs
- Built-in implementation: datafusion/functions-aggregate/src/average.rs
- Existing helper utilities: datafusion_functions_aggregate_common (filtered_null_mask, set_nulls)

## Suggested GitHub Labels

- refactor
- aggregate-functions
- spark-compat
- good-first-follow-up
