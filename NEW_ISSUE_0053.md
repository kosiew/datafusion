source: pr-22296_a
# Issue: Centralize array_resize size and capacity validation

## Summary
The validation logic for array_resize in datafusion/functions-nested/src/resize.rs is currently split across multiple helpers and call sites. This makes the full safety contract hard to see and easy to violate in future edits.

A single validation pipeline should be introduced to compute and return:
1. validated per-row target counts
2. validated cumulative output values length
3. validated growth metadata used for allocation and fill planning

This change is intended to reduce correctness regressions and make pre-allocation invariants explicit.

## Context
Current validation responsibilities are distributed:
- target_count validates per-row non-negative size and per-row offset fit
- general_list_resize accumulates output_values_len and max_extra
- validate_value_capacity checks value-byte capacity bounds

Because these checks are spread out, contributors can accidentally fix only one invariant (for example per-row count fit) while missing another (for example cumulative offset fit for List).

## Problem Statement
The code path currently requires reasoning across multiple functions to determine whether all pre-allocation invariants hold before constructing buffers. This creates two risks:

1. Partial correctness fixes
A patch can address one overflow path while leaving another path unguarded.

2. Maintainability cost
Reviewers and future contributors must manually reconstruct the full invariant chain across helpers.

## Why This Matters
The resize implementation handles memory sizing and offset construction for list arrays. Incomplete validation can lead to:
- late failures after expensive preparation
- increased chance of edge-case regressions
- fragile behavior when adding optimizations in fast and slow fill paths

Even when runtime errors eventually occur, delayed rejection is harder to reason about and test than explicit early validation.

## Proposed Solution
Introduce a single internal helper in resize.rs that performs all size-related pre-allocation validation and returns a compact validated plan used by downstream logic.

Suggested shape:
- Input: list array offsets, null bitmap, count array, value data type, offset type O
- Output:
  - per_row_counts: Vec<usize>
  - output_values_len: usize
  - max_extra: usize

The helper should enforce, in one place:
1. each requested size is non-negative
2. each row count fits offset type O
3. cumulative output_values_len fits offset type O
4. capacity bounds are valid for value buffers and fill buffers

Then general_list_resize should consume this validated plan rather than recomputing counts and totals independently.

## Acceptance Criteria
1. A single helper owns all size/offset pre-allocation invariants for array_resize list paths.
2. general_list_resize no longer duplicates invariant logic across loops and helper calls.
3. cumulative output_values_len is explicitly validated against offset type O before fill array or mutable buffer allocation.
4. Existing behavior for valid inputs is unchanged.
5. Regression tests cover:
   - per-row count overflow rejection
   - cumulative List overflow rejection across multiple non-null rows
   - no regression for valid large-but-legal inputs

## Test Plan
Add or update tests in datafusion/functions-nested/src/resize.rs:
- Keep existing single-row overflow coverage.
- Add multi-row cumulative overflow case for List where each row is valid individually but the total exceeds i32 offset range.
- Add at least one positive control showing large valid resize remains accepted.

## Scope
In scope:
- datafusion/functions-nested/src/resize.rs
- targeted unit tests in the same module

Out of scope:
- broad API redesign for array_resize
- unrelated list function refactors

## Risks and Mitigations
Risk: Refactor changes behavior in edge cases.
Mitigation: Add focused regression tests before and after refactor; preserve existing error messages where practical.

Risk: Performance overhead from materializing per-row counts.
Mitigation: Use a single pass to compute and store values needed by both fast and slow paths; avoid duplicate recomputation.

## Implementation Notes
- Keep the helper local/private to resize.rs.
- Prefer clear naming that communicates contract boundaries (for example validate_and_plan_resize).
- Return actionable execution errors when invariants fail.

## Definition of Done
- Refactor merged with tests passing.
- Invariant coverage is centralized and documented via helper structure.
- Reviewer can verify full pre-allocation contract by reading one function.