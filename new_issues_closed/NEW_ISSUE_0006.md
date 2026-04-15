# Issue Draft 02

## Title
Split dictionary min/max batch handling from generic complex-type fallback

## Summary
`min_batch` and `max_batch` in `datafusion/functions-aggregate-common/src/min_max.rs` currently route `DataType::Dictionary(_, _)` through `min_max_batch_generic`, shared with `Struct` and list-like complex types. This is correct for dictionary semantics, but dictionary behavior is now coupled to unrelated complex-type fallback logic.

This issue proposes introducing a dictionary-specific batch path that preserves correctness while creating a clear extension point for future dictionary min/max optimizations.

## Current State
- `DataType::Dictionary(_, _)` is grouped with:
  - `DataType::Struct(_)`
  - `DataType::List(_)`
  - `DataType::LargeList(_)`
  - `DataType::FixedSizeList(_, _)`
- All of these use `min_max_batch_generic`, which scans row-wise via `ScalarValue::try_from_array`.
- Row-wise scanning is required for dictionaries to avoid incorrect results from scanning dictionary values directly (unreferenced values and null-key semantics).

## Problem
- Dictionary-specific correctness constraints are not visible as a first-class code path.
- Any future dictionary optimization work is coupled to generic complex-type fallback behavior.
- Performance trade-offs for dictionary handling are harder to isolate and measure.

## Goals
- Keep dictionary min/max semantically correct.
- Introduce a dedicated dictionary helper as the single dispatch point for dictionary arrays.
- Preserve existing behavior for struct and list-like types.
- Make future dictionary optimization work independent from other complex types.

## Non-Goals
- No behavior change for `Struct`/`List`/`LargeList`/`FixedSizeList` min/max in this issue.
- No immediate specialization to Arrow dictionary kernels unless parity and correctness are proven.

## Proposed Approach
1. Add dedicated helper(s), for example:
   - `fn min_max_batch_dictionary(values: &ArrayRef, ordering: Ordering) -> Result<ScalarValue>`
2. Update dispatch in `min_batch` and `max_batch`:
   - Dictionary types route to dictionary helper.
   - Existing complex non-dictionary types continue using `min_max_batch_generic`.
3. Initially implement dictionary helper as correctness-preserving row-wise scan (same semantics as today).
4. Keep this helper narrowly scoped so later optimization can happen without touching unrelated fallback logic.

## Implementation Checklist
- [ ] Add `min_max_batch_dictionary` helper in `datafusion/functions-aggregate-common/src/min_max.rs`.
- [ ] Wire dictionary match arms in `min_batch` and `max_batch` to the new helper.
- [ ] Keep generic helper for non-dictionary complex types.
- [ ] Add tests that ensure dictionary semantics remain unchanged.

## Testing Plan
- Validate existing dictionary tests in `datafusion/functions-aggregate/src/min_max.rs`, including:
  - null handling
  - unreferenced dictionary value handling
  - multi-batch aggregation
  - non-`Int32` key type coverage (`Int8`)
  - float dictionary with NaN behavior
- Add/retain unit coverage in aggregate-common if dictionary helper internals gain specific branches.
- Run crate-scoped tests:
  - `cargo test -p datafusion-functions-aggregate min_max`
  - `cargo test -p datafusion-functions-aggregate-common min_max`

## Acceptance Criteria
- Dictionary dispatch is isolated behind a dedicated helper.
- Existing dictionary correctness tests pass without behavior changes.
- Complex non-dictionary fallback path remains intact and covered.
- Follow-up optimization can be implemented by modifying dictionary helper only.

## Performance Note
Dictionary handling currently prioritizes correctness through row-wise logical scanning. This issue should document that trade-off explicitly and leave room for a benchmark-backed optimization follow-up.

## Related Context
This follows review feedback that dictionary handling should remain semantically correct but have a dedicated path so future optimization work is not coupled to struct/list fallback behavior.
