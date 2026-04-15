not an issue anymore
source: pr-22143_a
# Issue: Consolidate array_replace Benchmarks to Avoid Split Ownership

## Summary
PR #22143 adds array_replace kernel benchmarks to `datafusion/functions-nested/benches/array_expression.rs`, but `datafusion/functions-nested/benches/array_replace.rs` already provides a dedicated, more comprehensive benchmark that covers the same int64 cases plus nested, string, boolean, and fixed-size-binary variants. This creates split ownership and duplicate maintenance burden.

## Problem Statement
**Current state:**
- `array_expression.rs` benchmarks int64 variants:
  - `array_replace_int64` (line 61)
  - `array_replace_n_int64` (line 78)
  - `array_replace_all_int64` (line 99)
- `array_replace.rs` benchmarks the same functions over:
  - Same int64 sizes and densities: `SIZES = [(4_000, 10), (10_000, 100), (10_000, 500)]`
  - Plus nested, string, boolean, and fixed-size-binary variants
  - Dedicated `bench_array_replace_int64()`, `bench_array_replace_n_int64()`, `bench_array_replace_all_int64()` functions

**Risks:**
1. **Split maintenance:** Updates to test sizes, constants, or fixture patterns must be synchronized across two files.
2. **Unclear ownership:** It's ambiguous which file is the "source of truth" for array_replace benchmarks.
3. **Duplicate logic:** ScalarFunctionArgs construction and list array generation patterns are repeated.
4. **Misleading name:** `array_expression.rs` name suggests broader expression benchmarks, not array_replace-specific tests.

## Proposed Solution
**Option A (Recommended):** Move the int64 array_replace benchmarks from `array_expression.rs` into the dedicated `array_replace.rs` file, removing the split.

**Option B:** Keep `array_expression.rs` as a lightweight integration benchmark (smaller dataset) and consolidate detailed kernel benchmarks into `array_replace.rs`.

**Option C:** Remove the `array_replace` benchmarks from `array_expression.rs` entirely, using the comprehensive `array_replace.rs` as the canonical benchmark.

## Acceptance Criteria
- [ ] Single authoritative benchmark file for array_replace functions
- [ ] No duplicate int64 array_replace benchmarks across files
- [ ] All three array_replace variants (replace, replace_n, replace_all) benchmarked with same test sizes/densities in one location
- [ ] Benchmark targets remain registered and executable (`cargo bench -p datafusion-functions-nested`)
- [ ] All existing benchmark results reproducible (same SEED, densities, sizes)

## Implementation Notes
- Review current benchmark group names in both files to ensure consistent naming
- Verify field names (`"array"` vs `"haystack"`) are normalized after consolidation
- Update Cargo.toml `[[bench]]` entries if needed
- Run full benchmark suite after consolidation to confirm correctness

## Related
- PR #22143: Original PR adding array_expression benchmarks
- File: `datafusion/functions-nested/benches/array_expression.rs` (lines 61–118)
- File: `datafusion/functions-nested/benches/array_replace.rs` (lines 61–180)
