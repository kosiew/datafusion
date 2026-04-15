partially stale and not high impact
source: pr-22143_a
# Issue: Consolidate Nested Array Benchmark Fixtures and Constants

## Summary
Multiple benchmark files in `datafusion/functions-nested/benches/` (particularly `array_expression.rs` and `array_replace.rs`) duplicate test constants, list array generation helpers, and `ScalarFunctionArgs` construction patterns. This creates drift risk and maintenance burden. A shared fixture module would reduce duplication and ensure consistent test data across all nested array benchmarks.

## Problem Statement
**Duplicated constants across benchmarks:**
- `SEED = 42` (appears in `array_expression.rs`, `array_replace.rs`, others)
- `HAYSTACK_NULL_DENSITY = 0.1`
- `NEEDLE_DENSITY = 0.1`
- Test sizes: `SIZES = [(4_000, 10), (10_000, 100), (10_000, 500)]`

**Duplicated helper functions:**
- `create_list_array()` — implemented separately in `array_expression.rs` (hardcoded needle/values) and `array_replace.rs` (generic template version)
- `create_args()`, `create_args_n()` — repeated ScalarFunctionArgs construction patterns
- Similar patterns in `array_remove.rs`, `array_slice.rs`, etc.

**Current architecture issues:**
1. **No single source of truth:** Constants duplicated; no clear place to adjust test parameters globally
2. **Maintenance drift:** Changing needle density or seed in one file doesn't propagate to others
3. **Generic helpers exist but unused:** `array_replace.rs` has generic `create_list_array<Builder, Item>()` but `array_expression.rs` reimplements a simpler version
4. **Field naming inconsistency:** `array_replace.rs` uses `"haystack"` field name; `array_expression.rs` uses `"array"`

## Proposed Solution
Create a shared benchmark fixture module: `datafusion/functions-nested/benches/common_fixtures.rs`

**Module exports:**
```rust
// Constants
pub const SEED: u64 = 42;
pub const HAYSTACK_NULL_DENSITY: f64 = 0.1;
pub const NEEDLE_DENSITY: f64 = 0.1;
pub const BENCHMARK_SIZES: &[(usize, usize)] = &[(4_000, 10), (10_000, 100), (10_000, 500)];

// Helpers
pub fn create_list_array<Builder, Item>(...) -> ArrayRef { ... }
pub fn create_fixed_size_binary_list_array<const SIZE: usize>(...) -> ArrayRef { ... }
pub fn create_nested_i64_list_array(...) -> ArrayRef { ... }

// Standard ScalarFunctionArgs builders
pub fn create_args(haystack: ArrayRef, from: ScalarValue, to: ScalarValue) -> ScalarFunctionArgs { ... }
pub fn create_args_n(haystack: ArrayRef, from: ScalarValue, to: ScalarValue, n: ScalarValue) -> ScalarFunctionArgs { ... }
```

**Migration strategy:**
1. Extract constants and generic helpers into `common_fixtures.rs`
2. Update `array_replace.rs` to import from `common_fixtures`
3. Incrementally migrate other benchmark files (`array_expression.rs`, `array_remove.rs`, etc.)
4. Normalize field naming across all benchmarks

## Acceptance Criteria
- [ ] New `common_fixtures.rs` module created with shared constants and helpers
- [ ] `array_replace.rs` refactored to import from `common_fixtures` (no behavior change)
- [ ] `array_expression.rs` refactored to use `common_fixtures` helpers
- [ ] At least 3 benchmark files migrated to use the shared module
- [ ] Field naming standardized (e.g., `"haystack"` consistently used for input arrays)
- [ ] All benchmarks produce identical results pre/post refactoring
- [ ] `cargo bench -p datafusion-functions-nested` runs successfully
- [ ] No duplicate constant definitions across files

## Implementation Notes
- **Scope:** Start with constants and generic helpers; defer specialized helpers (e.g., `create_nested_i64_list_array` variants) for phase 2
- **Feature flag:** Consider `#[cfg(test)]` scoping if fixtures are benchmark-only
- **Naming:** Use `common_fixtures.rs` or `benchmark_fixtures.rs` for clarity
- **Testing:** Verify benchmark output stability via `criterion` comparison tools
- **Risk:** Field name normalization may require updating benchmark group names if they hard-depend on field names

## Dependencies
- Depends on resolution of Issue #[NEW_ISSUE_01] (consolidate array_replace benchmarks) — resolving that first simplifies fixture consolidation
- Related to broader benchmark infrastructure improvements in `datafusion-functions-nested`

## References
- `datafusion/functions-nested/benches/array_expression.rs` (lines 36–45: duplicate constants)
- `datafusion/functions-nested/benches/array_replace.rs` (lines 37–44: same constants; lines 480–600: helper functions)
- `datafusion/functions-nested/benches/array_remove.rs` (similar patterns)
- PR #22143: Added array_expression benchmarks, making duplication newly visible
