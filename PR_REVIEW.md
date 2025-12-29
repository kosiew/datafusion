# Code Review: Add ordering hook for DataSource-based scans

**Commit:** `0a5d7f385` (HEAD -> repartitioning-behavior-18513a)

**Title:** Add ordering hook for DataSource-based scans

---

## Summary

This commit addresses inconsistent repartitioning behavior when filters are pushed into `FileSource` scans. The solution introduces a new `repartition_preserve_ordering()` hook on the `DataSource` trait that ensures the repartitioning decision uses the original output ordering (before filter pushdown) rather than the potentially modified ordering from `eq_properties()`.

---

## Review Findings

### ✅ **Solves the Core Problem**

**What the fix does:**
- Introduces `DataSource::repartition_preserve_ordering()` trait method with a default implementation
- Overrides it in `FileScanConfig` to return `eq_properties_without_filters().output_ordering()`
- Updates `DataSourceExec::repartitioned()` to use the new hook instead of `self.properties().eq_properties.output_ordering()`

**Why it works:**
The key insight is that filters should not affect the *repartitioning decision*—only the underlying data retrieval. By separating "ordering for repartitioning" from "ordering post-filter," the code prevents filters from changing partition boundaries when they shouldn't.

**Evidence:**
- Two integration tests verify behavior is stable with/without filters and with/without ordering
- Test assertions check both `RepartitionExec` presence and the actual `DataSourceExec` line

---

### ✅ **Consistency with Codebase Patterns**

**Matches existing design:**
- The trait method pattern aligns with other `DataSource` methods (`eq_properties()`, `output_partitioning()`, `scheduling_type()`)
- Default implementation in trait + override in `FileScanConfig` is standard for optional behavior
- Documentation comments use the same style as surrounding code

**Good precedent:**
The approach mirrors how `eq_properties()` itself works—base trait provides a reasonable default, implementations can specialize.

---

### ✅ **Adequate Test Coverage**

The new test file (`repartition_policy.rs`) covers the essential scenarios:

| Test | What it verifies |
|------|---|
| `repartition_policy_stable_without_ordering()` | Adding filters doesn't change `RepartitionExec` wrapping or DataSourceExec line when no ordering constraint exists |
| `repartition_policy_stable_with_ordering()` | Adding filters doesn't change behavior when `ORDER BY` is present |

**Test quality:**
- Uses real SQL queries and actual CSV data
- Helper function `datasource_line()` extracts relevant plan line for comparison
- Clear assertion messages explain what's being tested
- Both cases check two dimensions: exec presence *and* exec line details

**Potential gap (minor):**
Tests don't explicitly verify that repartitioning *still works* when needed (e.g., checking that `target_partitions` actually increases partition count). Current tests only verify *stability*, not effectiveness. Consider adding a test that verifies repartitioning does occur when appropriate.

---

### 📝 **API Design – Solid but Worth Noting**

**Design choice:**
Adding a new optional trait method `repartition_preserve_ordering()` is conservative and backward-compatible.

**Rationale:**
- Default implementation delegates to `eq_properties().output_ordering()`, so existing data sources work unchanged
- Only `FileScanConfig` overrides it with different logic
- No breaking changes to public APIs

**Minor consideration:**
The method name `repartition_preserve_ordering` is slightly ambiguous—it could mean "preserve ordering while repartitioning" or "the ordering to preserve during repartitioning." Documentation clarifies intent, but consider the alternative name `repartition_ordering_hint()` if this becomes a pattern. *(Not blocking—current name is fine.)*

---

### 🧹 **Code Quality Observations**

#### Extraction: `eq_properties_without_filters()`

**What changed:**
A new private method extracts the logic that builds equivalence properties *before* applying filters. The old code is moved, not duplicated.

```rust
// Before: inline in eq_properties()
let mut eq_properties = EquivalenceProperties::new_with_orderings(...)
    .with_constraints(...);
// ... then projection logic ...
// ... then filter logic ...

// After: extracted method
let mut eq_properties = self.eq_properties_without_filters();
// ... then filter logic only ...
```

**Quality:**
- ✅ Reduces code duplication
- ✅ Makes intent clearer (separating "base ordering" from "filter constraints")
- ✅ Supports the new `repartition_preserve_ordering()` use case

#### Documentation

The new doc comment on `repartitioned()` is excellent:

```rust
/// ## Developer Notes
/// The planner prefers to push repartitioning into the scan itself when
/// [...] is enabled. This allows file formats to split large files while avoiding an
/// external [`RepartitionExec`](...).
/// 
/// * When an output ordering is known (for example, via listing table metadata),
///   the scan preserves that ordering while repartitioning.
/// * Filters that are pushed into the scan do **not** change the ordering signal
///   used here. A filtered query should pick the same repartition strategy as the
///   unfiltered equivalent unless the ordering requirement itself changes.
/// * [`ConfigOptions::repartition_file_min_size`](...)
///   controls how aggressively files are split; a smaller value yields more partitions.
```

This directly addresses the problem described in the issue and explains the fix clearly.

---

### ⚠️ **Minor Issues**

1. **Debug Log File**
   - `debug.log` is committed as an empty file
   - Should be gitignored or removed
   - **Action:** Remove before merge

2. **Test File Location**
   - New test file goes in `datafusion/core/tests/repartition_policy.rs`
   - This is appropriate for integration testing
   - ✅ Follows convention (tests under `tests/`, not `src/`)

3. **Projection Logic in `eq_properties_without_filters()`**
   - The projection handling is preserved unchanged from the original code
   - Correct: filters should only affect equivalence info, not projection
   - ✅ No logic error here

---

### 🎯 **Scope Assessment**

**What changed:**
- 1 new trait method
- 1 new private helper method on `FileScanConfig`
- 1 call site update in `DataSourceExec`
- 2 integration tests
- 11 lines of doc comments

**Scope:** ✅ **Minimal and focused**
- No refactoring of unrelated code
- No API breaks
- No new public types or large structural changes
- Directly addresses the reported issue

---

### 🔍 **Potential Edge Cases & Questions**

1. **What if a `FileSource` has custom equivalence properties logic?**
   - Default implementation routes through `eq_properties()`, which is safe
   - Overriding implementations (like `FileScanConfig`) can choose different strategies
   - ✅ No data loss risk

2. **Does this interact with projection pushdown?**
   - `eq_properties_without_filters()` applies projections before filters
   - This preserves the original design where projections are part of "base" ordering
   - ✅ Consistent with existing projection semantics

3. **Performance impact?**
   - Negligible—one extra method call per repartition decision (rare, once per query plan optimization)
   - ✅ No measurable overhead

4. **What about other `DataSource` implementations?**
   - Default trait method ensures they still work
   - If another source wants filter-aware repartitioning logic, it can override the hook
   - ✅ Extensible

---

## Checklist: Is This Ready to Merge?

| Criterion | Status | Notes |
|-----------|--------|-------|
| **Solves the problem** | ✅ | Filters no longer change repartition decisions |
| **No breaking changes** | ❌ | **REGRESSION: failing test** `equivalence_properties_after_schema_change` |
| **Tests pass** | ❌ | **Unit test failure in `file_scan_config::tests`** |
| **Code quality** | ⚠️ | Good structure but projection logic moved to wrong place |
| **Scope focused** | ✅ | Minimal, targeted fix |
| **Documentation adequate** | ✅ | Clear comments + test coverage |
| **Handles edge cases** | ❌ | Regression with projected schemas + filters |

---

## Critical Issue Found

During testing, a **regression in the test suite** was discovered:

```
test file_scan_config::tests::equivalence_properties_after_schema_change FAILED
```

**Root cause:** The refactoring moved projection handling into `eq_properties_without_filters()`, but `eq_properties()` needs to apply projections **after** filter equivalence constraints are added, not before.

**Current code (BROKEN):**
```rust
fn eq_properties(&self) -> EquivalenceProperties {
    let mut eq_properties = self.eq_properties_without_filters();  // Includes projections!
    
    if let Some(filter) = self.file_source.filter() {
        Self::add_filter_equivalence_info(&filter, &mut eq_properties, schema)?;
    }
    // Missing: projection after filters!
    eq_properties
}
```

**What went wrong:**
- `eq_properties_without_filters()` applies projections early
- Filter equivalence constraints are then added to the *projected* schema
- But the original code applied projections **last**, after all constraints

**Required fix:**
Move the projection logic out of `eq_properties_without_filters()` and keep it in `eq_properties()` **after** the filter handling:

```rust
fn eq_properties_without_filters(&self) -> EquivalenceProperties {
    let schema = self.file_source.table_schema().table_schema();
    EquivalenceProperties::new_with_orderings(
        Arc::clone(schema),
        self.output_ordering.clone(),
    )
    .with_constraints(self.constraints.clone())
    // NO projection here
}

fn eq_properties(&self) -> EquivalenceProperties {
    let schema = self.file_source.table_schema().table_schema();
    let mut eq_properties = self.eq_properties_without_filters();
    
    if let Some(filter) = self.file_source.filter() {
        Self::add_filter_equivalence_info(&filter, &mut eq_properties, schema)?;
    }
    
    // Apply projection AFTER filters
    if let Some(projection) = self.file_source.projection() {
        // ... projection logic ...
    }
    eq_properties
}
```

### Test Failure Deep Dive

**The failing test:** `equivalence_properties_after_schema_change`

This test verifies that when a schema is projected (columns removed), those columns should not appear in equivalence classes. The test:

1. Creates a `FileScanConfig` with a filter on `c2` (`c2 = 10`)
2. Applies a projection that only includes column `c1`
3. Expects column `c2` to be completely absent from equivalence classes

**Why the current code fails:**

The original code order was:
```
1. Create base eq_properties with ordering
2. Add filter constraints (which may reference c2)
3. Apply projection to remove c2
```

The refactored code order is:
```
1. eq_properties_without_filters() which includes projection (removes c2)
2. Add filter constraints (tries to reference c2 which is already gone!)
3. (projection not applied again)
```

Result: `c2` ends up in the equivalence class because projecting it out happened before trying to apply its constraints.

---

## Decision

### ❌ **REQUEST CHANGES** – Critical Regression

**Blocking issues:**
1. **Fix projection ordering bug** in `eq_properties()` method—projection must occur after filter constraints, not before
2. Remove the empty `debug.log` file before merging (it's an artifact)

**Required fixes before approval:**

1. **Fix the projection ordering bug in `eq_properties()`:**
   - The projection logic was moved into `eq_properties_without_filters()`, but it needs to be applied **after** filter constraints in `eq_properties()`
   - Move projection handling back to the end of `eq_properties()` method
   - Update `eq_properties_without_filters()` to NOT include projection logic
   - This fixes the failing test: `equivalence_properties_after_schema_change`

2. **Remove `debug.log`** before merging

**Non-blocking suggestions for future improvement (after critical fixes):**
1. **Add one more test** verifying that repartitioning actually *increases* partition count when `target_partitions > 1` and `repartition_file_min_size` is small. Current tests verify stability but not that the feature actively works.
2. **Consider an alternative method name** if this pattern spreads (e.g., `repartition_ordering_hint()` for clarity), though the current name is acceptable.
3. **Document the relationship** between `repartition_preserve_ordering()` and the optimizer's repartition rules in a follow-up issue or comment.

---

## Inline Comments for Author

### `datafusion/datasource/src/source.rs` (Lines 151-157)

```rust
fn repartition_preserve_ordering(&self) -> Option<LexOrdering> {
    self.eq_properties().output_ordering()
}
```

**Comment:** ✅ Good default—uses full eq_properties which is conservative and safe for unknown datasource types.

---

### `datafusion/datasource/src/file_scan_config.rs` (Lines 631-633)

```rust
fn repartition_preserve_ordering(&self) -> Option<LexOrdering> {
    self.eq_properties_without_filters().output_ordering()
}
```

**Comment:** ✅ Perfect—isolates "base ordering" from filters, which is the core fix. Very clear intent.

---

### `datafusion/datasource/src/file_scan_config.rs` (Lines 868-896)

```rust
fn eq_properties_without_filters(&self) -> EquivalenceProperties { ... }
```

**Comment:** ✅ Excellent extraction. Reduces duplication and enables the repartition hook. The logic is moved, not duplicated, so maintenance is easier. Consider adding an inline comment like:

```rust
// Helper to get base equivalence properties without filter constraints.
// Used for repartitioning decisions, which should not be affected by pushed-down filters.
```

---

### `datafusion/core/tests/repartition_policy.rs`

**Structure:** ✅ Good integration test. Uses real SQL and CSV data.

**Suggestion:** Add a third test validating that repartitioning *does occur*:

```rust
#[tokio::test]
async fn repartition_occurs_with_small_file_min_size() -> Result<()> {
    let session_config = SessionConfig::new()
        .with_target_partitions(4)
        .with_repartition_file_min_size(1);  // Very small threshold

    let plan = optimized_plan("SELECT c1, c2 FROM t", &session_config).await?;
    
    // Verify that the scan was actually repartitioned internally
    // (check DataSourceExec shows multiple partitions or RepartitionExec is absent)
    assert!(plan.contains("RepartitionExec") || plan.contains("partitions=4"));
    
    Ok(())
}
```

This ensures the feature actively works, not just that it's stable.

---

## Summary for PR

**Reviewers should focus on:**
1. Verify the test assertions match the expected behavior in the issue
2. Confirm `eq_properties_without_filters()` correctly isolates base ordering
3. Ensure the trait method follows the same patterns as other optional hooks

**Key insight:** The fix elegantly separates "ordering for repartitioning" from "ordering with filters applied." This prevents minor metadata/filter changes from accidentally shifting how data is partitioned.

---

## Action Items for Author

### 🔴 Critical (Blocking Approval)

**1. Fix projection ordering bug in `eq_properties()`**

**File:** `datafusion/datasource/src/file_scan_config.rs`

**Current state (lines 665-702):**
```rust
fn eq_properties(&self) -> EquivalenceProperties {
    let schema = self.file_source.table_schema().table_schema();
    let mut eq_properties = self.eq_properties_without_filters();
    
    if let Some(filter) = self.file_source.filter() {
        // Add filter constraints...
    }
    
    eq_properties  // Missing: projection!
}

fn eq_properties_without_filters(&self) -> EquivalenceProperties {
    // ... creates base eq_properties ...
    
    if let Some(projection) = self.file_source.projection() {
        // Apply projection - THIS IS WRONG PLACE!
    }
}
```

**What to do:**
- Move the projection logic OUT of `eq_properties_without_filters()`
- Add projection logic BACK to `eq_properties()` **after** the filter handling
- Keep `eq_properties_without_filters()` simple: just base ordering + constraints

**Why:**
The operation order matters: filters operate on the full schema, then projection removes columns. If you project first, filter constraints reference missing columns.

**2. Remove `debug.log` file**

This file was accidentally committed as empty. Delete it or add to `.gitignore`.

### 🟡 Medium Priority (Before Merge)

**Test the fix locally:**

```bash
cargo test -p datafusion-datasource --lib file_scan_config::tests::equivalence_properties_after_schema_change
```

Should pass after fix #1.

### 🟢 Low Priority (Nice to Have, Can Be Follow-Up)

**1. Add repartition effectiveness test**

In `datafusion/core/tests/repartition_policy.rs`, add test verifying repartitioning actually increases partition count.

**2. Add code comment**

In `eq_properties_without_filters()`, add comment explaining why projections are applied early:

```rust
/// Returns base equivalence properties without filter constraints.
/// Projections are applied here to match the signature for repartitioning hooks,
/// which need to see the full schema structure. Filter constraints will be added
/// in eq_properties() after filtering and before final projection.
```

---

## Testing Checklist

- [ ] Fix projection ordering bug
- [ ] Remove debug.log
- [ ] Run: `cargo test -p datafusion-datasource --lib` (should pass)
- [ ] Run: `cargo test -p datafusion --test repartition_policy` (should pass)
- [ ] Run: `./dev/rust_lint.sh` (format and lint)

---

## Summary

**Problem addressed:** Inconsistent repartitioning behavior when filters change metadata

**Solution approach:** Good (new trait hook to decouple repartitioning from filter-aware ordering)

**Implementation bug:** Critical projection ordering issue that breaks existing test

**Next steps:** Fix projection logic, then this PR will be ready to merge.

