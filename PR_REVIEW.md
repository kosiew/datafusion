# PR Review: Update DML Filter Extraction and Delete Tests

**Commit**: 2246b8bd7  
**Reviewer**: GitHub Copilot  
**Date**: January 19, 2026

## Overview

This commit addresses issue #19840 where DELETE operations with WHERE clauses were deleting all records instead of just the targeted ones. The root cause was that when a `TableProvider` supports filter pushdown (e.g., `TableProviderFilterPushDown::Exact`), the optimizer inlines filters into the `TableScan` node rather than keeping them in a separate `Filter` node. The DML filter extraction function only looked at explicit `Filter` nodes, thus missing pushed-down filters.

**Changes**:
1. **`datafusion/core/src/physical_planner.rs`**: Enhanced `extract_dml_filters()` to extract filters from both `Filter` and `TableScan` nodes, plus deduplication logic.
2. **`datafusion/core/tests/custom_sources_cases/dml_planning.rs`**: Added test infrastructure and a new test case validating filter extraction with filter pushdown.

---

## Decision: ✅ **Approve with Suggestions**

The solution is **functionally correct and solves the problem**, but has several opportunities for improvement:

---

## Detailed Review

### 1. ✅ Problem Understanding & Solution Approach

**What's Good**:
- Correctly identifies that filters can exist in both `Filter` nodes and `TableScan.filters`.
- The deduplication logic prevents the same filter from being applied twice (Filter + TableScan).
- Test case specifically validates filter pushdown scenarios.

**Note**: The fix is localized to the physical planner without requiring changes to multiple crates, which is appropriate.

---

### 2. 🔍 Code Quality & Clarity

#### `extract_dml_filters()` in `physical_planner.rs`

**Current Implementation**:
```rust
fn extract_dml_filters(input: &Arc<LogicalPlan>) -> Result<Vec<Expr>> {
    let mut filters = Vec::new();

    input.apply(|node| {
        match node {
            LogicalPlan::Filter(filter) => {
                filters.extend(split_conjunction(&filter.predicate).into_iter().cloned());
            }
            LogicalPlan::TableScan(TableScan { filters: scan_filters, .. }) => {
                for filter in scan_filters {
                    filters.extend(split_conjunction(filter).into_iter().cloned());
                }
            }
            _ => {}
        }
        Ok(TreeNodeRecursion::Continue)
    })?;

    let mut seen = HashSet::new();
    let mut deduped = Vec::new();

    for filter in filters {
        let filter = strip_column_qualifiers(filter)?;
        if seen.insert(filter.clone()) {
            deduped.push(filter);
        }
    }

    Ok(deduped)
}
```

**Issues**:

1. **Deduplication via `HashSet` with `clone()`** ⚠️

   The code uses `if seen.insert(filter.clone())` to deduplicate. While this works, it:
   - Requires `Expr` to implement `Hash` (already does) and `Eq`.
   - Clones each filter once more for the HashSet check.
   - The variable name `seen` could be more descriptive (e.g., `seen_filters`).

   **Suggestion**: This is acceptable for correctness, but consider if there are fewer-clone patterns elsewhere in the codebase (e.g., using indices or using the HashSet more directly).

2. **Imperative Post-Processing Loop** ⚠️

   The deduplication loop is imperative. Consider if it can be made more declarative:
   ```rust
   // Alternative (more functional):
   let mut seen = HashSet::new();
   filters
       .into_iter()
       .map(|f| strip_column_qualifiers(f))
       .collect::<Result<Vec<_>>>()?
       .into_iter()
       .filter(|f| seen.insert(f.clone()))
       .collect()
   ```

   However, this has the same clone overhead and may be less readable to some. **Current imperative version is fine.**

3. **Missing Comment on Deduplication Rationale** ⚠️

   The code doesn't explain *why* deduplication is needed. When would the same filter appear in both `Filter` and `TableScan`?

   **Suggestion**: Add a comment:
   ```rust
   // Deduplication is necessary because filters may appear in both Filter nodes
   // and TableScan.filters when the optimizer pushes some predicates down.
   // We deduplicate by (unqualified) expression to avoid passing the same filter twice.
   ```

---

### 3. 📋 Test Coverage

**New Test**: `test_delete_filter_pushdown_extracts_table_scan_filters()`

**What's Good**:
- Directly tests the scenario described in the issue (filter pushdown with `Exact` support).
- Validates both the logical plan (scan filters present) and runtime behavior (filters passed to `delete_from`).
- Uses realistic setup: registers provider with filter pushdown support, executes DELETE, checks results.

**Observations**:

1. **Test Infrastructure Enhancement** ✅
   - Added `new_with_filter_pushdown()` constructor and `filter_pushdown` field to `CaptureDeleteProvider`.
   - Added `supports_filters_pushdown()` implementation to enable filter pushdown.
   - These changes are minimal and appropriate.

2. **Test Assertions** ✅
   - `assert_eq!(scan_filters.len(), 1)` validates optimizer behavior.
   - `assert!(scan_filters[0].to_string().contains("id"))` checks filter content.
   - `assert_eq!(filters.len(), 1)` validates that `delete_from()` receives exactly one filter.

3. **Potential Missing Test Case** 📌
   - Current test uses a simple `WHERE id = 1` filter.
   - **Consider adding**: A test with compound filters (`WHERE id = 1 AND status = 'active'`) to verify that deduplication doesn't over-suppress filters.
   - **Consider adding**: A test without filter pushdown to verify backward compatibility (that the fix doesn't break existing DELETE without pushdown).

---

### 4. 🏗️ Design & Architecture

#### Match Statement Structure

The match statement handling `Filter` and `TableScan` is clear, but note:

```rust
match node {
    LogicalPlan::Filter(filter) => {
        filters.extend(split_conjunction(&filter.predicate).into_iter().cloned());
    }
    LogicalPlan::TableScan(TableScan { filters: scan_filters, .. }) => {
        for filter in scan_filters {
            filters.extend(split_conjunction(filter).into_iter().cloned());
        }
    }
    _ => {}
}
```

**Pattern**: The `Filter` case calls `split_conjunction(&filter.predicate)` and clones each element, while the `TableScan` case iterates and calls `split_conjunction(filter)` for each filter directly.

**Observation**: Both branches should produce a consistent interface. The `TableScan` loop could potentially be written similarly:

```rust
LogicalPlan::TableScan(TableScan { filters: scan_filters, .. }) => {
    for filter in scan_filters {
        filters.extend(split_conjunction(filter).into_iter().cloned());
    }
}
```

This is already done—consistency is good. ✅

---

### 5. 🔗 Integration & Scope

**Call Sites** (two locations):
- Line 616: `LogicalPlan::Dml(DmlStatement { op: WriteOp::Delete, ... })`
- Line 642: `LogicalPlan::Dml(DmlStatement { op: WriteOp::Update, ... })`

Both DELETE and UPDATE benefit from the fix, which is appropriate since they share the same filter extraction logic. ✅

---

### 6. 📝 Documentation

**Comment Quality**:
```rust
/// Extract filter predicates from a DML input plan (DELETE/UPDATE).
/// Walks the logical plan tree and collects Filter predicates,
/// splitting AND conjunctions into individual expressions.
/// Column qualifiers are stripped so expressions can be evaluated against
/// the TableProvider's schema.
```

**Issue**: The doc comment doesn't mention `TableScan` filters.

**Suggestion**: Update to:
```rust
/// Extract filter predicates from a DML input plan (DELETE/UPDATE).
/// Walks the logical plan tree and collects Filter predicates and any filters
/// pushed down into TableScan nodes, splitting AND conjunctions into individual expressions.
/// Column qualifiers are stripped so expressions can be evaluated against
/// the TableProvider's schema. Deduplicates filters to avoid passing the same
/// predicate twice when filters appear in both Filter and TableScan nodes.
```

---

### 7. 🛡️ Edge Cases & Error Handling

**Current Error Handling**:
- `strip_column_qualifiers(filter)?` is called inside the deduplication loop.
- If any filter fails to strip qualifiers, the entire operation returns an error.

**Assessment**: ✅ Appropriate. If a filter can't be processed, it's correct to fail the DML operation.

**Potential Edge Cases**:
1. **Empty Filters**: Both `Filter` and `TableScan` have no filters → produces empty `Vec<Expr>`. ✅ Correct.
2. **Duplicate Filters from Same Source**: The deduplication handles this. ✅
3. **Filters with Complex Expressions**: Handled by `split_conjunction()`. ✅
4. **Filters After Multiple Optimizer Passes**: Handled by tree walking with `apply()`. ✅

---

### 8. 🎯 Performance Considerations

**Clone Analysis**:
```rust
if seen.insert(filter.clone()) {
    deduped.push(filter);
}
```

- Clones `Expr` to check membership in `HashSet`.
- Accepts the clone cost in exchange for correctness.
- **Assessment**: Acceptable. DML operations (DELETE/UPDATE) are typically not performance-critical, and the number of filters is usually small (<10). ✅

---

### 9. ✨ Minor Suggestions

### Suggestion 1: Extract Helper for Filter Collection
*This is a polish suggestion, not blocking.*

The pattern of collecting filters from multiple sources could be extracted:
```rust
// Not necessary, but if this pattern repeats elsewhere:
fn collect_filters_from_node(node: &LogicalPlan) -> Vec<Expr> {
    match node {
        LogicalPlan::Filter(filter) => {
            split_conjunction(&filter.predicate).into_iter().cloned().collect()
        }
        LogicalPlan::TableScan(TableScan { filters: scan_filters, .. }) => {
            scan_filters
                .iter()
                .flat_map(|f| split_conjunction(f).into_iter().cloned())
                .collect()
        }
        _ => vec![],
    }
}
```

**Current Code is Fine**: The inline match is clear and localized. Only extract if reused elsewhere.

---

### Suggestion 2: Consider UpdateExpr for Future Enhancement
*Informational, not blocking.*

If the UPDATE operation also supports filter pushdown in the future, this code is ready. No changes needed now.

---

### Suggestion 3: Add Test for Backward Compatibility
*Recommended but not blocking.*

Add a test without filter pushdown to ensure existing behavior isn't broken:
```rust
#[tokio::test]
async fn test_delete_without_filter_pushdown() -> Result<()> {
    let provider = Arc::new(CaptureDeleteProvider::new(test_schema()));
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::clone(&provider) as Arc<dyn TableProvider>)?;
    
    ctx.sql("DELETE FROM t WHERE id = 2").await?.collect().await?;
    
    let filters = provider.captured_filters().expect("filters should be captured");
    assert_eq!(filters.len(), 1);
    assert!(filters[0].to_string().contains("id"));
    Ok(())
}
```

**Status**: Already passing via `test_delete_simple()` and `test_delete_complex_expr()`. ✅

---

## Checklist Summary

| Item | Status | Notes |
|------|--------|-------|
| **Consistency** | ✅ | Matches codebase patterns (match statements, error handling, imports). |
| **Simplicity** | ✅ | No unnecessary abstractions; code is straightforward. |
| **Design** | ✅ | Localized fix to `extract_dml_filters()`; appropriate scope. |
| **Effectiveness** | ✅ | Solves the issue; handles both Filter and TableScan cases. |
| **Scope** | ✅ | Minimal; only two files modified. |
| **Docs** | ⚠️ | Doc comment should mention TableScan filters and deduplication. |
| **Tests** | ✅ | Good coverage; backward compatibility maintained. |
| **Error Handling** | ✅ | Propagates errors correctly. |
| **Performance** | ✅ | Acceptable clone overhead; typical filters are small. |

---

## Final Recommendations

### ✅ Approve If:
1. Doc comment is updated to mention TableScan filter extraction and deduplication.
2. (Optional but recommended) Add an inline comment explaining the deduplication logic.

### 📝 Nice-to-Have (Non-Blocking):
1. Add a test for compound filters in the pushdown case to validate deduplication.
2. Extract a helper function if this filter collection pattern emerges elsewhere.

---

## Conclusion

This commit **successfully fixes the DELETE filter pushdown issue** with clean, maintainable code. The changes are well-scoped, properly tested, and integrate seamlessly with existing DML planning logic. The implementation correctly handles both explicit `Filter` nodes and pushed-down `TableScan` filters, with appropriate deduplication to prevent double-application.

**Recommendation**: ✅ **APPROVE WITH MINOR DOCUMENTATION IMPROVEMENTS**

Key improvements requested:
- [ ] Update `extract_dml_filters()` doc comment to mention TableScan filters and deduplication rationale.
- [ ] (Optional) Add inline comment in the deduplication loop explaining the purpose.

After these minor docs updates, this change is production-ready and should resolve the reported issue comprehensively.
