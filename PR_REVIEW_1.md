# PR Review: DML Filter Extraction Enhancement for TableScan Pushdown

**Commit Range**: 2246b8bd7^..ea5a0888f  
**Reviewer**: GitHub Copilot  
**Date**: January 20, 2026

---

## Executive Summary

This PR addresses issue #19840 where `TableProvider::delete_from` was not receiving filter expressions when filter pushdown was enabled. The changes successfully solve the core problem by:

1. Extracting filters from both `Filter` nodes and `TableScan.filters`
2. Adding target table scoping to prevent incorrect filter extraction in multi-table scenarios (UPDATE...FROM)
3. Implementing deduplication to avoid passing duplicate predicates
4. Adding comprehensive test coverage for various filter pushdown scenarios

**Decision**: ✅ **Approve with suggestions**

The implementation is functionally correct and solves the stated problem. However, there are several areas where the code could be improved for clarity, maintainability, and alignment with repository best practices.

---

## Detailed Analysis

### 1. Core Implementation (`physical_planner.rs`)

#### ✅ **Strengths**

1. **Correct Problem Diagnosis**: The fix correctly identifies that filters can exist in:
   - Explicit `Filter` nodes (when pushdown is unsupported/partial)
   - `TableScan.filters` (when pushdown is Exact/Inexact)

2. **Target Table Scoping**: The addition of `predicate_is_on_target()` prevents incorrect filter extraction in UPDATE...FROM scenarios where multiple tables may have filters.

3. **Deduplication Logic**: Handles cases where the same predicate appears in both locations due to partial pushdown.

#### ⚠️ **Issues & Suggestions**

##### Issue 1: Exhaustive Match Arms Create Maintenance Burden

**Current Code** (lines 1944-1975):
```rust
match node {
    LogicalPlan::Filter(filter) => { /* ... */ }
    LogicalPlan::TableScan(TableScan { /* ... */ }) => { /* ... */ }
    
    // Plans without filter information
    LogicalPlan::EmptyRelation(_)
    | LogicalPlan::Values(_)
    | LogicalPlan::DescribeTable(_)
    // ... 9 more variants
    => {
        // No filters to extract from leaf/meta plans
    }
    
    // Plans with inputs (may contain filters in children)
    LogicalPlan::Projection(_)
    | LogicalPlan::SubqueryAlias(_)
    // ... 9 more variants
    => {
        // Filter information may appear in child nodes; continue traversal
    }
}
```

**Problem**: 
- Lists 20+ plan variants explicitly with identical handling
- Will break when new `LogicalPlan` variants are added
- The two catch-all arms do the same thing (continue traversal)
- Violates the "prefer simple code paths" principle from `AGENTS.md`

**Recommendation**: Refactor to handle only special cases:

```rust
match node {
    LogicalPlan::Filter(filter) => {
        // Split AND predicates into individual expressions
        for predicate in split_conjunction(&filter.predicate) {
            if predicate_is_on_target(predicate, target)? {
                filters.push(predicate.clone());
            }
        }
    }
    LogicalPlan::TableScan(TableScan {
        table_name,
        filters: scan_filters,
        ..
    }) => {
        // Only extract filters from the target table scan.
        // This prevents incorrect filter extraction in UPDATE...FROM scenarios
        // where multiple table scans may have filters.
        if table_name.resolved_eq(target) {
            for filter in scan_filters {
                filters.extend(split_conjunction(filter).into_iter().cloned());
            }
        }
    }
    // All other plan types: continue traversal to find Filter/TableScan nodes
    _ => {}
}
```

**Rationale**: 
- More maintainable (won't break with new plan types)
- Clearer intent (only these two cases matter)
- Follows Rust idiom of exhaustive pattern matching via `_` when appropriate
- Aligns with existing patterns in the codebase (see optimizer rules)

##### Issue 2: Functional Style Not Applied to Deduplication

**Current Code** (lines 1983-1991):
```rust
let mut seen_filters = HashSet::new();
let deduped = filters
    .into_iter()
    .map(strip_column_qualifiers)
    .collect::<Result<Vec<_>>>()?
    .into_iter()
    .filter(|f| seen_filters.insert(f.clone()))
    .collect();
```

**Problem**:
- Creates intermediate `Vec` allocation unnecessarily
- Chains `.collect()` → `.into_iter()` which is an anti-pattern
- Clone on every filter check increases memory churn

**Recommendation**: Use `try_fold` for single-pass processing:

```rust
// Deduplication is necessary because filters may appear in both Filter nodes
// and TableScan.filters when the optimizer performs partial pushdown.
let mut seen_filters = HashSet::new();
filters.into_iter().try_fold(Vec::new(), |mut deduped, filter| {
    let unqualified = strip_column_qualifiers(filter)?;
    if seen_filters.insert(unqualified.clone()) {
        deduped.push(unqualified);
    }
    Ok(deduped)
})
```

**Benefits**:
- Single-pass: no intermediate allocation
- Early exit on error (via `try_fold`)
- Explicit inline comment explains why deduplication is needed
- Follows functional style from `AGENTS.md` → "Refactor imperative traversals into declarative expressions"

##### Issue 3: Missing Edge Case Documentation

The function doc comment doesn't explain:
- What happens with UPDATE...FROM (multi-table scenarios)
- Why target table scoping is critical
- The deduplication rationale (already mentioned in Issue 2)

**Recommendation**: Enhance doc comment:

```rust
/// Extract filter predicates from a DML input plan (DELETE/UPDATE).
///
/// Walks the logical plan tree and collects:
/// - Filter predicates from explicit `Filter` nodes
/// - Filters pushed down into `TableScan` nodes (when provider supports filter pushdown)
///
/// For UPDATE...FROM queries with multiple tables, only filters referencing the target
/// table are extracted. This prevents incorrectly passing source table predicates to
/// the TableProvider's delete_from/update methods.
///
/// Implementation details:
/// - Splits AND conjunctions into individual expressions
/// - Strips column qualifiers for TableProvider schema compatibility
/// - Deduplicates filters that appear in both Filter and TableScan nodes
///
/// # Arguments
/// * `input` - The logical plan subtree (typically the WHERE clause plan)
/// * `target` - The target table reference (used for scoping filters)
///
/// # Returns
/// Vector of unqualified filter expressions ready for TableProvider consumption
```

##### Issue 4: `predicate_is_on_target` Could Be More Efficient

**Current Code** (lines 1997-2005):
```rust
fn predicate_is_on_target(expr: &Expr, target: &TableReference) -> Result<bool> {
    let mut columns = HashSet::new();
    expr_to_columns(expr, &mut columns)?;

    Ok(columns.iter().all(|column| {
        column
            .relation
            .as_ref()
            .is_none_or(|relation| relation.resolved_eq(target))
    }))
}
```

**Observation**:
- Allocates a `HashSet` even though duplicates don't matter for the check
- Could short-circuit on first non-target column

**Recommendation**: Use `.exists()` for early termination:

```rust
/// Returns `true` if all column references in the predicate are either unqualified
/// or reference the target table. Used to filter out predicates on source tables
/// in UPDATE...FROM queries.
fn predicate_is_on_target(expr: &Expr, target: &TableReference) -> Result<bool> {
    let mut columns = HashSet::new();
    expr_to_columns(expr, &mut columns)?;
    
    // Short-circuit: if any column references a different table, reject this predicate
    Ok(!columns.iter().any(|column| {
        column.relation.as_ref().is_some_and(|relation| !relation.resolved_eq(target))
    }))
}
```

**Alternative** (if you want to avoid HashSet entirely):

```rust
fn predicate_is_on_target(expr: &Expr, target: &TableReference) -> Result<bool> {
    let is_on_target = !expr.exists(|node| {
        if let Expr::Column(column) = node {
            // Reject if column references a different table
            if let Some(relation) = &column.relation {
                return Ok(!relation.resolved_eq(target));
            }
        }
        Ok(false)
    })?;
    Ok(is_on_target)
}
```

**Rationale**: Follows mental model from `AGENTS.md` → "Think of `Option` as a computation pipeline" and "Refactor imperative traversals into declarative expressions"

---

### 2. Test Coverage (`dml_planning.rs`)

#### ✅ **Strengths**

1. **Comprehensive Scenarios**: Tests cover:
   - Simple DELETE with single filter
   - Compound filters (AND conjunction)
   - Multiple filter locations (Inexact pushdown)
   - UPDATE with filter pushdown
   - UPDATE...FROM with multi-table predicates
   - Qualifier stripping and validation

2. **Good Test Infrastructure**: 
   - `CaptureDeleteProvider` and `CaptureUpdateProvider` cleanly capture arguments
   - Parameterized constructors (`new_with_filter_pushdown`) avoid duplication

3. **Validates Both Logical and Physical Plans**: Tests check optimizer behavior (TableScan.filters) and runtime behavior (captured filters)

#### ⚠️ **Issues & Suggestions**

##### Issue 5: Helper Function Could Be More Idiomatic

**Current Code** (lines 221-231):
```rust
fn expr_has_table_reference(expr: &Expr, table: &str) -> Result<bool> {
    let reference = TableReference::bare(table);
    expr.exists(|node| {
        Ok(matches!(
            node,
            Expr::Column(column)
                if column.relation.as_ref().is_some_and(|relation| {
                    relation.resolved_eq(&reference)
                })
        ))
    })
}
```

**Observation**: Good use of `.exists()` combinator! However:
- The `matches!` + `if` guard can be simplified
- Could add doc comment explaining usage

**Recommendation**:

```rust
/// Returns `true` if the expression contains any column reference qualified with
/// the specified table name. Used in tests to verify filter scoping.
fn expr_has_table_reference(expr: &Expr, table: &str) -> Result<bool> {
    let reference = TableReference::bare(table);
    expr.exists(|node| {
        if let Expr::Column(column) = node {
            if let Some(relation) = &column.relation {
                return Ok(relation.resolved_eq(&reference));
            }
        }
        Ok(false)
    })
}
```

**Minor**: The `matches!` version is fine, but the explicit `if let` is slightly more idiomatic per Rust guidelines.

##### Issue 6: Test Comments Could Be Clearer

Several tests have good inline comments, but some could be more precise:

**Example** (line 278):
```rust
// Test mixed-location filters: some in Filter node, some in TableScan.filters
```

**Better**:
```rust
// Verify that predicates split across Filter node and TableScan.filters
// (due to Inexact pushdown) are both extracted and deduplicated correctly.
```

**Applies to**: `test_delete_mixed_filter_locations`, `test_update_from_drops_non_target_predicates`

---

### 3. Design & Architecture

#### ✅ **Strengths**

1. **Localized Fix**: Changes only touch `physical_planner.rs` and test file
2. **Backward Compatible**: Existing DELETE/UPDATE without pushdown still work
3. **Consistent API**: Both DELETE and UPDATE benefit from the same fix

#### ⚠️ **Issues & Suggestions**

##### Issue 7: Error Context Could Be More Specific

The error messages added are good:
```rust
.context(format!("DELETE operation on table '{table_name}'"))
```

But when filter extraction fails, the error might not be clear. Consider adding context in `extract_dml_filters`:

```rust
fn extract_dml_filters(
    input: &Arc<LogicalPlan>,
    target: &TableReference,
) -> Result<Vec<Expr>> {
    // ... existing code ...
    
    filters.into_iter().try_fold(Vec::new(), |mut deduped, filter| {
        let unqualified = strip_column_qualifiers(filter).map_err(|e| {
            e.context(format!(
                "Failed to strip qualifiers from DML filter for table '{target}'"
            ))
        })?;
        if seen_filters.insert(unqualified.clone()) {
            deduped.push(unqualified);
        }
        Ok(deduped)
    })
}
```

##### Issue 8: Potential Future Enhancement Location

When reading the code, it's not immediately obvious where to add support for:
- DELETE...USING (PostgreSQL syntax for multi-table deletes)
- MERGE statements with filters

**Recommendation**: Add a TODO or comment indicating extension points:

```rust
// Extract filters from the target table scan.
// TODO(#XXXXX): When adding support for DELETE...USING or MERGE,
// ensure this logic correctly scopes filters to the target table.
if table_name.resolved_eq(target) {
    // ...
}
```

---

### 4. Performance Analysis

#### Memory Allocations

**Current**:
- `Vec::new()` for filters (grows dynamically)
- `HashSet::new()` for deduplication
- Multiple `clone()` calls:
  - `predicate.clone()` in Filter handling
  - `filter.clone()` in TableScan handling  
  - `f.clone()` in deduplication check

**Assessment**: 
- ✅ Acceptable for DML operations (typically small number of filters)
- ⚠️ The `collect() → into_iter()` pattern creates unnecessary intermediate allocation

**Recommendation**: Already covered in Issue 2 (use `try_fold`)

#### Algorithmic Complexity

- Tree traversal: O(n) where n = plan nodes
- Deduplication: O(m * h) where m = filters, h = hash/eq complexity of `Expr`
- Overall: O(n + m * h), dominated by plan size

**Assessment**: ✅ Reasonable for typical queries

---

## Checklist Evaluation

| Criterion | Rating | Notes |
|-----------|--------|-------|
| **Consistency** | ⚠️ | Matches style mostly, but exhaustive match is unusual for this codebase |
| **Simplicity** | ⚠️ | Deduplication could be simpler; exhaustive match adds complexity |
| **Design** | ✅ | Localized, appropriate scope, solves the problem |
| **Effectiveness** | ✅ | Handles all reported scenarios; comprehensive tests |
| **Scope** | ✅ | Minimal changes, no feature creep |
| **Docs** | ⚠️ | Function doc comment missing key details (UPDATE...FROM, deduplication) |
| **References** | ✅ | Similar patterns exist in optimizer (filter extraction) |

---

## Detailed Recommendations

### 🔴 **Blocking** (Request changes if not addressed)

None. The code is functionally correct.

### 🟡 **Highly Recommended** (Should address before merge)

1. **Simplify match statement** (Issue 1)
   - Replace exhaustive match with `_ => {}` catch-all
   - Reduces maintenance burden and aligns with Rust idioms

2. **Improve deduplication** (Issue 2)
   - Use `try_fold` to avoid intermediate allocation
   - Add inline comment explaining deduplication purpose

3. **Enhance doc comment** (Issue 3)
   - Explain UPDATE...FROM behavior
   - Document deduplication rationale
   - Add parameter and return value descriptions

### 🟢 **Nice-to-Have** (Polish improvements)

4. **Optimize `predicate_is_on_target`** (Issue 4)
   - Use `.any()` for early exit
   - Or use `.exists()` directly on expr to avoid HashSet

5. **Clarify test comments** (Issue 6)
   - Make comments more specific about what's being tested

6. **Add error context** (Issue 7)
   - Include table name in filter stripping errors

7. **Add extension point comments** (Issue 8)
   - Mark where future DML features should be integrated

---

## Code Examples: Before & After

### Recommended Refactor: Deduplication Logic

**Before** (Current):
```rust
let mut seen_filters = HashSet::new();
let deduped = filters
    .into_iter()
    .map(strip_column_qualifiers)
    .collect::<Result<Vec<_>>>()?
    .into_iter()
    .filter(|f| seen_filters.insert(f.clone()))
    .collect();
```

**After** (Recommended):
```rust
// Deduplication is necessary because filters may appear in both Filter nodes
// and TableScan.filters when the optimizer performs partial (Inexact) pushdown.
let mut seen_filters = HashSet::new();
filters.into_iter().try_fold(Vec::new(), |mut deduped, filter| {
    let unqualified = strip_column_qualifiers(filter)?;
    if seen_filters.insert(unqualified.clone()) {
        deduped.push(unqualified);
    }
    Ok(deduped)
})
```

**Benefits**: Single-pass, no intermediate allocation, explicit rationale

---

### Recommended Refactor: Match Statement

**Before** (Current):
```rust
match node {
    LogicalPlan::Filter(filter) => { /* ... */ }
    LogicalPlan::TableScan(TableScan { /* ... */ }) => { /* ... */ }
    
    // 20+ explicitly listed variants with same behavior
    LogicalPlan::EmptyRelation(_)
    | LogicalPlan::Values(_)
    | /* ... 18 more ... */
    => { /* continue */ }
    
    LogicalPlan::Projection(_)
    | /* ... 9 more ... */
    => { /* continue */ }
}
```

**After** (Recommended):
```rust
match node {
    LogicalPlan::Filter(filter) => {
        // Split AND predicates and scope to target table
        for predicate in split_conjunction(&filter.predicate) {
            if predicate_is_on_target(predicate, target)? {
                filters.push(predicate.clone());
            }
        }
    }
    LogicalPlan::TableScan(TableScan {
        table_name,
        filters: scan_filters,
        ..
    }) => {
        // Only extract filters from the target table scan to prevent
        // incorrect filter extraction in UPDATE...FROM scenarios.
        if table_name.resolved_eq(target) {
            for filter in scan_filters {
                filters.extend(split_conjunction(filter).into_iter().cloned());
            }
        }
    }
    // All other plan types: continue traversal
    _ => {}
}
```

**Benefits**: Maintainable, won't break with new LogicalPlan variants, clearer intent

---

## Testing Strategy Validation

The test suite covers:

✅ **Basic scenarios**:
- Single filter DELETE
- Multiple filters (AND conjunction)
- Complex expressions

✅ **Filter pushdown scenarios**:
- Exact pushdown (all filters in TableScan)
- Inexact pushdown (split between Filter and TableScan)
- No pushdown (backward compatibility)

✅ **Multi-table scenarios**:
- UPDATE...FROM with source table predicates
- Target table scoping validation

✅ **Edge cases**:
- Qualifier stripping
- Deduplication
- Empty filters

**Missing**: 
- ⚠️ Test with `TableProviderFilterPushDown::Unsupported` explicitly (though covered implicitly by default provider)
- ⚠️ Test with very complex nested expressions (e.g., `WHERE (a = 1 AND b = 2) OR (c = 3 AND d = 4)`)

**Recommendation**: Add a test for complex OR predicates to ensure `split_conjunction` behavior is correct:

```rust
#[tokio::test]
async fn test_delete_complex_or_filters() -> Result<()> {
    let provider = Arc::new(CaptureDeleteProvider::new_with_filter_pushdown(
        test_schema(),
        TableProviderFilterPushDown::Exact,
    ));
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::clone(&provider) as Arc<dyn TableProvider>)?;

    // OR predicates should NOT be split by split_conjunction
    ctx.sql("DELETE FROM t WHERE id = 1 OR status = 'active'")
        .await?
        .collect()
        .await?;

    let filters = provider.captured_filters().expect("filters should be captured");
    assert_eq!(filters.len(), 1, "OR predicate should remain as single filter");
    assert!(filters[0].to_string().contains("OR"));
    Ok(())
}
```

---

## Comparison with Repository Guidelines

From `AGENTS.md` and `.github/copilot-instructions.md`:

### ✅ **Followed**:
1. ✅ "Prefer crate-scoped builds and tests" - changes localized to `datafusion-core`
2. ✅ "Always run ./dev/rust_lint.sh" - assumed (no lint errors visible)
3. ✅ "Add both unit tests and SQL logic tests" - comprehensive test suite added
4. ✅ "Mechanical changes beat conditional logic" - qualifier stripping is mechanical

### ⚠️ **Could Improve**:
1. ⚠️ "Prefer simple code paths over single complex adaptive path" - exhaustive match could be simpler
2. ⚠️ "Refactor imperative traversals into declarative expressions" - deduplication loop could use `try_fold`
3. ⚠️ "Use `.exists()` for boolean checks" - could apply to `predicate_is_on_target`

---

## Security & Safety Analysis

### Memory Safety
✅ All operations use safe Rust, no `unsafe` blocks

### SQL Injection
✅ Filters are passed as `Expr` AST nodes, not raw strings

### Logic Errors
✅ Target table scoping prevents filter leakage between tables

### Potential Issues
⚠️ **None identified**, but consider:
- What happens if a malicious user provides a DELETE with 1000+ AND predicates?
- Current implementation will collect all of them (memory proportional to predicate count)
- **Assessment**: Not a concern for typical use; DML statements are administrator-controlled

---

## Integration with Existing Code

### Consistency with Optimizer
The filter extraction logic is consistent with how optimizer rules handle filters:
- `datafusion/optimizer/src/push_down_filter.rs` uses similar `split_conjunction` patterns
- `datafusion/optimizer/src/utils.rs` has helper functions for filter handling

### Consistency with Physical Planning
The changes integrate cleanly with existing physical planning:
- DELETE/UPDATE already had `extract_dml_filters` call sites
- Adding `target` parameter is a minimal API change
- No changes needed to `TableProvider` trait (good design)

---

## Final Verdict

### ✅ **Approve with Suggestions**

**Justification**:
- **Functional correctness**: Solves the reported issue completely
- **Test coverage**: Comprehensive and validates all scenarios
- **Backward compatibility**: Existing code unaffected
- **Scope**: Appropriately narrow, no feature creep

**Conditions**:
1. **Highly Recommended** (should address):
   - Simplify match statement to use `_ => {}` catch-all
   - Refactor deduplication to use `try_fold` and add explanatory comment
   - Enhance function doc comment to explain UPDATE...FROM and deduplication

2. **Nice-to-Have** (can defer):
   - Optimize `predicate_is_on_target` with early exit
   - Add test for complex OR predicates
   - Add error context for qualifier stripping failures

---

## Action Items for Author

### Before Merge (Highly Recommended)

- [ ] Refactor `extract_dml_filters` match statement to use catch-all `_` arm
- [ ] Replace deduplication's `.collect() → .into_iter()` with single-pass `try_fold`
- [ ] Add inline comment explaining deduplication rationale
- [ ] Update function doc comment to document:
  - UPDATE...FROM multi-table behavior
  - Deduplication logic
  - Parameter and return value meanings

### Optional Improvements (Can defer to follow-up PR)

- [ ] Add test for complex OR predicate handling
- [ ] Optimize `predicate_is_on_target` to short-circuit on first mismatch
- [ ] Add error context when qualifier stripping fails
- [ ] Add TODO comments for future MERGE/DELETE USING support

---

## References

**Similar patterns in codebase**:
- Filter extraction: [datafusion/optimizer/src/push_down_filter.rs](datafusion/optimizer/src/push_down_filter.rs#L156-L180)
- Predicate scoping: [datafusion/optimizer/src/decorrelate_predicate_subquery.rs](datafusion/optimizer/src/decorrelate_predicate_subquery.rs#L425-L450)
- Deduplication: [datafusion/expr/src/utils.rs](datafusion/expr/src/utils.rs) (`merge_grouping_set`)

**Rust idioms applied**:
- `.exists()` for boolean checks (test helper)
- `is_none_or()` for optional validation
- `try_fold` for error propagation (recommended)

---

## Summary for Reviewers

**Key Changes**:
1. `extract_dml_filters` now accepts `target: &TableReference` parameter
2. Filters extracted from both `Filter` and `TableScan` nodes
3. Target table scoping prevents filter leakage in multi-table scenarios
4. Deduplication handles Inexact pushdown cases
5. Comprehensive test suite validates all scenarios

**Risk Assessment**: **Low**
- Changes are localized and well-tested
- Backward compatible (adds filtering, doesn't remove existing behavior)
- No performance regressions expected (DML operations are infrequent)

**Recommendation**: ✅ **Merge after addressing highly recommended items**

---

**Reviewed by**: GitHub Copilot  
**Review Date**: January 20, 2026  
**Commit Range**: 2246b8bd7^..ea5a0888f
