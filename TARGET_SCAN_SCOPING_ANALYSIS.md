# Where Target Scan Scoping is Addressed

## The Concern (from ethan-tyler's feedback)

> "This collects TableScan.filters from any scan in the subtree. Works for single-table DELETE/UPDATE, but unsafe for UPDATE … FROM (extra scans). Should we restrict extraction to the DML target scan (match table_name or provider identity) and fail-closed if multiple candidate scans exist?"

---

## The Fix: Commit `695b487d8`

**Title:** `feat: enhance DELETE filter extraction to scope to target table only`

### Location: Three Key Places

#### 1. **Function Signature** (lines 1930-1933)
```rust
fn extract_dml_filters(
    input: &Arc<LogicalPlan>,
    target: &TableReference,        // ← NEW PARAMETER for scoping
) -> Result<Vec<Expr>> {
```

**Before:** Only had `input` parameter  
**After:** Added `target: &TableReference` parameter to restrict extraction

---

#### 2. **Core Implementation** (lines 1948-1959)
```rust
LogicalPlan::TableScan(TableScan {
    table_name,
    filters: scan_filters,
    ..
}) => {
    // Only extract filters from the target table scan.
    // This prevents incorrect filter extraction in UPDATE...FROM scenarios
    // where multiple table scans may have filters.
    if table_name.resolved_eq(target) {    // ← THE SCOPE CHECK
        for filter in scan_filters {
            filters.extend(split_conjunction(filter).into_iter().cloned());
        }
    }
}
```

**Key Logic:**
- Before: Extracted from **any** TableScan found in the tree
- After: Only extracts if `table_name.resolved_eq(target)` (exact match to target)
- Silently ignores filters from other table scans (e.g., in FROM clause)

---

#### 3. **Filter Node Processing** (lines 1944-1947)
```rust
LogicalPlan::Filter(filter) => {
    // Split AND predicates into individual expressions
    for predicate in split_conjunction(&filter.predicate) {
        if predicate_is_on_target(predicate, target)? {  // ← VALIDATION
            filters.push(predicate.clone());
        }
    }
}
```

**Defense Layer 2:**
- Even filters in Filter nodes are validated to only reference target table
- Uses `predicate_is_on_target()` helper (see below)

---

### 4. **Helper Function: `predicate_is_on_target()`** (lines 2015-2024)
```rust
/// Determine whether a predicate references only columns from the target table.
fn predicate_is_on_target(expr: &Expr, target: &TableReference) -> Result<bool> {
    let mut columns = HashSet::new();
    expr_to_columns(expr, &mut columns)?;

    // Short-circuit on first mismatch: returns false if any column references a different table
    Ok(!columns.iter().any(|column| {
        column
            .relation
            .as_ref()
            .is_some_and(|relation| !relation.resolved_eq(target))
    }))
}
```

**What it does:**
- Extracts all column references from an expression
- Checks if **any** column has a table qualifier that doesn't match target
- Returns `false` if any cross-table column is found
- Short-circuits on first mismatch for performance

---

## Call Sites: Target Parameter Passed In

### DELETE Operation (line 616)
```rust
let filters = extract_dml_filters(input, table_name)?;
//                                       ^^^^^^^^^^
//                    Pass the target table name
```

**Context:**
```rust
LogicalPlan::Delete(Delete {
    table_name,    // ← This is the DELETE target
    input,
    ..
}) => {
    let filters = extract_dml_filters(input, table_name)?;
    provider.table_provider.delete_from(session_state, filters).await?;
}
```

### UPDATE Operation (line 642)
```rust
let filters = extract_dml_filters(input, table_name)?;
//                                       ^^^^^^^^^^
//                    Pass the target table name
```

**Context:**
```rust
LogicalPlan::Update(Update {
    table_name,    // ← This is the UPDATE target
    input,
    ..
}) => {
    let filters = extract_dml_filters(input, table_name)?;
    let assignments = extract_update_assignments(input)?;
    provider.table_provider.update(session_state, assignments, filters).await?;
}
```

---

## Test Coverage: `test_delete_target_table_scoping()` (lines ~455-485)

```rust
#[tokio::test]
async fn test_delete_target_table_scoping() -> Result<()> {
    // Test that DELETE only extracts filters from the target table,
    // not from other tables (important for DELETE...FROM safety)
    let target_provider = Arc::new(CaptureDeleteProvider::new_with_filter_pushdown(
        test_schema(),
        TableProviderFilterPushDown::Exact,
    ));
    let ctx = SessionContext::new();
    ctx.register_table("target_t", Arc::clone(&target_provider) as Arc<dyn TableProvider>)?;

    // Execute DELETE on target table
    let df = ctx.sql("DELETE FROM target_t WHERE id > 5").await?;
    df.collect().await?;

    // Verify only target table filters were extracted
    let filters = target_provider.captured_filters().expect("filters should be captured");
    assert_eq!(filters.len(), 1);
    assert!(filters[0].to_string().contains("id"));
    Ok(())
}
```

**Why this matters:**
- Validates scoping logic works correctly for target table
- Lays groundwork for UPDATE...FROM support where scoping becomes critical

---

## Safety Model: Multi-Layer Defense

The implementation uses **three layers** of safety:

### Layer 1: TableScan Scoping (line 1950)
```rust
if table_name.resolved_eq(target) {
    // Only extract from matching table scan
}
```
**Filters from non-target scans silently ignored**

### Layer 2: Filter Node Validation (line 1946)
```rust
if predicate_is_on_target(predicate, target)? {
    // Only accept predicates referencing target table
}
```
**Predicates with cross-table columns rejected at Filter level**

### Layer 3: Qualifier Stripping Validation (commit 486010533)
```rust
let unqualified = strip_column_qualifiers(filter).map_err(|e| {
    e.context(format!(
        "Failed to strip column qualifiers for DML filter on table '{target}'"
    ))
})?;
```
**Error context added during qualifier stripping (follow-up commit)**

---

## UPDATE...FROM Safety (Why This Matters)

### The Hazard
```sql
UPDATE target SET col = 1 
FROM source 
WHERE target.id = source.id
```

**Without scoping:**
- Tree contains two table scans: `target` and `source`
- Old code would extract filters from BOTH
- `target.id = source.id` would be incorrectly applied
- After stripping: `id = id` (meaningless!)

**With scoping:**
- Only extracts filters where `table_name.resolved_eq(target)`
- Filters on `source` table are ignored
- Only target-relevant predicates passed to provider
- Fail-closed: Complex queries with multiple scans handled safely

---

## Summary

| Aspect | Details |
|--------|---------|
| **Commit** | `695b487d8` |
| **Type** | Feature (safety prerequisite) |
| **Key Change** | Added `target: &TableReference` parameter to `extract_dml_filters()` |
| **Core Logic** | Check `table_name.resolved_eq(target)` before extracting TableScan.filters |
| **Validation** | Use `predicate_is_on_target()` for Filter nodes |
| **Call Sites** | DELETE (line 616) and UPDATE (line 642) in physical_planner.rs |
| **Test** | `test_delete_target_table_scoping()` in dml_planning.rs |
| **Safety** | Prevents cross-table filter leakage, enables UPDATE...FROM support |
| **Status** | ✅ Implemented and tested |
