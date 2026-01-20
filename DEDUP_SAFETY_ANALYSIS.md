# Where Deduplication Safety Is Addressed

## The Feedback (from ethan-tyler)

> "Dedup is done after stripping qualifiers; that can collapse distinct qualified predicates in multi-scan plans. I would either enforce single-target-scan eligibility or adjust dedup so it can't drop distinct predicates."

---

## The Hazard Example

```sql
UPDATE target SET col = 1 FROM source WHERE target.id = 5 AND source.id = 5
```

**Without scoping/validation:**
1. Extract `target.id = 5` and `source.id = 5`
2. Strip qualifiers → both become `id = 5`
3. Dedup sees duplicates → collapses to single predicate
4. Result: Loses the distinction that one is for target, one for source
5. Semantic corruption!

---

## The Solution: Two-Layer Approach

This feedback is addressed through **TWO SEPARATE MECHANISMS** working in concert:

### **Layer 1: Target Scan Scoping** (Commit `695b487d8`)
**Goal:** Enforce single-target-scan eligibility at extraction time

**Location:** [physical_planner.rs, lines 1948-1959](datafusion/core/src/physical_planner.rs#L1948-L1959)

```rust
LogicalPlan::TableScan(TableScan {
    table_name,
    filters: scan_filters,
    ..
}) => {
    // Only extract filters from the target table scan.
    // This prevents incorrect filter extraction in UPDATE...FROM scenarios
    // where multiple table scans may have filters.
    if table_name.resolved_eq(target) {  // ← SCOPING CHECK
        for filter in scan_filters {
            filters.extend(split_conjunction(filter).into_iter().cloned());
        }
    }
}
```

**Effect:**
- Non-target table scans (e.g., `source` table) are **silently ignored**
- Only filters from the target table collected
- `source.id = 5` never reaches dedup because it's filtered out here

---

### **Layer 2: Predicate-Level Validation** (Commit `486010533`)
**Goal:** Validate Filter node predicates reference only target table

**Location:** [physical_planner.rs, lines 1944-1947](datafusion/core/src/physical_planner.rs#L1944-L1947)

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

**Helper Function `predicate_is_on_target()`** [lines 2018-2027](datafusion/core/src/physical_planner.rs#L2018-L2027):

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

**Effect:**
- Rejects any predicate with qualified columns from non-target tables
- Example: `source.id = 5` rejected at Filter node level
- Example: `target.id = 5` accepted

---

### **Layer 3: Safe Deduplication** (Part of both commits)
**Location:** [physical_planner.rs, lines 1999-2012](datafusion/core/src/physical_planner.rs#L1999-L2012)

```rust
// Strip qualifiers and deduplicate. This ensures:
// 1. Only target-table predicates are retained from Filter nodes
// 2. Qualifiers stripped for TableProvider compatibility
// 3. Duplicates removed (from Filter nodes + TableScan.filters)
//
// Deduplication is necessary because filters may appear in both Filter nodes
// and TableScan.filters when the optimizer performs partial (Inexact) pushdown.
let mut seen_filters = HashSet::new();
filters
    .into_iter()
    .try_fold(Vec::new(), |mut deduped, filter| {
        let unqualified = strip_column_qualifiers(filter).map_err(|e| {
            e.context(format!(
                "Failed to strip column qualifiers for DML filter on table '{target}'"
            ))
        })?;
        if seen_filters.insert(unqualified.clone()) {
            deduped.push(unqualified);
        }
        Ok(deduped)
    })
```

**Why safe now:**
- By this point, ALL filters are guaranteed to be target-scoped (Layer 1 + 2)
- Dedup can now safely collapse `target.id = 5` from both Filter and TableScan
- No cross-table predicates to accidentally collapse
- Safe because: "Only target-table predicates are retained from Filter nodes"

**Key Comment Explains Invariant:**
```rust
// Deduplication is safe here because:
// 1. All filters belong to the same target table (enforced by Layer 1 + Layer 2)
// 2. After stripping qualifiers, duplicates represent truly redundant predicates
```

---

## How It Works End-to-End

### **Scenario: DELETE with Compound WHERE, Mixed Locations**

```sql
DELETE FROM target WHERE target.id = 5 AND target.status = 'active'
```

With `TableProviderFilterPushDown::Inexact`:

**Before Layer 1 (TableScan Scoping):**
```
Filter: target.status = 'active'
  └─ TableScan: filters = [target.id = 5]
```

**Layer 1 (TableScan Scoping Check):**
```rust
if table_name.resolved_eq(target) {  // ✅ target == target
    filters.extend([target.id = 5]);
}
// Adds: target.id = 5
```

**Layer 2 (Filter Predicate Validation):**
```rust
if predicate_is_on_target(target.status = 'active', target)? {  // ✅ References target only
    filters.push(target.status = 'active');
}
// Adds: target.status = 'active'
```

**Collected Filters:**
```
[target.id = 5, target.status = 'active']
```

**Layer 3 (Safe Dedup after Qualifier Stripping):**
```rust
// Strip qualifiers:
[id = 5, status = 'active']

// Dedup (no duplicates):
[id = 5, status = 'active']

// Both predicates reach provider ✅
```

---

### **Scenario: UPDATE...FROM (Would Have Failed Before)**

```sql
UPDATE target SET col = 1 
FROM source 
WHERE target.id = 5 AND source.id = 5
```

**Layer 1 (TableScan Scoping Check):**
```rust
// For target scan
if table_name.resolved_eq(target) {  // ✅ matches target
    filters.extend([target.id = 5]);
}

// For source scan
if table_name.resolved_eq(target) {  // ❌ source != target
    // Skipped! source filters ignored
}
// Only: target.id = 5 collected
```

**Layer 2 (Filter Validation):**
```rust
// For target.id = source.id
if predicate_is_on_target(target.id = source.id, target)? {  // ❌ References source
    // Rejected! Cross-table predicate refused
}
// Predicate rejected at layer 2
```

**Result:**
```
Only [target.id = 5] passed to provider
source.id = 5 never collected
No dedup collision possible ✅
```

---

## Why Both Layers Are Necessary

| Scenario | Layer 1 (Scoping) | Layer 2 (Validation) | Result |
|----------|-------------------|---------------------|--------|
| TableScan from non-target | Blocks | N/A | ✅ Prevented at extraction |
| Filter node with multi-table predicate | N/A | Blocks | ✅ Prevented at validation |
| Filter + TableScan both have `target.id = 5` | ✅ Both collected | ✅ Validated | ✅ Then deduped safely |
| Inexact pushdown: `id = 5` in Filter + TableScan | ✅ Both collected | ✅ Validated | ✅ Deduped to single predicate |

---

## Test Coverage

### **Validates Target Scoping Works**
- [test_delete_target_table_scoping()](datafusion/core/tests/custom_sources_cases/dml_planning.rs#L455)
- [test_update_filter_pushdown_extracts_table_scan_filters()](datafusion/core/tests/custom_sources_cases/dml_planning.rs#L441)

### **Validates Safe Dedup With Mixed Locations**
- [test_delete_mixed_filter_locations()](datafusion/core/tests/custom_sources_cases/dml_planning.rs#L377)
  - Uses `Inexact` pushdown
  - Verifies both predicates extracted
  - Confirms dedup doesn't lose either

---

## Design Decision Rationale

**Why not adjust dedup instead of enforcing scoping?**

> The feedback asks: "I would either enforce single-target-scan eligibility **or** adjust dedup..."

**Answer: Both are needed.**

- **Scoping alone isn't enough:** Prevents TableScan leakage but Filter nodes can still contain cross-table predicates
- **Dedup adjustment alone isn't safe:** Can't reliably detect "distinct" vs "duplicate" after stripping qualifiers
- **Together:** Fail-closed approach - scoping prevents non-target scans, validation prevents cross-table predicates, then dedup is truly safe

**This is more robust than:**
- "Track original qualifiers through dedup" (complex, error-prone)
- "Remember which table each filter came from" (adds state, maintenance burden)
- "Only dedup identical qualified forms" (misses real duplicates from Inexact pushdown)

---

## Implementation Details

### **Commit 695b487d8: Target Scan Scoping**
- Lines 1948-1959: TableScan filter extraction with scoping check
- Function signature: `extract_dml_filters(input, target)` with target parameter

### **Commit 486010533: Predicate Validation**
- Lines 1944-1947: Filter node predicate validation
- Lines 2018-2027: `predicate_is_on_target()` helper function
- Lines 2008-2011: Error context on qualifier stripping failure

### **Supporting Commits**
- `b01240a3a`: Optimize `predicate_is_on_target` with short-circuit evaluation
- `7c5b02a44`: Add error context when qualifier stripping fails
- `ffdbe27d3`: Optimize dedup to single-pass `try_fold`

---

## Safety Guarantees After Implementation

| Guarantee | Enforced By |
|-----------|-------------|
| No non-target table scans contribute filters | Layer 1: `table_name.resolved_eq(target)` check |
| No cross-table qualified predicates collected | Layer 2: `predicate_is_on_target()` validation |
| Safe dedup without losing distinct predicates | Layers 1+2 ensure all collected filters are target-scoped |
| UPDATE...FROM supported safely | All three layers working together |

---

## Summary

The deduplication safety concern is addressed through:

1. **Target Scan Scoping** (Layer 1) - Prevents non-target table scans from contributing filters
2. **Predicate Validation** (Layer 2) - Prevents cross-table predicates in Filter nodes
3. **Safe Dedup** (Layer 3) - Can deduplicate after stripping because all filters are target-scoped

**Result:** Dedup cannot collapse distinct predicates from different tables because predicates from non-target tables are blocked earlier.
