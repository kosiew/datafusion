# Where UPDATE Test Coverage and Mixed-Location Tests Are Addressed

## The Feedback (from ethan-tyler)

> "Nice DELETE+Exact pushdown regression. I would add the same coverage for UPDATE … WHERE … + TableProviderFilterPushDown::Exact (since extract_dml_filters is used by both). I would think about adding a "mixed location" case (some conjuncts in residual Filter, others in TableScan.filters) to lock in union+dedup behavior."

---

## The Fix: Two Separate Commits

### **Commit 1: `b607cd675`** - UPDATE Test Coverage
**Title:** `feat: add UPDATE tests, filter pushdown support to CaptureUpdateProvider and corresponding tests`

### **Commit 2: `d80191b61`** - Mixed-Location Filter Test  
**Title:** `test: add test for DELETE with mixed filter locations in CaptureDeleteProvider`

---

## Implementation Details

### Part 1: CaptureUpdateProvider Enhancement (Commit b607cd675)

#### **A. Added Filter Pushdown Support** (lines 123-127)
```rust
struct CaptureUpdateProvider {
    schema: SchemaRef,
    received_filters: Arc<Mutex<Option<Vec<Expr>>>>,
    received_assignments: Arc<Mutex<Option<Vec<(String, Expr)>>>>,
    filter_pushdown: TableProviderFilterPushDown,  // ← NEW FIELD
}
```

#### **B. New Constructor** (lines 139-149)
```rust
fn new_with_filter_pushdown(
    schema: SchemaRef,
    filter_pushdown: TableProviderFilterPushDown,
) -> Self {
    Self {
        schema,
        received_filters: Arc::new(Mutex::new(None)),
        received_assignments: Arc::new(Mutex::new(None)),
        filter_pushdown,
    }
}
```

#### **C. Implemented supports_filters_pushdown()** (lines 211-218)
```rust
fn supports_filters_pushdown(
    &self,
    filters: &[&Expr],
) -> Result<Vec<TableProviderFilterPushDown>> {
    Ok(vec![self.filter_pushdown.clone(); filters.len()])
}
```

**Why this matters:**
- Tells the optimizer whether this provider can push down filters
- `Exact` = can push down all filters into TableScan.filters
- `Inexact` = can only push down some filters (others stay in Filter node)
- Enables testing different optimization scenarios

---

### Part 2: UPDATE Test Coverage (Commit b607cd675)

#### **Test 1: `test_update_filter_pushdown_extracts_table_scan_filters()`** (Lines 441-474)

```rust
#[tokio::test]
async fn test_update_filter_pushdown_extracts_table_scan_filters() -> Result<()> {
    // Create UPDATE provider with Exact pushdown support
    let provider = Arc::new(CaptureUpdateProvider::new_with_filter_pushdown(
        test_schema(),
        TableProviderFilterPushDown::Exact,  // ← Can push down all filters
    ));
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::clone(&provider) as Arc<dyn TableProvider>)?;

    // Execute UPDATE with WHERE clause
    let df = ctx.sql("UPDATE t SET value = 100 WHERE id = 1").await?;
    let optimized_plan = df.clone().into_optimized_plan()?;

    // Verify optimizer pushed the filter into TableScan
    let mut scan_filters = Vec::new();
    optimized_plan.apply(|node| {
        if let LogicalPlan::TableScan(TableScan { filters, .. }) = node {
            scan_filters.extend(filters.clone());
        }
        Ok(TreeNodeRecursion::Continue)
    })?;

    assert_eq!(scan_filters.len(), 1);
    assert!(scan_filters[0].to_string().contains("id"));

    // Execute the UPDATE and verify filters were extracted and passed to update()
    df.collect().await?;

    let filters = provider
        .captured_filters()
        .expect("filters should be captured");
    assert_eq!(filters.len(), 1);
    assert!(filters[0].to_string().contains("id"));
    Ok(())
}
```

**What this tests:**
- Exact pushdown scenario (all filters moved to TableScan.filters)
- `extract_dml_filters()` correctly extracts from TableScan for UPDATE
- UPDATE operation has parity with DELETE for filter handling

---

#### **Test 2: `test_update_filter_pushdown_passes_table_scan_filters()`** (Lines 476-514)

```rust
#[tokio::test]
async fn test_update_filter_pushdown_passes_table_scan_filters() -> Result<()> {
    let provider = Arc::new(CaptureUpdateProvider::new_with_filter_pushdown(
        test_schema(),
        TableProviderFilterPushDown::Exact,
    ));
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::clone(&provider) as Arc<dyn TableProvider>)?;

    let df = ctx
        .sql("UPDATE t SET value = 42 WHERE status = 'ready'")
        .await?;
    let optimized_plan = df.clone().into_optimized_plan()?;

    let mut scan_filters = Vec::new();
    optimized_plan.apply(|node| {
        if let LogicalPlan::TableScan(TableScan { filters, .. }) = node {
            scan_filters.extend(filters.clone());
        }
        Ok(TreeNodeRecursion::Continue)
    })?;

    assert!(
        !scan_filters.is_empty(),
        "expected filter pushdown to populate TableScan filters"
    );

    df.collect().await?;

    let filters = provider
        .captured_filters()
        .expect("filters should be captured");
    assert!(
        !filters.is_empty(),
        "expected filters extracted from TableScan during UPDATE"
    );
    Ok(())
}
```

**What this tests:**
- Another UPDATE scenario with different WHERE clause
- Validates filters are consistently extracted across different predicates

---

### Part 3: Mixed-Location Filter Test (Commit d80191b61)

#### **Enhanced CaptureDeleteProvider** (similar to UPDATE provider)

First, the DELETE provider was enhanced with the same pattern:

```rust
struct CaptureDeleteProvider {
    schema: SchemaRef,
    received_filters: Arc<Mutex<Vec<Expr>>>,
    filter_pushdown: TableProviderFilterPushDown,  // ← NEW
}

fn new_with_filter_pushdown(
    schema: SchemaRef,
    filter_pushdown: TableProviderFilterPushDown,
) -> Self { /* ... */ }
```

---

#### **Test: `test_delete_mixed_filter_locations()`** (Lines 377-416)

```rust
#[tokio::test]
async fn test_delete_mixed_filter_locations() -> Result<()> {
    // Test mixed-location filters: some in Filter node, some in TableScan.filters
    // This happens when provider uses TableProviderFilterPushDown::Inexact,
    // meaning it can push down some predicates but not others.
    let provider = Arc::new(CaptureDeleteProvider::new_with_filter_pushdown(
        test_schema(),
        TableProviderFilterPushDown::Inexact,  // ← Partial pushdown support
    ));
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::clone(&provider) as Arc<dyn TableProvider>)?;

    // Execute DELETE with compound WHERE clause
    ctx.sql("DELETE FROM t WHERE id = 1 AND status = 'active'")
        .await?
        .collect()
        .await?;

    // Verify that both predicates are extracted and passed to delete_from(),
    // even though they may be split between Filter node and TableScan.filters
    let filters = provider
        .captured_filters()
        .expect("filters should be captured");
    assert_eq!(
        filters.len(),
        2,
        "should extract both predicates (union of Filter and TableScan.filters)"
    );

    let filter_strs: Vec<String> = filters.iter().map(|f| f.to_string()).collect();
    assert!(
        filter_strs.iter().any(|s| s.contains("id")),
        "should contain id filter"
    );
    assert!(
        filter_strs.iter().any(|s| s.contains("status")),
        "should contain status filter"
    );
    Ok(())
}
```

**What this tests:**
- `TableProviderFilterPushDown::Inexact` scenario (partial pushdown)
- Optimizer splits predicates: `id = 1` → TableScan.filters, `status = 'active'` → Filter node
- `extract_dml_filters()` union + dedup correctly collects from both locations
- Both predicates reach the provider (neither lost)

---

## How Filters Split in Inexact Pushdown

```
Original Query:
DELETE FROM t WHERE id = 1 AND status = 'active'

With TableProviderFilterPushDown::Inexact:
┌─────────────────────────┐
│  Filter node (residual) │
│  status = 'active'      │ ← Can't push down
└──────────────┬──────────┘
               │
┌──────────────▼──────────────┐
│     TableScan.filters       │
│     [id = 1]                │ ← Pushed down
└─────────────────────────────┘

extract_dml_filters() collects from BOTH:
1. Filter node → status = 'active'
2. TableScan.filters → id = 1

Result: Both predicates passed to delete_from()
```

---

## Deduplication & Union Behavior

The implementation validates union + dedup with this key section (lines 1999-2012 in physical_planner.rs):

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

**Test validates this path:**
- Filters from both locations collected
- Duplicates removed (though in this test, no duplicates exist)
- All unique predicates passed through

---

## Test Summary Matrix

| Test Name | Type | Scenario | Provider Support | What It Validates |
|-----------|------|----------|------------------|-------------------|
| `test_update_filter_pushdown_extracts_table_scan_filters` | UPDATE | Exact pushdown | `Exact` | UPDATE can extract from TableScan.filters like DELETE |
| `test_update_filter_pushdown_passes_table_scan_filters` | UPDATE | Exact pushdown | `Exact` | UPDATE filters consistently extracted across scenarios |
| `test_delete_mixed_filter_locations` | DELETE | Partial pushdown | `Inexact` | Union + dedup works when filters split between Filter node and TableScan.filters |

---

## Coverage Comparison

### Before These Commits
```
DELETE + Exact pushdown:    ✅ test_delete_filter_pushdown_extracts_table_scan_filters
UPDATE + Exact pushdown:    ❌ Missing
DELETE + Inexact pushdown:  ❌ Missing (mixed locations)
```

### After These Commits
```
DELETE + Exact pushdown:    ✅ test_delete_filter_pushdown_extracts_table_scan_filters
UPDATE + Exact pushdown:    ✅ test_update_filter_pushdown_extracts_table_scan_filters
                            ✅ test_update_filter_pushdown_passes_table_scan_filters
DELETE + Inexact pushdown:  ✅ test_delete_mixed_filter_locations
```

---

## Why This Matters

### 1. **Parity Between DELETE and UPDATE**
Both operations use `extract_dml_filters()`, so both must handle:
- Filter pushdown (Exact)
- Multiple filter sources (mixed locations)
- Deduplication

### 2. **Lock In Union + Dedup Behavior**
The mixed-location test validates the specific design choice:
- Collect from both Filter nodes AND TableScan.filters
- Deduplicate after collecting
- Pass all unique predicates to provider

This prevents future changes from accidentally breaking this behavior.

### 3. **Prepare for UPDATE...FROM**
When UPDATE...FROM support is added, these tests ensure:
- Single-table UPDATE works correctly
- Filter collection is predictable
- Scoping logic (target table only) is established

---

## Commit References

| Commit | Feature | Files Changed | Lines Added |
|--------|---------|----------------|-------------|
| `b607cd675` | UPDATE test coverage + CaptureUpdateProvider enhancement | dml_planning.rs | ~90 |
| `d80191b61` | Mixed-location filter test + CaptureDeleteProvider enhancement | dml_planning.rs | ~50 |

Both commits are part of the comprehensive test suite added in the 20-commit series.
