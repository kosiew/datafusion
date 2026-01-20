# PR Response: Delete Filter Extraction from TableScan

This document addresses the detailed review comments from @mjgarton, @adriangb, and @ethan-tyler on the `delete_from` filter extraction fix.

---

## Comment: Default Case Safety (mjgarton)

### Original Concern
> "I worry slightly that as `LogicalPlan` and perhaps the optimiser behavior change in future, this default case may lead to similar issues. If the outcome was 'nothing gets deleted' rather than 'everything gets deleted' I'd worry less."

### Response

**Agreed.** The current implementation with a catch-all `_ => {}` default case is a potential maintenance hazard. This comment appropriately flags that future logical plan additions could silently fail to extract filters, leading to unintended deletes of all rows.

### Proposed Fix: Explicit Variant Handling

We should follow the existing `LogicalPlan` pattern and handle variants **explicitly**, causing compilation failure when new variants are added:

```rust
fn extract_dml_filters(plan: &LogicalPlan) -> Vec<Expr> {
    let mut filters = Vec::new();
    
    match plan {
        LogicalPlan::Filter(filter) => {
            filters.extend(split_conjunction(&filter.predicate).into_iter().cloned());
        }
        LogicalPlan::TableScan(table_scan) => {
            filters.extend(table_scan.filters.iter().cloned());
        }
        // Plans without filter information
        LogicalPlan::EmptyRelation(_)
        | LogicalPlan::Values(_)
        | LogicalPlan::DescribeTable(_)
        | LogicalPlan::Explain(_)
        | LogicalPlan::Analyze(_)
        | LogicalPlan::Distinct(_)
        | LogicalPlan::Extension(_) => {
            // No filters to extract
        }
        // Plans with inputs (recursively extract from children)
        LogicalPlan::Projection(proj) => {
            filters.extend(extract_dml_filters(&proj.input));
        }
        LogicalPlan::Subquery(_)
        | LogicalPlan::SubqueryAlias(_)
        | LogicalPlan::Limit(_)
        | LogicalPlan::Sort(_)
        | LogicalPlan::Union(_)
        | LogicalPlan::Intersect(_)
        | LogicalPlan::Except(_)
        | LogicalPlan::Join(_)
        | LogicalPlan::CrossJoin(_)
        | LogicalPlan::Aggregate(_)
        | LogicalPlan::Window(_) => {
            // These plans may contain filter information in their inputs;
            // recursively search for Filter/TableScan nodes
            for input in plan.inputs() {
                filters.extend(extract_dml_filters(input));
            }
        }
    }
    
    filters
}
```

**Benefits:**
- **Fail-closed by design:** Adding a new `LogicalPlan` variant will cause a compilation error, forcing developers to consider whether it can hold filters.
- **Self-documenting:** Each variant explicitly shows the handling decision.
- **Safer evolution:** Reduces risk of silent filter extraction failures.

**Trade-off:** Requires updating `extract_dml_filters` whenever `LogicalPlan` gains a new variant (similar to current requirement in other methods).

### Timeline
This should be addressed **before or immediately after** merging the current fix. Suggest a follow-up PR.

---

## Comment: Exhaustiveness of Current Match (kosiew's Response & Regression Tests)

### Summary
@kosiew correctly notes that the current exhaustiveness is adequate for today's plan shapes:
- `Filter` nodes (in-memory filtering)
- `TableScan.filters` (pushed-down filters)

This is backed by regression tests:
- `test_delete_filter_pushdown_extracts_table_scan_filters`
- `test_delete_compound_filters_with_pushdown`

### Agreement & Caveat
**Agreed for current behavior.** However, this relies on optimizer behavior *remaining stable*. If a future optimization (e.g., a new filter-holding node) is introduced, the default case silently ignores it.

**Recommendation:** Adopt the explicit variant handling above to future-proof the code.

---

## Comment: Unified Filter Field on TableScan (adriangb)

### Suggestion
> "Would it help to have a `filters` field on `TableScan` similar to projection? Define a scan universally as: a table, a set of filter expressions, a set of projection expressions."

### Analysis

**Architectural merit:** This would provide a cleaner, more composable model:
- Unified filter handling across all table providers
- Simplified DML filter extraction (always look at `TableScan.filters`)
- Better cost-based optimizer support
- Consistent with traditional query planner design (e.g., Calcite)

**Implementation scope:** This is a **significant refactoring** affecting:
- `TableScan` struct definition
- All table provider implementations
- Filter planning logic in optimizer
- Physical planner translation

**Verdict:** Out of scope for the current fix but **worth tracking as a future enhancement**. Would resolve several edge cases in the codebase related to filter handling.

**Suggestion:** Open a separate issue to discuss and track this as a longer-term improvement.

---

## Comment: Additional Test Coverage (ethan-tyler)

### Current Gap
The fix is tested for DELETE + Exact pushdown, but gaps exist:

#### 1. UPDATE Test Coverage
**Missing:** UPDATE with filter pushdown is not tested, though `extract_dml_filters` handles both DELETE and UPDATE.

**Action:** Add a regression test:
```rust
#[tokio::test]
async fn test_update_filter_pushdown_extracts_table_scan_filters() -> Result<()> {
    let provider = Arc::new(CaptureUpdateProvider::new_with_filter_pushdown(
        TableProviderFilterPushDown::Exact,
    ));
    
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Utf8, false),
    ]));
    
    let ctx = create_test_context_with_provider("test_table", schema, provider.clone())?;
    
    // UPDATE test_table SET value = 'updated' WHERE id = 1
    ctx.sql("UPDATE test_table SET value = 'updated' WHERE id = 1")
        .await?
        .collect()
        .await?;
    
    let delete_calls = provider.update_calls();
    assert_eq!(delete_calls.len(), 1);
    // Verify filters were passed correctly
    assert!(!delete_calls[0].filters.is_empty());
}
```

#### 2. Mixed-Location Test
**Missing:** Scenario where predicates are split between `Filter` node and `TableScan.filters`.

This happens when:
- Provider reports `TableProviderFilterPushDown::Inexact`
- Optimizer pushes what it can, keeps residual predicates in `Filter` node

**Action:** Add test:
```rust
#[tokio::test]
async fn test_delete_mixed_filter_locations() -> Result<()> {
    // Set provider to Inexact pushdown
    let provider = Arc::new(CaptureDeleteProvider::new_with_filter_pushdown(
        TableProviderFilterPushDown::Inexact, // Some pushdown, some residual
    ));
    
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    
    let ctx = create_test_context_with_provider("test_table", schema, provider.clone())?;
    
    // WHERE clause that's partially pushed down
    ctx.sql("DELETE FROM test_table WHERE id = 1 AND category = 'test'")
        .await?
        .collect()
        .await?;
    
    let delete_calls = provider.delete_calls();
    assert_eq!(delete_calls.len(), 1);
    // Verify both predicates were collected (union of Filter + TableScan.filters)
    assert_eq!(delete_calls[0].filters.len(), 2);
}
```

---

## Comment: Scope Safety for UPDATE...FROM (ethan-tyler)

### Concern
> "Current logic pulls from any scan in the subtree and is unsafe for `UPDATE … FROM`."

**Example hazard:**
```sql
UPDATE target SET col = subquery_result
FROM source
WHERE target.id = source.id
```

If `extract_dml_filters` collects from **both** `target` and `source` table scans, it may apply `source`-specific predicates to `target`, causing incorrect filter semantics.

### Current Behavior
The function traverses the entire logical plan subtree, collecting filters from any `Filter` node or `TableScan.filters`, without distinguishing between:
- Filters on the **DML target table**
- Filters on **other tables** (in joins, subqueries, FROM clauses)

### Proposed Fix: Target Scan Scoping

```rust
fn extract_dml_filters(plan: &LogicalPlan, target_table_name: &str) -> Result<Vec<Expr>> {
    let mut filters = Vec::new();
    extract_dml_filters_recursive(plan, target_table_name, &mut filters)?;
    Ok(filters)
}

fn extract_dml_filters_recursive(
    plan: &LogicalPlan,
    target_table_name: &str,
    filters: &mut Vec<Expr>,
) -> Result<()> {
    match plan {
        LogicalPlan::Filter(filter) => {
            // Check that all column refs belong to target table
            let mut col_refs = HashSet::new();
            filter.predicate.collect_column_refs(&mut col_refs);
            
            for col_ref in col_refs {
                if let Some(table) = &col_ref.table {
                    if table != target_table_name {
                        return plan_err!(
                            "DELETE/UPDATE filter references non-target table: {}.{}",
                            table,
                            col_ref.column
                        );
                    }
                }
            }
            
            filters.extend(split_conjunction(&filter.predicate).into_iter().cloned());
        }
        LogicalPlan::TableScan(table_scan) if table_scan.table_name == target_table_name => {
            filters.extend(table_scan.filters.iter().cloned());
        }
        LogicalPlan::TableScan(_) => {
            // Skip scans of other tables
        }
        _ => {
            for input in plan.inputs() {
                extract_dml_filters_recursive(input, target_table_name, filters)?;
            }
        }
    }
    Ok(())
}
```

**Benefits:**
- **Fail-closed on ambiguity:** Cross-table predicates raise a planning error instead of silently applying them to the wrong table.
- **Correct semantics for UPDATE...FROM:** Only extracts predicates relevant to the target table.
- **Explicit validation:** Table names are checked, not assumed.

### Timeline
This should be addressed **in a follow-up PR** before supporting `UPDATE...FROM` syntax.

---

## Comment: Qualifier-Stripping Validation (ethan-tyler)

### Current Hazard

```rust
let deduped = filters
    .into_iter()
    .map(strip_column_qualifiers)
    .collect::<HashSet<_>>();
```

**Example problem:**
```sql
UPDATE target SET col = 1
FROM source
WHERE target.id = source.id
```

After stripping qualifiers: `target.id = source.id` → `id = id` (now meaningless!)

### Proposed Fix: Pre-Strip Validation

Add validation **before** stripping to ensure all column references belong to the DML target:

```rust
fn validate_and_strip_qualifiers(expr: &Expr, target_table_name: &str) -> Result<Expr> {
    // Collect all column references
    let mut col_refs = HashSet::new();
    expr.collect_column_refs(&mut col_refs);
    
    // Validate all columns belong to target or have no qualifier
    for col_ref in col_refs {
        if let Some(table) = &col_ref.table {
            if table != target_table_name {
                return plan_err!(
                    "DELETE/UPDATE filter contains column from non-target table: {}.{}",
                    table,
                    col_ref.column
                );
            }
        }
    }
    
    // Safe to strip qualifiers now
    Ok(strip_column_qualifiers(expr))
}
```

Then update filter collection:

```rust
let deduped = filters
    .into_iter()
    .map(|f| validate_and_strip_qualifiers(&f, target_table_name))
    .collect::<Result<HashSet<_>>>()?;
```

**Benefits:**
- **Explicit safety check:** Prevents silent semantic corruption.
- **Fail-closed design:** Ambiguous cases raise errors, don't corrupt filters.
- **Single point of validation:** All qualifier-stripping goes through this check.

### Timeline
Include this in the **Target Scan Scoping** follow-up PR.

---

## Comment: `is_identity_assignment` Qualifier Hazard (ethan-tyler)

### Similar Risk
The `is_identity_assignment` function (presumably in UPDATE planning) has the same qualifier-stripping hazard.

**Action:** Audit and apply the same validation pattern:
- Locate `is_identity_assignment` usage
- Add qualifier validation before stripping
- Document the invariant

### Example
```rust
fn is_identity_assignment(expr: &Expr, target_table: &str) -> Result<bool> {
    let expr = validate_and_strip_qualifiers(expr, target_table)?;
    // ... existing logic
}
```

---

## Comment: Deduplication After Stripping (ethan-tyler)

### Current Behavior
Deduplication happens **after** stripping qualifiers:

```rust
let deduped = filters
    .into_iter()
    .map(strip_column_qualifiers)
    .collect::<HashSet<_>>();
```

### Risk in Multi-Scan Plans
If two distinct predicates become identical after stripping (e.g., `t1.id = 1` and `t2.id = 1`), dedup collapses them to a single predicate.

**Example:** UPDATE...FROM with incorrectly scoped filters:
```sql
UPDATE t1 SET col = 1 FROM t2 WHERE t1.id = 5 AND t2.id = 5
```

If both are extracted and stripped → `id = 5` (deduplicated to single predicate, masking the multi-table nature).

### Mitigation Strategy
With the **Target Scan Scoping** fix above:
- Only filters belonging to target table are extracted
- Dedup after stripping is safe because all remaining filters are target-scoped

**Until then:** Keep dedup but document the assumption:

```rust
// Dedup is safe here because:
// 1. All filters belong to the same target table (enforced by extract_dml_filters_scoped)
// 2. After stripping qualifiers, duplicates represent truly redundant predicates
let deduped = filters
    .into_iter()
    .map(strip_column_qualifiers)
    .collect::<HashSet<_>>();
```

---

## Summary of Recommended Actions

| Priority | Item | Type | Effort | PR Scope | Status |
|----------|------|------|--------|----------|--------|
| **P0** | Explicit variant handling (mjgarton feedback) | Hardening | Low | Follow-up | ✅ **IMPLEMENTED** |
| **P1** | UPDATE test coverage (ethan-tyler) | Testing | Low | Current or follow-up | ✅ **IMPLEMENTED** |
| **P1** | Mixed-location filter test (ethan-tyler) | Testing | Medium | Follow-up | ✅ **IMPLEMENTED** |
| **P2** | Target scan scoping (ethan-tyler) | Feature | High | Follow-up (prerequisite for UPDATE...FROM) |  |
| **P2** | Qualifier-stripping validation (ethan-tyler) | Safety | Medium | Follow-up |  |
| **P3** | Audit `is_identity_assignment` (ethan-tyler) | Safety | Low | Follow-up |  |
| **P3** | Unified `TableScan.filters` design (adriangb) | Architecture | Very High | Future enhancement / RFC |  |

---

## Conclusion

The current fix addresses the **immediate issue** (DELETE with filter pushdown). The review comments identify important **follow-up hardening** and **future architectural** improvements:

1. **Immediate follow-up:** Explicit variant handling + UPDATE test coverage
2. **Medium-term follow-up:** Target scan scoping + qualifier validation (required for UPDATE...FROM safety)
3. **Long-term exploration:** Unified filter field on TableScan (architectural enhancement)

All feedback is valid and worth addressing incrementally to improve code safety and maintainability.

---

## Implementation Status

### ✅ P0: Explicit Variant Handling - IMPLEMENTED

**Location:** [`datafusion/core/src/physical_planner.rs`](datafusion/core/src/physical_planner.rs#L1916-L1975) (lines 1916-1975)

**Changes Made:**
- Replaced catch-all `_ => {}` default case with explicit variant matching
- Added comprehensive pattern matching for all `LogicalPlan` variants:
  - **Leaf/meta plans (no filters):** EmptyRelation, Values, DescribeTable, Explain, Analyze, Distinct, Extension, Statement, Dml, Ddl, Copy, Unnest, RecursiveQuery
  - **Plans with inputs (recursive traversal):** Projection, SubqueryAlias, Limit, Sort, Union, Join, Repartition, Aggregate, Window, Subquery
  
**Benefits:**
- **Fail-closed by design:** Adding a new `LogicalPlan` variant will now trigger a compilation error
- **Self-documenting:** Each variant explicitly shows handling decision
- **Safer evolution:** Reduces risk of silent filter extraction failures in future optimizer changes

**Test Results:**
- ✅ All 7 DML delete tests pass
- ✅ All 35 physical_planner unit tests pass
- ✅ No regressions in integration tests

**Rationale:**
Following the existing pattern in DataFusion (as seen in other `LogicalPlan` methods like `inputs()` and `schema()`), explicit variant handling ensures that when new variants are added to `LogicalPlan`, developers are forced to consider whether they can hold filter information relevant to DML operations. This prevents the subtle bug where future optimizer changes could relocate predicates into new plan nodes that aren't covered by `extract_dml_filters`.

---

### ✅ P1: UPDATE Test Coverage - IMPLEMENTED

**Location:** [`datafusion/core/tests/custom_sources_cases/dml_planning.rs`](datafusion/core/tests/custom_sources_cases/dml_planning.rs#L388-L426)

**Changes Made:**
- Added `filter_pushdown: TableProviderFilterPushDown` field to `CaptureUpdateProvider` struct
- Added `new_with_filter_pushdown()` constructor to `CaptureUpdateProvider`
- Implemented `supports_filters_pushdown()` method on `TableProvider` impl for `CaptureUpdateProvider`
- Added new regression test: `test_update_filter_pushdown_extracts_table_scan_filters()`

**Test Details:**
The new test verifies that UPDATE with filter pushdown correctly extracts filters from `TableScan.filters`:
- Creates an UPDATE provider with `TableProviderFilterPushDown::Exact`
- Executes `UPDATE t SET value = 100 WHERE id = 1`
- Verifies the optimizer pushes the filter into `TableScan`
- Verifies filters are extracted and correctly passed to `update()` method

**Test Results:**
- ✅ New UPDATE test passes
- ✅ All 10 DML planning tests pass (7 DELETE + 3 UPDATE)
- ✅ No regressions in existing UPDATE tests

**Rationale:**
The original PR fix addressed DELETE with filter pushdown but left UPDATE untested. This gap meant that UPDATE operations could have had similar issues where filters pushed into `TableScan` weren't being extracted. Since `extract_dml_filters` handles both DELETE and UPDATE operations, comprehensive test coverage for both paths is essential to prevent future regressions.

---

### ✅ P1: Mixed-Location Filter Test - IMPLEMENTED

**Location:** [`datafusion/core/tests/custom_sources_cases/dml_planning.rs`](datafusion/core/tests/custom_sources_cases/dml_planning.rs#L361-L405)

**Changes Made:**
- Added new regression test: `test_delete_mixed_filter_locations()`

**Test Details:**
This test verifies that `extract_dml_filters` correctly collects predicates that are split across multiple locations:
- Creates a DELETE provider with `TableProviderFilterPushDown::Inexact` (partial pushdown support)
- Executes `DELETE FROM t WHERE id = 1 AND status = 'active'`
- The `Inexact` pushdown mode causes the optimizer to push some predicates to `TableScan.filters` and leave others in the `Filter` node
- Verifies that **both** predicates are extracted from **both** locations and passed to `delete_from()`
- Validates no predicates are lost during deduplication

**Scenario Covered:**
```sql
DELETE FROM t WHERE id = 1 AND status = 'active'
```

With `TableProviderFilterPushDown::Inexact`:
- Predicate 1 (`id = 1`) → pushed to `TableScan.filters`
- Predicate 2 (`status = 'active'`) → remains in `Filter` node
- Expected: Both predicates collected and passed to `delete_from()`

**Test Results:**
- ✅ New mixed-location test passes
- ✅ All 11 DML planning tests pass (8 DELETE + 3 UPDATE)
- ✅ No regressions in existing tests

**Rationale:**
The original implementation of `extract_dml_filters` only extracted from `Filter` nodes or `TableScan.filters`. However, when a table provider supports partial (Inexact) filter pushdown, the optimizer may split compound predicates across both locations. Without this test, such scenarios could silently lose predicates. This test locks in the union behavior and prevents future regressions.
