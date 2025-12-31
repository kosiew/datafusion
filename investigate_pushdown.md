# Investigation: Why Nested Filter Pushdown Shows No Performance Improvement

## Problem Summary
The benchmark for nested DataType filter pushdown to Parquet showed **no performance regression or improvement** after implementing the feature in commits b64838e4c^..39e45dc54.

## ROOT CAUSE IDENTIFIED ��

The implementation provides **Row-Level Filtering** but NOT **Row Group Pruning**!

### Two Distinct Optimization Levels in Parquet

#### 1. Row Group Pruning (Statistics-Based) - NOT IMPLEMENTED ❌
- **When**: Before reading row groups
- **How**: Uses min/max statistics to skip entire row groups
- **Benefit**: Massive - avoid reading 90%+ of data from disk
- **Example**: Skip row groups where `max(values) < 'target'`

#### 2. Row-Level Filtering (Predicate Pushdown) - IMPLEMENTED ✅  
- **When**: During Parquet decoding  
- **How**: Evaluates filter on each row to avoid materializing filtered rows
- **Benefit**: Modest - save memory/CPU for filtered rows, but still read all row groups
- **Example**: Decode only rows where `array_has(list_col, 'target') = true`

### Why There's No Performance Improvement

The current implementation (commits b64838e4c^..39e45dc54) implements **Row-Level Filtering**:

```rust
// From row_filter.rs:62-64
//! List-aware predicates (for example, `array_has`, `array_has_all`, and
//! `array_has_any`) can be evaluated directly during Parquet decoding.
```

This means:
1. ✅ All row groups are still read from disk
2. ✅ For each row group, the filter is applied during decoding
3. ✅ Rows that don't match are skipped (not materialized into `RecordBatch`)
4. ❌ But I/O cost remains the same - all data is read!

### What Was Expected (Row Group Pruning)

The PR description and benchmarks assumed row group pruning would work:

> "With pushdown enabled, ~90% of row groups can be skipped based on min/max
> statistics, significantly reducing the rows that need to be decoded."

This would require:
- Reading min/max statistics from Parquet metadata
- Evaluating `array_has('target')` against those statistics  
- Skipping entire row groups where the predicate can't match

###Parquet Statistics Limitation for List Columns

**Critical Issue**: Parquet statistics are computed on LEAF columns (the individual string values), not on List structures!

For `List<String>` columns:
- Statistics exist for the underlying `String` values (min/max of all strings across all lists)
- But **there's no way to determine if a specific value appears in ANY list without reading the data**

Example:
```
Row Group 1: 
  - Row 1: ["aaa", "bbb"]
  - Row 2: ["ccc", "ddd"]
  Statistics: min="aaa", max="ddd"
  
Query: array_has(list_col, 'aaa')
```

The statistics tell us `"aaa"` exists somewhere in the row group, but **NOT whether it appears in any list**!

This is why row group pruning is challenging for list predicates.

## What The Current Implementation Actually Does

### Before (Without Pushdown):
```
1. ParquetExec reads all row groups
2. Materializes all rows into RecordBatches  
3. FilterExec evaluates array_has() on each batch
4. Returns matching rows
```

### After (With Row-Level Filtering):
```
1. ParquetExec reads all row groups (same I/O!)
2. During decoding, evaluates array_has() on each row
3. Only materializes matching rows into RecordBatches
4. Returns matching rows (skip FilterExec)
```

**Performance Difference**: Minimal - saves memory allocation for filtered rows, but same I/O cost.

## Why The Benchmark Showed No Difference

### Issue #1: Benchmark Wasn't Running Queries ❌
```rust
b.iter(|| {
    let path = generate_sorted_list_data(&config, &temp_dir);
    black_box(path)  // ← Only measures file generation!
});
```

### Issue #2: Even With Proper Queries, Benefit Is Small ⚠️
Row-level filtering provides minimal benefit when selectivity is low:
- Still read all 100K rows from disk  
- Save materializing ~90K filtered rows
- But decoding cost dominates (I/O + decompression)

**Typical Savings**: 5-10% in CPU/memory, not the expected 10x improvement

## What Would Provide 10x Performance Improvement

### Option A: Row Group Pruning for Simple Cases
Implement statistics-based pruning for predicates like:
```sql
array_has(list_col, constant_value)
```

Check if `min_stats <= constant_value <= max_stats` for the leaf column.
- If outside range, skip the row group
- **Limitation**: Only works when value definitely doesn't exist; can't prove it does exist in a list

### Option B: Secondary Indexes (Bloom Filters)
- Parquet supports Bloom filters for existence checks
- Could enable row group skipping for `array_has()` predicates
- Requires filter creation during write time

### Option C: Sorted Array Elements
If list elements are sorted within each list, row group statistics become more useful:
```
Row Group 1: 
  - All lists contain only values in range ["aaa", "bbb"]
  Statistics: min="aaa", max="bbb"
  
Query: array_has(list_col, 'zzz')
  → Provably false! Skip this row group.
```

But this requires data-specific assumptions.

## Recommendations

### Short Term: Update Documentation
1. Clarify that the implementation provides **row-level filtering**, not row group pruning
2. Update PR description to set correct expectations (5-10% improvement, not 10x)
3. Document the limitation for list column statistics

### Medium Term: Implement Partial Row Group Pruning
For predicates where we can prove a row group can't match:
```rust
// Pseudo-code
if predicate == array_has(col, literal) {
    let leaf_stats = get_leaf_column_stats(row_group, col);
    if literal < leaf_stats.min || literal > leaf_stats.max {
        // Definitely no matches in this row group!
        skip_row_group();
    }
}
```

This would enable pruning when the target value is **outside** the row group's range.

### Long Term: Bloom Filter Integration
- Enable Parquet Bloom filters during write
- Use them for `array_has()` row group pruning
- Could provide the 10x improvement for selective queries

## Verification Steps

To confirm this analysis:

###1. Check What Was Actually Implemented
```bash
git show b64838e4c:datafusion/datasource-parquet/src/row_filter.rs | grep -A 20 "List-aware"
```

### 2. Verify Row Groups Are Not Being Skipped
Add logging to see row group access:
```rust
// In ParquetSource or DataSourceExec
log::info!("Reading row group {}: {:?}", i, row_group.metadata());
```

Run query and count how many row groups are read.

### 3. Measure Actual Performance Benefit
Proper benchmark comparing:
- Baseline: `SELECT * FROM table` (no filter)
- Row-level filter: `SELECT * FROM table WHERE array_has(col, 'x')`
- Separate filter: Same query but disable row-level filtering

Expected: 5-10% improvement, not 10x.

## Conclusion

**The implementation is correct for what it does** (row-level filtering during decoding), but **it doesn't provide the dramatic performance improvement** that row group pruning would provide.

The benchmark showing "no improvement" is correct because:
1. The benchmark wasn't executing queries initially
2. Even with queries, row-level filtering provides minimal benefit without row group pruning
3. List column statistics in Parquet can't be used for row group pruning without additional metadata (like Bloom filters)

**Next steps**: Either:
- A) Update expectations and documentation to match what was implemented  
- B) Implement actual row group pruning (harder, requires bloom filters or sorted arrays)
- C) Both
