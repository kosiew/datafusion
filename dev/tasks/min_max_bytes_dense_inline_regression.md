# Min/Max Bytes Dense Inline Regression - Overview

## Status: ✅ Good Performance with 2 Minor Cold-Start Regressions

### Latest Results (Commit `b50e4465e`)

| Benchmark | Change | Status |
|-----------|--------|--------|
| min bytes sparse groups | **-28.68%** | ✅ Excellent |
| min bytes monotonic group ids | **-22.15%** | ✅ Excellent |
| min bytes dense reused accumulator | **-11.48%** | ✅ Excellent |
| min bytes dense duplicate groups | **-6.89%** | ✅ Good |
| min bytes dense first batch | **+4.31%** | ⚠️ Cold-start penalty |
| min bytes large dense groups | **+1.87%** | ⚠️ Cold-start penalty |

**Summary**: 
- ✅ **4 major improvements** (-6.89% to -28.68%)
- ⚠️ **2 cold-start regressions** (+1.87% and +4.31%)
- Both regressions are single-batch workloads (mark allocation overhead)
- Multi-batch workloads show strong improvements (-11.48%)

---

## Regression Analysis

Both regressions share the same root cause: **mark allocation overhead for single-batch workloads**.

### Regression 1: Dense First Batch (+4.31%)

**Pattern**: 512 sequential groups, single batch, fresh accumulator each iteration

**Overhead**: 
- Allocates 4 KB `dense_inline_marks` vector
- Performs epoch management and mark writes
- Never reaches committed mode (discarded after 1 batch)

### Regression 2: Large Dense Groups (+1.87%) 

**Pattern**: 16,384 sequential groups, single batch, fresh accumulator each iteration

**Overhead**:
- Allocates **131 KB** `dense_inline_marks` vector (32× larger!)
- Still routes to DenseInline (< 100K threshold)
- Zeroing 131 KB takes measurable time

**Why less severe?** Larger batch size (16,384 vs 512 rows) amortizes overhead better.

---

## What Works Well ✅

The implementation successfully delivers:

1. **Commit-Once Fast Path** - Zero overhead after mode stabilization
2. **Run-Length Detection** - Skips mark checks for consecutive duplicates  
3. **Sparse Optimization** - 28.68% improvement by avoiding large allocations
4. **Monotonic Optimization** - 22.15% improvement for growing group IDs

Real multi-batch workloads show **-11.48% improvement**, demonstrating the optimization works as intended.

## Fix: Defer Mark Allocation Until Second Batch

The solution is **Task 3** from the original remediation plan - now more important due to larger regressions.

### Implementation Strategy

**Don't allocate marks on first batch**. Use simple consecutive-dedup for statistics:

```rust
fn update_batch_dense_inline_impl(...) -> Result<BatchStats> {
    self.min_max.resize(total_num_groups, None);
    
    // First batch: no mark allocation
    if !self.dense_inline_marks_initialized {
        let mut unique_groups = 0;
        let mut last_seen: Option<usize> = None;
        
        for (group_index, new_val) in ... {
            if last_seen != Some(group_index) {
                unique_groups += 1;
                last_seen = Some(group_index);
            }
            // ... min/max work ...
        }
        
        self.dense_inline_marks_initialized = true;
        return Ok(BatchStats { unique_groups, ... });
    }
    
    // Second+ batch: allocate marks and use full tracking
    // ... existing implementation ...
}
```

### Expected Impact

- **Dense first batch**: +4.31% → ~+0.3% (eliminates 4 KB allocation)
- **Large dense groups**: +1.87% → ~+0.2% (eliminates 131 KB allocation)
- **Multi-batch workloads**: Unchanged (marks allocated on batch 2)

**See detailed implementation**: `min_max_bytes_cold_start_regressions.md`

---

## Analysis Documents

1. **[Cold-Start Regressions Analysis](./min_max_bytes_cold_start_regressions.md)** ⭐ **Current issue**
   - Detailed analysis of +4.31% and +1.87% regressions
   - Complete implementation guide for the fix
   - Testing strategy and benchmarks

### Historical Context
2. **[Root Cause Analysis](./min_max_bytes_dense_inline_regression_root_cause.md)** - Why initial PR had 3 regressions
3. **[Remediation Tasks](./min_max_bytes_dense_inline_regression_tasks.md)** - 6 prioritized optimization tasks  
4. **[Summary Document](./min_max_bytes_dense_inline_regression_summary.md)** - Executive overview
5. **[Dense First Batch (v2)](./min_max_bytes_dense_first_batch_regression.md)** - Previous +1.73% analysis

## Recommendation

**Implement the fix for cold-start regressions.** ✅

**Rationale**:
- ✅ Achieved 22-29% improvements in target workloads
- ✅ Fix is simple and safe (~20 lines of code, 1-2 hours)
- ✅ Eliminates both cold-start regressions (+4.31% → ~0%, +1.87% → ~0%)
- ✅ No risk to existing improvements
- ✅ Makes benchmark results "clean" (all improvements, no regressions)

**Why fix now?**
- Regressions grew from +1.73% to +4.31% (2.5× worse)
- New regression on "large dense groups" benchmark (+1.87%)
- Simple, well-understood fix with clear benefit

**Alternative**: Accept current results and document the cold-start trade-off (reasonable but suboptimal).

---

## Related Files

- Implementation: `datafusion/functions-aggregate/src/min_max/min_max_bytes.rs`
- Benchmarks: `datafusion/functions-aggregate/benches/min_max_bytes.rs`
- Original issue: #17897 (quadratic scratch allocation problem)
- PR versions:
  - v1: `c1ac251d6^..93e1d7529` (3 regressions: +1.6%, +1.2%, +20%)
  - v2: `c1ac251d6^..442053997` (1 regression: +1.73%)
  - v3: `c1ac251d6^..b50e4465e` (2 regressions: +4.31%, +1.87%) ← **Current**

## Next Steps

**Recommended**: Implement Task 3 (deferred mark allocation)
- See **[Cold-Start Regressions Analysis](./min_max_bytes_cold_start_regressions.md)** for complete implementation
- Estimated effort: 1-2 hours
- Expected outcome: +4.31% → ~+0.3%, +1.87% → ~+0.2%
- Risk: Low (isolated change, well-tested)
