# Min/Max Bytes Regression Analysis - Claude v5

## Executive Summary

The PR (commits `1eb9d9ac6^..352e69847`) successfully addressed the quadratic allocation problem in `MinMaxBytesAccumulator` but introduced **6 regressions** alongside **10 improvements**. The regressions share a common root cause: **overhead from tracking per-batch unique group statistics** (`unique_groups` counter) which wasn't present in the original implementation.

## Benchmark Results Analysis

### Improvements (10 benchmarks)
These show the optimization working as intended:
- **ultra sparse** (-92.76%): Sparse hash-based tracking vs. allocating 1M-group vectors
- **multi batch large** (-40.29%): Deferred dense allocation amortized over batches  
- **monotonic group ids** (-39.59%): Sequential dense fast path avoids scratch overhead
- **quadratic growing total groups** (-43.24%): Per-batch scratch sizing vs. growing allocations
- **sparse groups** (-27.38%): Hash-based tracking for high-cardinality workloads
- **mode transition** (-23.44%): Adaptive switching between dense/sparse strategies
- **growing total groups** (-20.59%): Better handling of expanding group domains
- **medium cardinality stable** (-3.26%): Moderate benefit from adaptive heuristics
- **extreme duplicates** (-3.02%): Consecutive duplicate detection optimization
- **dense duplicate groups** (-5.92%): Fast path for duplicate group IDs

### Regressions (6 benchmarks)  
All share the **same root cause** - overhead from unique group counting:

| Benchmark | Regression | Workload Pattern | Root Cause |
|-----------|-----------|------------------|------------|
| **dense reused accumulator** | +16.05% | Multi-batch, stable 512 groups | Redundant `unique_groups` tracking across 32 batches |
| **large dense groups** | +7.23% | Single-batch, 16K groups | Per-group iteration to count uniques in dense case |
| **dense first batch** | +6.06% | Single-batch, 512 groups | Overhead of adaptive mode selection on first batch |
| **single batch small** | +5.03% | Single-batch, 512 groups | Statistics tracking for trivial workload |
| **single batch large** | +4.57% | Single-batch, 10K groups | Unique counting in single-use accumulator |
| **sequential stable groups** | +1.59% | Multi-batch, reused groups | Minimal overhead from epoch-based tracking |

## Root Cause: Unique Groups Counting Overhead

### The Problem

The PR introduced `BatchStats::unique_groups` tracking to enable adaptive mode selection. This counter must distinguish between:
- **First touches**: Groups touched for the first time in this batch (should increment counter)
- **Revisits**: Groups already touched earlier in the same batch (should NOT increment)

### Current Implementation Issues

#### 1. **DenseInline Path** (`update_batch_dense_inline_impl`)
```rust
// Lines 827-896: Tracking logic
let mut unique_groups = 0_usize;

// Fast path detection for sequential groups
if fast_path {
    if fast_rows == 0 {
        fast_start = group_index;
        fast_last = group_index;
    } else if group_index == fast_last + 1 {
        fast_last = group_index;
    } else {
        // Fallback to mark-based tracking
        fast_path = false;
        if fast_rows > 0 {
            let fast_unique = fast_last.saturating_sub(fast_start).saturating_add(1);
            unique_groups = fast_unique;
            // ... mark all groups in fast range ...
        }
    }
}

// Slow path: check marks for each group
if !fast_path && !is_consecutive_duplicate {
    if !marks_ready {
        self.prepare_dense_inline_marks(total_num_groups);  // ❌ ALLOCATION
        marks_ready = true;
    }
    let mark = &mut self.dense_inline_marks[group_index];
    if *mark != self.dense_inline_epoch {
        *mark = self.dense_inline_epoch;
        unique_groups = unique_groups.saturating_add(1);  // ❌ PER-GROUP INCREMENT
        max_group_index = Some(match max_group_index {
            Some(current_max) => current_max.max(group_index),
            None => group_index,
        });
    }
}
```

**Issues:**
- Allocates `dense_inline_marks: Vec<u64>` (8 bytes × `total_num_groups`) even for single-batch workloads
- Checks and updates marks for every non-null group in the batch
- The "committed" fast path (after 3 stable batches) skips this, but initial batches pay full cost

#### 2. **Sequential Dense Path** (`update_batch_sequential_dense`)
```rust
// Lines 980-1001: Clean implementation
let mut unique_groups = 0_usize;

for (position, (new_val, group_index)) in iter.into_iter().zip(group_indices.iter()).enumerate() {
    let group_index = *group_index;
    debug_assert_eq!(group_index, position, "sequential dense path expects strictly sequential group ids");
    
    let Some(new_val) = new_val else {
        continue; // skip nulls
    };
    
    unique_groups = unique_groups.saturating_add(1);  // ✅ Simple increment - groups ARE unique
    
    // ... comparison and update logic ...
}
```

**Why this works:** Sequential dense groups `[0, 1, 2, ..., N-1]` are guaranteed unique by construction. No marks needed.

#### 3. **Sparse Path** (`update_batch_sparse_impl`)
```rust
// Lines 1119-1139: Hash-based tracking
let mut unique_groups = 0_usize;

match state.scratch_sparse.entry(group_index) {
    Entry::Occupied(_) => {
        // Already seen this group in this batch - skip
    }
    Entry::Vacant(vacant) => {
        vacant.insert(ScratchLocation::Existing);
        state.scratch_group_ids.push(group_index);
        unique_groups += 1;  // ✅ HashMap naturally deduplicates
        // ... dense candidate evaluation ...
    }
}
```

**Why this works:** The `HashMap` naturally prevents double-counting revisits. But hash lookups + potential dense candidate evaluation add overhead.

### Why This Causes Regressions

| Benchmark | Why `unique_groups` Tracking Hurts |
|-----------|-----------------------------------|
| **dense reused accumulator** | Allocates + checks 512-entry mark vector 32 times unnecessarily (groups are stable) |
| **large dense groups** | Allocates 16K×8 = 128KB mark vector for single-batch use, checks all 16K marks |
| **dense first batch** | Overhead of mode detection + potential mark allocation for 512 groups |
| **single batch {small,large}** | Statistics tracking infrastructure for accumulators that process one batch and die |
| **sequential stable groups** | Minimal overhead from epoch tracking, but still measurable |

## Tasks to Fix Regressions

### Task 1: Eliminate Unique Counting in Committed DenseInline Mode ⚡ HIGH IMPACT

**Problem:** After 3 stable batches, the accumulator commits to DenseInline mode but still tracks marks/stats for the first few batches.

**Solution:**
```rust
// In update_batch_dense_inline_impl, at the start:
if self.dense_inline_committed {
    // Skip ALL statistics tracking
    return self.update_batch_dense_inline_committed(iter, group_indices, total_num_groups, cmp);
}
```

**Expected Fix:** `dense reused accumulator` regression (+16.05%) should improve significantly as batches 4-32 skip mark allocation entirely.

**Status:** Already implemented (line 769-772), but commit happens too late (batch 3+).

**Improvement:** Commit earlier (after batch 2 if stable) or use a simpler heuristic:
- If `processed_batches == 1` and `total_num_groups` hasn't changed, assume stable
- Skip mark allocation from batch 2 onwards

---

### Task 2: Fast Path for Pure Sequential Dense Groups 🎯 MEDIUM IMPACT

**Problem:** Sequential dense groups `[0,1,2,...,N-1]` are detected early (line 716-726) but still increment `unique_groups` per-item when not using the committed path.

**Solution:**
The sequential dense path already does this correctly! The regression is in **non-sequential dense** workloads.

**Actual Issue:** The `update_batch_sequential_dense` function is called, but benchmarks like `dense_first_batch` and `large_dense_groups` hit the DenseInline path first (via Undecided mode) because the heuristic check happens inside `update_batch`.

**Fix:** Reorder the fast-path detection:
```rust
fn update_batch<'a, F, I>(...) -> Result<()> {
    // MOVE THIS CHECK TO THE TOP, before mode switching
    if group_indices.len() == total_num_groups && /* sequential check */ {
        let stats = self.update_batch_sequential_dense(...)?;
        self.record_batch_stats(stats, total_num_groups);
        return Ok(());
    }
    
    // Then handle mode-based dispatch
    match self.workload_mode {
        // ...
    }
}
```

**Expected Fix:**
- `dense_first_batch` (+6.06%) → should match original performance
- `large_dense_groups` (+7.23%) → should match original performance  
- `single_batch_small` (+5.03%) → should match original performance
- `single_batch_large` (+4.57%) → should match original performance

**Status:** The check IS already at the top (lines 716-726), but the Undecided mode (line 787) goes to DenseInline first for small workloads, which then allocates marks.

**Actual Fix:** In Undecided mode, for sequential dense patterns, call `update_batch_sequential_dense` directly without statistics:
```rust
WorkloadMode::Undecided => {
    // Detect sequential dense BEFORE choosing mode
    if is_sequential_dense(group_indices, total_num_groups) {
        let stats = self.update_batch_sequential_dense(...)?;
        self.record_batch_stats(stats, total_num_groups);
        return Ok(());
    }
    
    let stats = if total_num_groups <= DENSE_INLINE_MAX_TOTAL_GROUPS {
        self.update_batch_dense_inline_impl(...)?
    } else {
        self.update_batch_sparse_impl(...)?
    };
    self.record_batch_stats(stats, total_num_groups);
    Ok(())
}
```

---

### Task 3: Lazy Mark Allocation with Better Heuristics 🔧 LOW-MEDIUM IMPACT

**Problem:** `dense_inline_marks_ready` defaults to `false`, but gets set to `true` after the first batch with reuse evidence (line 1256). Single-batch workloads still pay allocation cost if they're non-sequential.

**Current Logic:**
```rust
// Line 1251-1256
if self.should_use_dense_inline(total_num_groups, stats.unique_groups) {
    if !matches!(self.workload_mode, WorkloadMode::DenseInline) {
        self.enter_dense_inline_mode();
    }
    self.workload_mode = WorkloadMode::DenseInline;
    self.dense_inline_marks_ready = true;  // ❌ Triggers allocation on NEXT batch
}
```

**Issue:** This still means the second batch allocates marks. For `dense_first_batch` (single batch), it doesn't matter. But for `dense reused accumulator` (32 batches), batch 2 allocates unnecessary marks.

**Solution:** Only set `marks_ready = true` if we've seen evidence of **non-sequential** access:
```rust
// In update_batch_dense_inline_impl
if fast_path {
    // Perfectly sequential - no marks needed
    self.dense_inline_marks_ready = false;
} else {
    // Non-sequential - will need marks
    if !marks_ready {
        self.prepare_dense_inline_marks(total_num_groups);
        marks_ready = true;
        self.dense_inline_marks_ready = true;  // Remember for next batch
    }
}
```

**Expected Fix:** `dense reused accumulator` (+16.05%) gets some additional improvement as batch 2+ stay on fast path.

---

### Task 4: Skip Statistics in Single-Batch Scenarios 📊 LOW IMPACT

**Problem:** The `Undecided` mode always gathers statistics even if the accumulator processes only one batch.

**Solution:** Add a "single-batch fast path" heuristic:
```rust
WorkloadMode::Undecided => {
    // If this looks like a single-batch workload, skip adaptive selection
    if self.processed_batches == 0 && is_likely_single_batch(total_num_groups, group_indices) {
        // Use simple path without statistics
        return self.update_batch_sequential_dense_or_simple(...);
    }
    
    // Otherwise, gather statistics and choose mode
    let stats = ...;
    self.record_batch_stats(stats, total_num_groups);
    Ok(())
}
```

**Heuristic for "likely single-batch":**
- `processed_batches == 0` (first batch)
- `total_num_groups` matches `group_indices.len()` (all groups present)
- No pre-existing state (`self.min_max.is_empty()`)

**Expected Fix:** Minor improvements to `single_batch_{small,large}` (+5.03%, +4.57%).

**Tradeoff:** Might misclassify some multi-batch workloads, forcing a mode switch on batch 2.

---

### Task 5: Optimize Consecutive Duplicate Detection 🏎️ MICRO-OPTIMIZATION

**Problem:** The code already detects consecutive duplicates (line 851):
```rust
let is_consecutive_duplicate = last_group_index == Some(group_index);
last_group_index = Some(group_index);
```

But it's only used in the slow path (line 893):
```rust
if !fast_path && !is_consecutive_duplicate {
    // Check marks...
}
```

**Observation:** The `dense duplicate groups` benchmark actually **improved** (-5.92%), so this optimization is working! This is NOT a regression source.

**No action needed.**

---

### Task 6: Profile and Validate Memory Layout 🧪 VALIDATION

**Goal:** Confirm that mark allocation is the bottleneck, not the counter increments themselves.

**Steps:**
1. Run benchmarks with `perf` or Instruments (macOS)
2. Look for hotspots in:
   - `prepare_dense_inline_marks` (allocation + zeroing)
   - The mark checking loop (lines 893-899)
   - `saturating_add` for unique_groups (cheap, but worth checking)

3. Validate assumptions:
   - Is `prepare_dense_inline_marks` called in `dense_first_batch`? (Should be NO after Task 2)
   - Is `prepare_dense_inline_marks` called in `dense reused accumulator`? (Should be NO after Task 1)

**Expected Findings:**
- Most regression time is in `vec!` allocation or `memset` zeroing the mark vector
- Actual counting (`saturating_add`) is negligible (<1% of time)

**Deliverable:** Confirm which tasks to prioritize based on profiling data.

---

## Implementation Priority

### Phase 1: High-Impact Fixes (Target: Eliminate 80% of Regressions)
1. ✅ **Task 2**: Fix Undecided → Sequential Dense routing
   - Fixes: `dense_first_batch`, `large_dense_groups`, `single_batch_{small,large}`
   - Expected total improvement: ~5-7% across 4 benchmarks

2. ✅ **Task 1**: Earlier commitment or better reuse detection  
   - Fixes: `dense reused accumulator`
   - Expected improvement: ~10-15% (from +16% to +1-5%)

### Phase 2: Medium-Impact Optimizations  
3. **Task 3**: Lazy mark allocation refinement
   - Incremental improvement to Task 1 results
   - Expected improvement: ~2-3% additional

### Phase 3: Low-Impact / Validation
4. **Task 4**: Single-batch heuristic (optional)
   - Minor improvements, risk of misclassification
   - Expected improvement: ~1-2%

5. **Task 6**: Profiling validation
   - Confirms optimization effectiveness
   - Guides future work

---

## Success Criteria

### Target: Zero Statistically Significant Regressions

After implementing Tasks 1-3, expect:
- ✅ `dense reused accumulator`: +16.05% → **-2%** (slight improvement over baseline)
- ✅ `large dense groups`: +7.23% → **±0%** (match baseline)
- ✅ `dense first batch`: +6.06% → **±0%** (match baseline)  
- ✅ `single batch large`: +4.57% → **±0%** (match baseline)
- ✅ `single batch small`: +5.03% → **±0%** (match baseline)
- ✅ `sequential stable groups`: +1.59% → **-1%** (slight improvement)

**All improvements retained:** The 10 improvements should remain unchanged as they rely on the sparse/mode-switching logic, which is unaffected by dense-path optimizations.

---

## Long-Term Recommendations

1. **Remove `unique_groups` from `BatchStats` for committed modes**
   - Once committed, the adaptive heuristic is no longer needed
   - Can skip all statistics tracking entirely

2. **Consider a "compilation" phase**
   - After 2-3 batches, "compile" the optimal code path (sequential dense, dense inline, sparse)
   - Use function pointers or enum dispatch to eliminate mode-checking overhead

3. **Add a benchmark for "mixed sequential/non-sequential"**
   - Current benchmarks are either purely sequential or purely random
   - Real workloads might have patterns like `[0,1,2,5,6,7,10,11,12]`
   - Validate that the fast-path detection doesn't regress mixed patterns

4. **Document the performance model**
   - Add comments explaining when each path is optimal
   - Include big-O complexity for each mode
   - Help future maintainers avoid reintroducing quadratic behavior

---

## Conclusion

The root cause of all 6 regressions is **per-batch unique group counting overhead**, specifically:
1. Allocating mark vectors for single-batch or stable-reused workloads (Tasks 1, 2)
2. Per-group mark checking in dense inline mode (Tasks 1, 3)
3. Statistics gathering for single-batch scenarios (Task 4)

The fixes are **surgical and low-risk**:
- Reorder fast-path detection (Task 2)
- Commit to dense inline earlier (Task 1)  
- Refine mark allocation heuristics (Task 3)

No fundamental algorithm changes needed - just better routing to avoid overhead in scenarios where statistics aren't beneficial.

**Estimated effort:** 2-4 hours of implementation + testing
**Estimated improvement:** Eliminate 5-6 regressions, retain all 10 improvements
