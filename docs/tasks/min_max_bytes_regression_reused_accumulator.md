# Min/Max Bytes Regression: Dense Reused Accumulator

**Date:** October 9, 2025  
**Status:** Analysis Complete, Implementation Pending  
**Severity:** Medium (20% regression in specific workload)  
**PR Range:** `c1ac251d6^..b4c40ad32`

---

## Executive Summary

The PR introducing sparse workload optimizations to `MinMaxBytesAccumulator` caused a **19.80% regression** in the "dense reused accumulator" benchmark. The root cause is that the sequential fast path optimization allocates a `locations` vector on every batch, while the DenseInline committed mode achieves zero-allocation steady-state after initial learning. The fix requires extending the sequential fast path with stability tracking and a committed mode.

---

## Benchmark Results

### Impact Summary

| Benchmark                          | Mean Change | P-value  | Status       |
|------------------------------------|-------------|----------|--------------|
| min bytes dense first batch        | **-2.05%**  | 0.000000 | ✅ Improved   |
| min bytes growing total groups     | **-2.40%**  | 0.000000 | ✅ Improved   |
| min bytes monotonic group ids      | **-2.42%**  | 0.000000 | ✅ Improved   |
| min bytes multi batch large        | **-2.50%**  | 0.000000 | ✅ Improved   |
| min bytes single batch small       | **-3.11%**  | 0.000000 | ✅ Improved   |
| **min bytes dense reused accumulator** | **+19.80%** | 0.000000 | ❌ **Regressed** |

**Summary:** 5 improvements, 1 regression (all statistically significant, p < 0.05)

---

## Root Cause Analysis

### The Sequential Fast Path

The PR introduced an optimization at the start of `update_batch` that detects perfectly sequential dense group indices `[0, 1, 2, ..., N-1]`:

```rust
// Fast path: detect perfectly sequential dense group indices [0, 1, 2, ..., N-1]
if group_indices.len() == total_num_groups
    && !group_indices.is_empty()
    && group_indices[0] == 0
    && group_indices[total_num_groups - 1] == total_num_groups - 1
{
    return self.update_batch_sequential_dense(
        iter,
        group_indices,
        total_num_groups,
        cmp,
    );
}
```

### The Problem

`update_batch_sequential_dense` **always** allocates a new `locations` vector:

```rust
fn update_batch_sequential_dense(...) -> Result<()> {
    self.resize_min_max(total_num_groups);
    
    // ❌ Allocated on EVERY batch
    let mut locations = vec![SequentialDenseLocation::ExistingMinMax; total_num_groups];
    
    // ... process batch ...
}
```

### The Competing Path: DenseInline Committed

The DenseInline mode, after observing `DENSE_INLINE_STABILITY_THRESHOLD` (3) consecutive stable batches, transitions to a **committed mode** that performs **zero allocations**:

```rust
fn update_batch_dense_inline_committed(...) -> Result<()> {
    self.resize_min_max(total_num_groups);
    
    // ✅ No allocations - direct in-place updates
    for (&group_index, new_val) in group_indices.iter().zip(iter.into_iter()) {
        let Some(new_val) = new_val else { continue; };
        
        let should_replace = match self.min_max[group_index].as_ref() {
            Some(existing_val) => cmp(new_val, existing_val.as_ref()),
            None => true,
        };
        
        if should_replace {
            self.set_value(group_index, new_val);
        }
    }
    Ok(())
}
```

### Why the Benchmark Regressed

The **"dense reused accumulator"** benchmark:

1. **Setup:** Sequential group indices `[0, 1, 2, ..., 511]` (BATCH_SIZE = 512)
2. **Workload:** Processes **32 batches** (MONOTONIC_BATCHES) with identical indices
3. **Pre-PR behavior:** Would have used DenseInline mode, achieving zero-allocation committed path after batch 3
4. **Post-PR behavior:** Sequential fast path triggers on **all 32 batches**, allocating 512-element vector each time

**Regression calculation:**
- **Before:** 3 batches with allocations + 29 batches zero-allocation
- **After:** 32 batches with allocations
- **Extra work:** 29 unnecessary allocations × (malloc + zeroing + dealloc + cache misses)
- **Measured impact:** **+19.80% slowdown**

---

## Implementation Plan

### Overview

Extend the sequential fast path with **stability tracking** and a **committed mode** that mirrors the DenseInline optimization but preserves the sequential-specific implementation.

---

### Task 1: Design Sequential Committed Mode

**Status:** Not Started

Add three new fields to `MinMaxBytesState`:

```rust
struct MinMaxBytesState {
    // ... existing fields ...
    
    /// Number of consecutive batches processed via sequential fast path
    /// with stable total_num_groups.
    sequential_stable_batches: usize,
    
    /// Whether the sequential fast path has committed to zero-allocation mode.
    sequential_committed: bool,
    
    /// The committed group count for the sequential path. Used to detect
    /// domain growth and revoke commitment.
    sequential_committed_groups: usize,
}
```

**Rationale:**
- `sequential_stable_batches`: Tracks consecutive stable batches (analogous to `dense_inline_stable_batches`)
- `sequential_committed`: Enables routing to committed implementation after threshold
- `sequential_committed_groups`: Detects when `total_num_groups` grows, requiring mode reset

---

### Task 2: Implement Sequential Dense Committed Path

**Status:** Not Started

Create a new method that performs **zero-allocation in-place updates**:

```rust
/// Committed fast path for sequential dense indices after stability threshold.
///
/// No per-batch allocations: compares and updates values directly in self.min_max.
fn update_batch_sequential_dense_committed<'a, F, I>(
    &mut self,
    iter: I,
    group_indices: &[usize],
    total_num_groups: usize,
    mut cmp: F,
) -> Result<()>
where
    F: FnMut(&[u8], &[u8]) -> bool + Send + Sync,
    I: IntoIterator<Item = Option<&'a [u8]>>,
{
    self.resize_min_max(total_num_groups);
    
    for (&group_index, new_val) in group_indices.iter().zip(iter.into_iter()) {
        let Some(new_val) = new_val else {
            continue;
        };
        
        debug_assert!(
            group_index < total_num_groups,
            "sequential fast path requires valid indices"
        );
        
        let should_replace = match self.min_max[group_index].as_ref() {
            Some(existing_val) => cmp(new_val, existing_val.as_ref()),
            None => true,
        };
        
        if should_replace {
            self.set_value(group_index, new_val);
        }
    }
    
    Ok(())
}
```

**Key properties:**
- ✅ Zero allocations per batch
- ✅ Single-pass iteration
- ✅ Direct in-place comparison
- ✅ Minimal overhead (only iterator + comparison + conditional set_value)

---

### Task 3: Add Stability Tracking

**Status:** Not Started

Modify `update_batch_sequential_dense` to track stability and transition to committed mode:

```rust
fn update_batch_sequential_dense<'a, F, I>(
    &mut self,
    iter: I,
    group_indices: &[usize],
    total_num_groups: usize,
    mut cmp: F,
) -> Result<()>
where
    F: FnMut(&[u8], &[u8]) -> bool + Send + Sync,
    I: IntoIterator<Item = Option<&'a [u8]>>,
{
    self.resize_min_max(total_num_groups);
    
    let mut locations = vec![SequentialDenseLocation::ExistingMinMax; total_num_groups];
    
    // ... existing processing logic ...
    
    // Track stability for commitment
    if !self.sequential_committed {
        if self.sequential_committed_groups == 0 {
            // First sequential batch
            self.sequential_committed_groups = total_num_groups;
            self.sequential_stable_batches = 1;
        } else if self.sequential_committed_groups == total_num_groups {
            // Stable batch with matching group count
            self.sequential_stable_batches = self.sequential_stable_batches.saturating_add(1);
            
            if self.sequential_stable_batches >= DENSE_INLINE_STABILITY_THRESHOLD {
                self.sequential_committed = true;
            }
        } else {
            // Group count changed - reset tracking
            self.sequential_committed_groups = total_num_groups;
            self.sequential_stable_batches = 1;
        }
    }
    
    Ok(())
}
```

---

### Task 4: Route to Committed Path

**Status:** Not Started

Update the sequential fast path entry point in `update_batch`:

```rust
fn update_batch<'a, F, I>(
    &mut self,
    iter: I,
    group_indices: &[usize],
    total_num_groups: usize,
    cmp: F,
) -> Result<()>
where
    F: FnMut(&[u8], &[u8]) -> bool + Send + Sync,
    I: IntoIterator<Item = Option<&'a [u8]>>,
{
    // Fast path: detect perfectly sequential dense group indices [0, 1, 2, ..., N-1]
    if group_indices.len() == total_num_groups
        && !group_indices.is_empty()
        && group_indices[0] == 0
        && group_indices[total_num_groups - 1] == total_num_groups - 1
    {
        // Handle domain growth: revoke commitment if group count increased
        if self.sequential_committed && total_num_groups > self.sequential_committed_groups {
            self.sequential_committed = false;
            self.sequential_stable_batches = 0;
            self.sequential_committed_groups = 0;
        }
        
        // Route to committed or learning path
        if self.sequential_committed && total_num_groups == self.sequential_committed_groups {
            return self.update_batch_sequential_dense_committed(
                iter,
                group_indices,
                total_num_groups,
                cmp,
            );
        } else {
            return self.update_batch_sequential_dense(
                iter,
                group_indices,
                total_num_groups,
                cmp,
            );
        }
    }
    
    // ... rest of update_batch logic ...
}
```

---

### Task 5: Handle Domain Growth

**Status:** Not Started

The logic is included in Task 4 above. When `total_num_groups` grows beyond `sequential_committed_groups`, reset the sequential state to re-enter learning mode. This prevents committed mode from operating with stale assumptions.

**Edge cases:**
- Domain shrinks: Commitment remains valid (committed_groups is an upper bound)
- Domain grows: Revoke commitment, reset counters, re-learn
- Empty batches: No state change (counters frozen)

---

### Task 6: Reset State in Transitions

**Status:** Not Started

Update `reset_after_full_emit` to clear sequential state:

```rust
fn reset_after_full_emit(&mut self) {
    self.total_data_bytes = 0;
    self.populated_groups = 0;
    // ... existing resets ...
    
    // Reset sequential fast path state
    self.sequential_stable_batches = 0;
    self.sequential_committed = false;
    self.sequential_committed_groups = 0;
    
    // ... rest of resets ...
}
```

Also verify that mode transitions (e.g., `enter_sparse_mode`, `enter_simple_mode`) do not interfere with sequential state. Since the sequential fast path is independent and checked before mode dispatch, no additional clearing should be needed in those transitions.

---

### Task 7: Add Test Coverage

**Status:** Not Started

Write a comprehensive test in `min_max_bytes.rs`:

```rust
#[test]
fn sequential_dense_commits_after_stable_batches() {
    let mut state = MinMaxBytesState::new(DataType::Utf8);
    let total_groups = 64_usize;
    let group_indices: Vec<usize> = (0..total_groups).collect();
    let values: Vec<&[u8]> = (0..total_groups)
        .map(|i| format!("value_{:02}", i).into_bytes())
        .collect();
    
    // Process DENSE_INLINE_STABILITY_THRESHOLD batches
    for batch in 0..DENSE_INLINE_STABILITY_THRESHOLD {
        let iter = values.iter().map(|v| Some(v.as_slice()));
        state
            .update_batch(iter, &group_indices, total_groups, |a, b| a < b)
            .expect("sequential batch");
        
        if batch < DENSE_INLINE_STABILITY_THRESHOLD - 1 {
            assert!(!state.sequential_committed);
        } else {
            assert!(state.sequential_committed);
            assert_eq!(state.sequential_committed_groups, total_groups);
        }
    }
    
    // Process additional batches to verify committed path is used
    for _ in 0..3 {
        let iter = values.iter().map(|v| Some(v.as_slice()));
        state
            .update_batch(iter, &group_indices, total_groups, |a, b| a < b)
            .expect("committed batch");
        
        assert!(state.sequential_committed);
    }
    
    // Verify values are correct
    for i in 0..total_groups {
        let expected = format!("value_{:02}", i);
        assert_eq!(state.min_max[i].as_deref(), Some(expected.as_bytes()));
    }
}

#[test]
fn sequential_dense_revokes_commitment_on_domain_growth() {
    let mut state = MinMaxBytesState::new(DataType::Utf8);
    let initial_groups = 32_usize;
    let initial_indices: Vec<usize> = (0..initial_groups).collect();
    let initial_values: Vec<&[u8]> = (0..initial_groups)
        .map(|i| format!("val_{:02}", i).into_bytes())
        .collect();
    
    // Commit with initial group count
    for _ in 0..=DENSE_INLINE_STABILITY_THRESHOLD {
        let iter = initial_values.iter().map(|v| Some(v.as_slice()));
        state
            .update_batch(iter, &initial_indices, initial_groups, |a, b| a < b)
            .expect("initial batch");
    }
    
    assert!(state.sequential_committed);
    assert_eq!(state.sequential_committed_groups, initial_groups);
    
    // Expand domain
    let expanded_groups = 64_usize;
    let expanded_indices: Vec<usize> = (0..expanded_groups).collect();
    let expanded_values: Vec<&[u8]> = (0..expanded_groups)
        .map(|i| format!("val_{:02}", i).into_bytes())
        .collect();
    
    let iter = expanded_values.iter().map(|v| Some(v.as_slice()));
    state
        .update_batch(iter, &expanded_indices, expanded_groups, |a, b| a < b)
        .expect("expanded batch");
    
    // Commitment should be revoked
    assert!(!state.sequential_committed);
    assert_eq!(state.sequential_stable_batches, 1); // Re-entered learning
    assert_eq!(state.sequential_committed_groups, expanded_groups);
}
```

---

### Task 8: Verify Benchmark Improvement

**Status:** Not Started

Run the specific benchmark to confirm regression is fixed:

```bash
cd /Users/kosiew/GitHub/datafusion
cargo bench --bench min_max_bytes -- "min bytes dense reused accumulator"
```

**Expected outcome:**
- ✅ Regression eliminated: performance matches or exceeds pre-PR baseline
- ✅ First 3 batches: minimal overhead from tracking (< 1%)
- ✅ Batches 4+: zero-allocation committed path, matching DenseInline committed performance
- ✅ Overall: ~20% improvement vs. current PR, reaching baseline or better

**Also verify no regressions in other benchmarks:**

```bash
cargo bench --bench min_max_bytes
```

All other benchmarks should remain unaffected since they either:
- Don't trigger sequential fast path (sparse, mixed patterns)
- Complete in < 3 batches (single batch benchmarks)
- Benefit equally from both paths (dense first batch)

---

## Alternative Approaches Considered

### Alternative 1: Remove Sequential Fast Path

**Idea:** Remove the sequential detection entirely and rely on DenseInline committed mode.

**Pros:**
- Simpler code
- No duplication of logic

**Cons:**
- ❌ Loses ~2-3% improvement in single-batch sequential workloads
- ❌ DenseInline has per-batch mark tracking overhead before commitment
- ❌ Sequential pattern is the most common aggregation case and deserves zero-overhead handling

**Decision:** Rejected. The sequential fast path is valuable for the common case.

---

### Alternative 2: Reuse Locations Vector

**Idea:** Make `locations` a reusable field in `MinMaxBytesState`, similar to scratch buffers.

**Pros:**
- Eliminates repeated allocations
- Simpler than committed mode

**Cons:**
- ❌ Still pays cost of zeroing/clearing the vector every batch
- ❌ Keeps large memory footprint (total_num_groups × size_of::<SequentialDenseLocation>())
- ❌ Cache inefficient: traverses large vector even when few groups update
- ❌ Doesn't match DenseInline committed performance

**Decision:** Rejected. The committed mode approach is superior.

---

### Alternative 3: Hybrid Approach with Lazy Locations

**Idea:** Start without locations vector, switch to it only if needed (e.g., multiple updates per group).

**Pros:**
- Could optimize both single-touch and multi-touch scenarios

**Cons:**
- ❌ Complex branching logic
- ❌ Unclear benefit: sequential pattern rarely has multi-touch per group
- ❌ Added complexity not justified by use case

**Decision:** Rejected. Keep it simple.

---

## Implementation Checklist

- [ ] **Task 1:** Add `sequential_stable_batches`, `sequential_committed`, `sequential_committed_groups` fields
- [ ] **Task 2:** Implement `update_batch_sequential_dense_committed` method
- [ ] **Task 3:** Add stability tracking in `update_batch_sequential_dense`
- [ ] **Task 4:** Update fast path routing in `update_batch`
- [ ] **Task 5:** Verify domain growth handling
- [ ] **Task 6:** Update `reset_after_full_emit` to clear sequential state
- [ ] **Task 7:** Add test `sequential_dense_commits_after_stable_batches`
- [ ] **Task 8:** Add test `sequential_dense_revokes_commitment_on_domain_growth`
- [ ] **Task 9:** Run benchmark and verify improvement
- [ ] **Task 10:** Run full benchmark suite and verify no new regressions

---

## Success Criteria

1. ✅ "min bytes dense reused accumulator" regression eliminated (< 1% overhead vs. baseline)
2. ✅ All other benchmarks remain within ±2% of current performance
3. ✅ Tests pass for committed mode and domain growth scenarios
4. ✅ Code maintains clarity and follows repository guidelines
5. ✅ Zero-allocation steady-state achieved for repeated sequential workloads

---

## References

- **PR Range:** `c1ac251d6^..b4c40ad32`
- **File:** `datafusion/functions-aggregate/src/min_max/min_max_bytes.rs`
- **Benchmark:** `datafusion/functions-aggregate/benches/min_max_bytes.rs`
- **Related Constants:**
  - `DENSE_INLINE_STABILITY_THRESHOLD = 3`
  - `BATCH_SIZE = 512`
  - `MONOTONIC_BATCHES = 32`

---

## Notes

- The sequential fast path is triggered by the most common aggregation pattern in practice
- The regression only manifests in multi-batch reused accumulator scenarios (≥4 batches)
- Single-batch workloads are unaffected and benefit from the fast path
- The fix preserves the benefits of the sequential optimization while adding DenseInline-style commitment
- This is a precision fix: surgical addition of stability tracking to an existing fast path
