# MinMaxBytesAccumulator Performance Regression Analysis

**Date:** October 9, 2025  
**Branch:** `hash-17897a`  
**PR Commit Range:** `3d0c357af^..af85f8a4a`  
**Status:** Root Cause Identified, Fix Planning Complete  
**Severity:** High (9 regressions up to +31%, 6 improvements up to -89%)

---

## Executive Summary

The PR attempted to fix a quadratic allocation issue in `MinMaxBytesAccumulator::update_batch` by introducing an adaptive mode system (DenseInline, SimpleTracked, SparseHashMap). While this successfully improved sparse workloads by up to 89%, it caused severe regressions (12-31%) in dense workloads that are more common in production.

**Root Cause:** The DenseInline mode still allocates and resizes `dense_scratch` proportional to `total_num_groups` (all historical groups) rather than just the groups touched in the current batch. Combined with epoch-based tracking overhead and lack of consecutive-group optimization, this makes dense patterns slower than the original naive implementation.

**Impact:** 9 benchmarks regressed (dense/medium workloads), 6 improved (sparse/growing workloads).

---

## Benchmark Results Summary

### Criterion Benchmark Summary (Statistically Significant Changes)

| Benchmark                                | Mean Change | P-value  | Category |
|------------------------------------------|-------------|----------|----------|
| **IMPROVEMENTS (Sparse/Growing)**        |             |          |          |
| min bytes ultra sparse                   |     -89.11% | 0.000000 | Sparse   |
| min bytes monotonic group ids            |     -43.45% | 0.000000 | Growing  |
| min bytes multi batch large              |     -43.21% | 0.000000 | Growing  |
| min bytes quadratic growing total groups |     -42.26% | 0.000000 | Growing  |
| min bytes growing total groups           |     -24.54% | 0.000000 | Growing  |
| min bytes mode transition                |     -12.74% | 0.000000 | Mixed    |
| **REGRESSIONS (Dense/Medium)**           |             |          |          |
| min bytes dense duplicate groups         |     +31.36% | 0.000000 | Dense    |
| min bytes dense reused accumulator       |     +26.12% | 0.000000 | Dense    |
| min bytes medium cardinality stable      |     +17.09% | 0.000000 | Medium   |
| min bytes single batch large             |     +16.79% | 0.000000 | Dense    |
| min bytes large dense groups             |     +14.72% | 0.000000 | Dense    |
| min bytes single batch small             |     +13.91% | 0.000000 | Dense    |
| min bytes sparse groups                  |     +13.23% | 0.000000 | Medium   |
| min bytes dense first batch              |     +12.94% | 0.000000 | Dense    |
| min bytes sequential stable groups       |      +3.57% | 0.000000 | Dense    |

**Summary:** 6 improvements, 9 regressions (p < 0.05)

### Workload Categorization

- **Dense Workloads (7 regressions):** High density (>50%), moderate cardinality (<20K groups), sequential/duplicate access patterns
- **Medium Workloads (2 regressions):** Medium density (20-50%), moderate-high cardinality (50K groups), mixed access
- **Sparse Workloads (1 improvement):** Low density (<10%), ultra-high cardinality (>100K groups)
- **Growing Workloads (5 improvements):** Monotonic or rapidly growing `total_num_groups` across batches

---

## Root Cause Analysis

### The Original Problem (Pre-PR)

The issue described was that `update_batch` allocated a `locations` buffer sized to `total_num_groups` for every batch:

```rust
fn update_batch(..., total_num_groups: usize) {
    // ❌ PROBLEM: Allocates vector sized to ALL historical groups
    let mut locations = vec![MinMaxLocation::ExistingMinMax; total_num_groups];
    
    for (row_idx, &group_index) in group_indices.iter().enumerate() {
        // Update locations[group_index] based on comparison
        locations[group_index] = MinMaxLocation::Input(new_val);
    }
    
    // ❌ PROBLEM: Iterates ALL groups (including untouched ones)
    for (group_index, location) in locations.iter().enumerate() {
        match location {
            MinMaxLocation::ExistingMinMax => {}
            MinMaxLocation::Input(new_val) => self.set_value(group_index, new_val),
        }
    }
}
```

**Issues:**
- Allocates O(total_num_groups) memory per batch
- Initializes O(total_num_groups) entries per batch  
- Iterates O(total_num_groups) entries at the end
- As `total_num_groups` grows, later batches become progressively slower (quadratic behavior)

### The Attempted Fix (Current Implementation)

The PR introduced an adaptive system with three modes:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkloadMode {
    DenseInline,      // For high-density access patterns
    SimpleTracked,    // For medium-density patterns (not yet implemented)
    SparseHashMap,    // For ultra-sparse patterns (not yet implemented)
    Undecided,        // Learning phase
}
```

#### DenseInline Implementation (Lines 556-650)

```rust
fn update_batch_dense_inline(..., total_num_groups: usize) {
    if self.min_max.len() < total_num_groups {
        self.min_max.resize(total_num_groups, None);
    }

    // ❌ STILL PROBLEMATIC: Resizes to ALL historical groups
    if self.dense_scratch.len() < total_num_groups {
        self.dense_scratch.resize(
            total_num_groups,
            DenseScratchSlot {
                epoch: 0,
                location: DenseLocation::Untouched,
            },
        );
    }

    self.dense_epoch = self.dense_epoch.wrapping_add(1);
    self.dense_touched_groups.clear();

    // ✅ GOOD: Only process rows in current batch
    for (row_idx, &group_index) in group_indices.iter().enumerate() {
        let Some(new_val) = batch_values[row_idx] else {
            continue;
        };

        let slot = &mut self.dense_scratch[group_index];

        // ❌ OVERHEAD: Epoch checking on every access
        if slot.epoch != self.dense_epoch {
            slot.epoch = self.dense_epoch;
            self.dense_touched_groups.push(group_index);
            // ... determine location ...
        } else {
            // ❌ OVERHEAD: Nested match on location enum
            match slot.location {
                DenseLocation::Untouched => { /* ... */ }
                DenseLocation::Existing => { /* ... */ }
                DenseLocation::BatchIndex(prev_idx) => { /* ... */ }
            }
        }
    }

    // ✅ GOOD: Only iterate touched groups
    for &group_index in &touched_groups {
        let location = self.dense_scratch[group_index].location;
        if let DenseLocation::BatchIndex(idx) = location {
            let value = batch_values[idx].expect("...");
            self.set_value(group_index, value);
        }
    }
}
```

### Why Dense Workloads Regressed

The DenseInline implementation has multiple performance issues:

#### 1. **Still O(total_num_groups) Allocation** (Primary Issue)

```rust
if self.dense_scratch.len() < total_num_groups {
    self.dense_scratch.resize(total_num_groups, /* ... */);
}
```

- Every time `total_num_groups` increases, resize allocates and initializes memory for ALL groups
- For growing workloads (common in streaming), this happens repeatedly
- Example: If groups grow from 1K → 10K over 10 batches, we allocate 1K + 2K + 3K + ... + 10K = 55K slots total

#### 2. **Epoch-Based Tracking Overhead**

Each group access requires:
- Check: `if slot.epoch != self.dense_epoch` (branch + memory load)
- Update: `slot.epoch = self.dense_epoch` (memory store)
- Bookkeeping: `self.dense_touched_groups.push(group_index)` (vector append)

For dense workloads where most groups are touched, this adds ~3-4 operations per group with no benefit.

#### 3. **Slot Indirection Overhead**

```rust
struct DenseScratchSlot {
    epoch: u64,           // 8 bytes
    location: DenseLocation,  // 16+ bytes (enum with usize variant)
}
```

- 24+ bytes per slot vs. 8 bytes for a simple `Option<usize>` pointer
- Extra cache line misses due to larger memory footprint
- Nested enum matching adds branch mispredictions

#### 4. **No Optimization for Consecutive Duplicates**

The "dense duplicate groups" benchmark (+31% regression) exposes this:

```rust
// Benchmark pattern: [0, 0, 1, 1, 2, 2, 3, 3, ...]
for (row_idx, &group_index) in group_indices.iter().enumerate() {
    let slot = &mut self.dense_scratch[group_index];  // ❌ Repeated lookup
    if slot.epoch != self.dense_epoch {
        // First occurrence: expensive path
    } else {
        // Second occurrence: still does full slot access + enum match
    }
}
```

A simple optimization would cache `last_group_index` and skip the lookup entirely for consecutive duplicates.

### Why Sparse Workloads Improved

For truly sparse patterns (e.g., "ultra sparse": 100 groups out of 1M total):

**Before:**
```rust
let mut locations = vec![ExistingMinMax; 1_000_000];  // ❌ Allocate 1M entries
// Touch 100 groups
for group in locations.iter() {  // ❌ Iterate 1M entries
    // ...
}
```

**After:**
```rust
self.dense_scratch.resize(1_000_000, ...);  // ❌ Still allocate 1M (bad)
self.dense_touched_groups = Vec::with_capacity(100);  // ✅ Only track 100
for &group in &self.dense_touched_groups {  // ✅ Only iterate 100
    // ...
}
```

The improvement comes from iterating only touched groups at the end (100 vs. 1M iterations), which dominates the per-access overhead. However, the allocation cost is still present—it's just amortized better in multi-batch scenarios.

---

## Detailed Performance Bottlenecks

### 1. Memory Allocation Patterns

| Implementation | Per-Batch Allocation | Cost for Growing Groups |
|----------------|---------------------|-------------------------|
| Original | `Vec<Location>` of size `total_num_groups` | O(N) per batch |
| Current DenseInline | `Vec<DenseScratchSlot>` resize to `total_num_groups` | O(N) on growth |
| Optimal Dense | `Vec<Slot>` of size `max_touched_group + 1` | O(U) per batch (U = unique groups) |

**Current cost:** For a workload growing from 1K to 10K groups over 10 batches:
- Original: 10 × (1K + 2K + ... + 10K) / 2 ≈ 55K allocations
- Current: Same (resize behavior is identical)
- Optimal: 10K allocations (once, at max size)

### 2. Access Pattern Analysis

**Dense Duplicate Groups** (worst regression: +31%):
```
Pattern: [0, 0, 1, 1, 2, 2, ...]
Access: 512 rows, 256 unique groups, 32 batches
```

Current implementation:
- First `0`: Epoch check ✗, update epoch, push to touched (3 ops)
- Second `0`: Epoch check ✓, enum match on location (2 ops)
- **Total: 512 × 2.5 ops = 1,280 operations**

Optimal implementation:
- First `0`: Process normally (3 ops)
- Second `0`: Check `group == last_group`, skip entirely (1 op)
- **Total: 256 × 3 + 256 × 1 = 1,024 operations (20% reduction)**

### 3. Cache Behavior

**DenseScratchSlot layout:**
```rust
struct DenseScratchSlot {
    epoch: u64,                    // 8 bytes
    location: DenseLocation,       // 24 bytes (enum)
}
// Total: 32 bytes per slot
```

For 10K groups: 320 KB of scratch space (fits in L2/L3 cache, but wastes bandwidth)

**Optimal layout:**
```rust
struct CompactSlot {
    epoch_and_flags: u32,    // Upper 31 bits = epoch, low bit = has_value
    batch_index: u32,        // Index into batch, or u32::MAX for "existing"
}
// Total: 8 bytes per slot
```

For 10K groups: 80 KB (4× smaller, better cache utilization)

---

## Adaptive Mode System Analysis

### Current Thresholds

```rust
const LEARNING_WINDOW: usize = 3;
const DENSE_THRESHOLD_PPM: u32 = 550_000;   // 55% density
const SIMPLE_THRESHOLD_PPM: u32 = 120_000;  // 12% density
const MAX_GROWTH_EVENTS: u32 = 2;
```

### Issues with Current Heuristics

1. **Learning Window Too Short:**
   - 3 batches is insufficient to determine true workload characteristics
   - SQL queries often have warm-up phases (initial batches may be atypical)
   - Variance in batch density isn't captured

2. **Threshold Positioning:**
   - DENSE_THRESHOLD_PPM = 550k (55%) is too low for "dense"
     - Dense workloads typically have >70% density
     - 55% is medium-density, which gets DenseInline overhead without benefits
   - SIMPLE_THRESHOLD_PPM = 120k (12%) is too high for "sparse"
     - True sparse workloads are <5% density
     - 12-55% range needs a proper hybrid implementation

3. **Growth Detection Too Aggressive:**
   - MAX_GROWTH_EVENTS = 2 means after 2 batch resizes, reset learning
   - Natural growth patterns (streaming data) trigger constant resets
   - Prevents mode convergence

4. **No Hysteresis:**
   - Switching modes on density boundary (55% → 54% → 55%) causes thrashing
   - Need sustained pattern change before mode transition

### Proposed Thresholds

```rust
const LEARNING_WINDOW: usize = 8;              // Capture true patterns
const DENSE_THRESHOLD_PPM: u32 = 700_000;      // 70% density (truly dense)
const SIMPLE_THRESHOLD_PPM: u32 = 200_000;     // 20% density (medium)
const SPARSE_THRESHOLD_PPM: u32 = 100_000;     // 10% density (truly sparse)
const MAX_GROWTH_EVENTS: u32 = 5;              // Allow more natural growth
const MODE_CONFIDENCE_THRESHOLD: f64 = 0.80;   // 80% of batches must agree
const HYSTERESIS_MULTIPLIER: f64 = 2.0;        // Require 2× density change
```

---

## Proposed Fix Strategy

### Phase 1: Fix DenseInline Mode (Eliminate Regressions)

#### Task 1: Decouple Scratch Size from total_num_groups

**Current problematic code:**
```rust
if self.dense_scratch.len() < total_num_groups {
    self.dense_scratch.resize(total_num_groups, ...);
}
```

**Proposed fix:**
```rust
// Only resize based on actual max group index in current batch
let max_group_in_batch = group_indices.iter().max().copied().unwrap_or(0);
if self.dense_scratch.len() <= max_group_in_batch {
    let new_size = (max_group_in_batch + 1).next_power_of_two();
    self.dense_scratch.resize(new_size, ...);
}
```

**Impact:** Eliminates quadratic growth behavior entirely. Memory usage scales with actual groups touched, not historical maximum.

#### Task 2: Add Consecutive Group Caching

**Proposed optimization:**
```rust
let mut last_group_index = usize::MAX;
let mut last_location = DenseLocation::Untouched;

for (row_idx, &group_index) in group_indices.iter().enumerate() {
    let Some(new_val) = batch_values[row_idx] else {
        continue;
    };

    // ✅ Fast path for consecutive duplicates
    if group_index == last_group_index {
        match last_location {
            DenseLocation::BatchIndex(prev_idx) => {
                if cmp(new_val, batch_values[prev_idx].unwrap()) {
                    last_location = DenseLocation::BatchIndex(row_idx);
                }
            }
            DenseLocation::Existing => {
                // Compare against existing, potentially update last_location
            }
            _ => {}
        }
        continue;  // Skip slot lookup entirely
    }

    // Normal path: full slot access
    let slot = &mut self.dense_scratch[group_index];
    // ... existing logic ...
    
    last_group_index = group_index;
    last_location = slot.location;
}
```

**Impact:** Estimated 20-30% improvement on "dense duplicate groups" benchmark.

#### Task 3: Optimize Memory Layout

**Proposed compact slot:**
```rust
#[derive(Debug, Clone, Copy)]
struct CompactDenseSlot {
    // Pack epoch (31 bits) + flag (1 bit) into u32
    epoch_and_flag: u32,
    // Use u32 for batch index (4B rows should be enough)
    // u32::MAX means "use existing value"
    batch_index: u32,
}

impl CompactDenseSlot {
    fn epoch(&self) -> u32 {
        self.epoch_and_flag >> 1
    }
    
    fn set_epoch(&mut self, epoch: u32) {
        self.epoch_and_flag = (epoch << 1) | (self.epoch_and_flag & 1);
    }
    
    fn has_batch_value(&self) -> bool {
        (self.epoch_and_flag & 1) != 0
    }
}
```

**Impact:** 4× memory reduction (32 bytes → 8 bytes per slot), better cache utilization.

### Phase 2: Implement SimpleTracked Mode (Medium Density)

**Design:** Bounded scratch buffer with overflow tracking

```rust
const SIMPLE_SCRATCH_SIZE: usize = 8_192;  // 64KB with CompactDenseSlot

struct SimpleState {
    // Fixed-size scratch using modulo addressing
    scratch: [CompactDenseSlot; SIMPLE_SCRATCH_SIZE],
    // Track collisions (when group_index % SIZE maps to occupied slot)
    overflow: HashSet<usize>,
    // Track which slots were touched (for apply phase)
    touched_slots: Vec<usize>,
}

fn update_batch_simple_tracked(...) {
    for (row_idx, &group_index) in group_indices.iter().enumerate() {
        let slot_index = group_index % SIMPLE_SCRATCH_SIZE;
        let slot = &mut self.simple_scratch[slot_index];
        
        if slot.epoch() != current_epoch {
            // First access to this slot this batch
            slot.set_epoch(current_epoch);
            self.simple_touched_slots.push(slot_index);
            // ... determine if this is the right group or collision ...
        } else {
            // Slot already accessed - check if same group or collision
            // ... handle collision by adding to overflow HashSet ...
        }
    }
    
    // Apply updates from slots + overflow
    for &slot_idx in &self.simple_touched_slots {
        // ...
    }
    for &group_idx in &self.simple_overflow {
        // ...
    }
}
```

**Impact:** Bounded O(1) overhead for medium-density workloads, graceful degradation to HashMap for true sparse patterns.

### Phase 3: Implement Monotonic Fast Path

**Detection:**
```rust
fn is_batch_monotonic(group_indices: &[usize]) -> bool {
    group_indices.windows(2).all(|w| w[0] <= w[1])
}
```

**Fast path:**
```rust
if is_batch_monotonic(group_indices) {
    // Direct append mode - no tracking needed
    let mut prev_group = usize::MAX;
    for (row_idx, &group_index) in group_indices.iter().enumerate() {
        if group_index != prev_group {
            // New group - just process directly
            if let Some(new_val) = batch_values[row_idx] {
                self.update_or_insert(group_index, new_val, cmp);
            }
            prev_group = group_index;
        } else {
            // Duplicate group - compare with previous in batch
            // ... simplified logic ...
        }
    }
    return Ok(());
}
```

**Impact:** Eliminates all tracking overhead for monotonic patterns (common in time-series, sorted inputs).

### Phase 4: Improve Adaptive Logic

**Enhanced learning:**
```rust
struct AdaptiveState {
    // Increase window size
    density_history: [u32; 8],
    unique_history: [usize; 8],
    total_history: [usize; 8],
    
    // Add variance tracking
    density_variance: f64,
    
    // Add mode confidence
    mode_votes: [u32; 4],  // Votes for each mode
    
    // Add hysteresis
    last_mode_switch_batch: u32,
    batches_since_switch: u32,
}

fn should_switch_mode(&self, current_mode: WorkloadMode, suggested_mode: WorkloadMode) -> bool {
    // Require sustained pattern change
    if self.batches_since_switch < 5 {
        return false;
    }
    
    // Require confidence threshold
    let confidence = self.mode_votes[suggested_mode as usize] as f64 / 
                     self.batches_seen as f64;
    if confidence < MODE_CONFIDENCE_THRESHOLD {
        return false;
    }
    
    // Require significant density change (hysteresis)
    let current_density = self.average_density_ppm();
    let threshold = match (current_mode, suggested_mode) {
        (WorkloadMode::DenseInline, WorkloadMode::SimpleTracked) => {
            current_density < (DENSE_THRESHOLD_PPM / 2)  // Must drop below 35%
        }
        (WorkloadMode::SimpleTracked, WorkloadMode::DenseInline) => {
            current_density > (DENSE_THRESHOLD_PPM * 3 / 2)  // Must rise above 105%
        }
        _ => true,
    };
    
    threshold
}
```

---

## Implementation Task List

### Critical Path (Fix Regressions)

1. **Task 1: Decouple DenseInline scratch sizing from total_num_groups** [Priority: P0]
   - Resize based on `max(group_indices)` instead of `total_num_groups`
   - Use `next_power_of_two()` for allocation strategy
   - Track high-water mark to avoid shrinking
   - **Expected impact:** Eliminate quadratic behavior, fix growing workloads

2. **Task 2: Add consecutive group caching** [Priority: P0]
   - Track `last_group_index` and `last_location` in loop
   - Skip slot access when `group_index == last_group_index`
   - **Expected impact:** +20-30% on "dense duplicate groups" benchmark

3. **Task 3: Optimize touched_groups vector** [Priority: P1]
   - Pre-size based on adaptive statistics
   - Use `clear()` instead of `take()` to preserve capacity
   - Consider SmallVec for common small cases
   - **Expected impact:** Reduce allocation churn by ~50%

4. **Task 4: Compact DenseScratchSlot layout** [Priority: P1]
   - Pack epoch + flag into u32
   - Use u32 for batch_index
   - Reduce from 32 bytes to 8 bytes per slot
   - **Expected impact:** 4× memory reduction, better cache utilization

### Optimization Path (Improve Sparse)

5. **Task 5: Implement SimpleTracked mode** [Priority: P2]
   - Fixed-size scratch buffer (8K slots)
   - Modulo addressing with collision detection
   - Overflow HashSet for collisions
   - **Expected impact:** Better medium-density performance

6. **Task 6: Add monotonic fast path** [Priority: P2]
   - Detect monotonic group sequences
   - Bypass all tracking for monotonic batches
   - **Expected impact:** Further improve growing/monotonic workloads

### Adaptive Improvements

7. **Task 7: Tune adaptive thresholds** [Priority: P2]
   - Increase LEARNING_WINDOW to 8
   - Adjust density thresholds (70%/20%/10%)
   - Increase MAX_GROWTH_EVENTS to 5
   - **Expected impact:** Better mode selection, less thrashing

8. **Task 8: Add mode transition hysteresis** [Priority: P3]
   - Require sustained pattern change (5+ batches)
   - Implement confidence scoring
   - Add variance detection
   - **Expected impact:** Stable mode selection

### Validation

9. **Task 9: Benchmark each optimization** [Priority: P0]
   - Create feature branches for each task
   - Run full benchmark suite per branch
   - Measure individual ROI
   - **Success criteria:** No regressions introduced

10. **Task 10: Integration testing** [Priority: P0]
    - Combine validated optimizations
    - Ensure mode transitions preserve correctness
    - Test edge cases (empty batches, null values, etc.)
    - **Success criteria:** All tests pass

11. **Task 11: Final validation** [Priority: P0]
    - All dense benchmarks: ≤5% regression vs. baseline
    - Sparse benchmarks: ≥80% of current improvements retained
    - Extended benchmarks: 1M+ groups, 10K+ batches
    - **Success criteria:** p < 0.05 for all changes

---

## Expected Outcome

### Target Performance Profile

| Benchmark Category | Current Status | Target Status | Primary Fix |
|-------------------|----------------|---------------|-------------|
| Dense Single Batch | +12-16% slower | ±2% (neutral) | Tasks 1, 2, 4 |
| Dense Multi-Batch | +26-31% slower | -5% (improved) | Tasks 1, 2, 3 |
| Medium Cardinality | +13-17% slower | ±5% (neutral) | Tasks 5, 7 |
| Growing Groups | -24% to -43% (good) | -30% to -50% (better) | Tasks 1, 6 |
| Ultra Sparse | -89% (excellent) | -85% (still excellent) | Task 5 |

### Memory Usage

| Scenario | Current | Target | Reduction |
|----------|---------|--------|-----------|
| 10K dense groups | 320 KB scratch | 80 KB scratch | 4× |
| 100K sparse groups | 3.2 MB scratch | 64 KB scratch + HashMap | ~50× |
| Growing 1K→10K | Allocate 55K slots | Allocate 16K slots | 3.4× |

---

## Risk Assessment

### Low Risk Changes
- ✅ Task 1 (resize logic): Simple, well-tested pattern
- ✅ Task 2 (consecutive caching): Additive optimization, easy to verify
- ✅ Task 3 (vector reuse): Standard performance pattern

### Medium Risk Changes
- ⚠️ Task 4 (compact layout): Bit manipulation, needs careful testing
- ⚠️ Task 5 (SimpleTracked): New code path, requires extensive testing
- ⚠️ Task 7 (thresholds): May need tuning on real workloads

### High Risk Changes
- 🔴 Task 8 (hysteresis): Complex state machine, potential for mode thrashing bugs

### Mitigation Strategy
1. Implement low-risk changes first (Tasks 1-3)
2. Validate each change independently
3. Use feature flags for medium/high-risk changes
4. Extensive benchmark coverage before merge
5. Consider gradual rollout with monitoring

---

## Conclusion

The current implementation correctly identified the need for adaptive strategies but failed in execution by:
1. Not actually eliminating the O(total_num_groups) allocation in DenseInline
2. Adding overhead (epoch tracking, slot indirection) without benefits for dense patterns
3. Choosing overly aggressive thresholds that select DenseInline for medium-density workloads

The fix requires:
1. **Critical:** Fix DenseInline to size based on touched groups, not total groups
2. **Critical:** Add consecutive group caching to handle duplicate patterns
3. **Important:** Implement proper medium-density mode (SimpleTracked)
4. **Nice-to-have:** Improve adaptive heuristics with hysteresis

With these changes, we expect to eliminate all dense regressions while retaining 80%+ of sparse improvements, achieving the original goal of fixing quadratic behavior without sacrificing common-case performance.

---

## References

- **Source File:** `datafusion/functions-aggregate/src/min_max/min_max_bytes.rs`
- **Benchmark File:** `datafusion/functions-aggregate/benches/min_max_bytes.rs`
- **Original Issue:** Quadratic allocation in `MinMaxBytesAccumulator::update_batch`
- **Related Files:**
  - `datafusion/functions-aggregate-common/src/aggregate/groups_accumulator/nulls.rs`
  - `datafusion/expr/src/groups_accumulator.rs`
