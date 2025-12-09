# Root Cause Analysis: Multi-Partition Aggregate Repartitioning Failure

**Date:** December 9, 2025  
**Branch:** optimizer-repartition-18989a  
**Issue:** EnforceDistribution fails to insert necessary RepartitionExec between aggregates

---

## Executive Summary

The fix for the multi-partition aggregate repartitioning bug is **incomplete**. While the original integration test passes, the actual production scenario (demonstrated by `multi_partition_aggregate.rs` example) still fails with the same sanity check error.

**Root Cause:** The physical plan structure changes AFTER the `EnforceDistribution` optimizer rule runs, violating distribution requirements that were previously satisfied or deemed unnecessary.

---

## Error Message

```
Failed to execute query: Context("SanityCheckPlan", Plan("Plan: [
  "AggregateExec: mode=SinglePartitioned, gby=[ts@0 as ts], aggr=[count(count(metrics.value))]", 
  "  ProjectionExec: expr=[ts@1 as ts, count(metrics.value)@2 as count(metrics.value)]", 
  "    AggregateExec: mode=FinalPartitioned, gby=[region@0 as region, ts@1 as ts], aggr=[count(metrics.value)]", 
  "      CoalesceBatchesExec: target_batch_size=8192", 
  "        RepartitionExec: partitioning=Hash([region@0, ts@1], 10), input_partitions=2", 
  "          AggregateExec: mode=Partial, gby=[region@1 as region, ts@0 as ts], aggr=[count(metrics.value)]", 
  "            DataSourceExec: partitions=2, partition_sizes=[0, 0]"
] does not satisfy distribution requirements: HashPartitioned[[ts@0]]). 
Child-0 output partitioning: Hash([region@0, ts@0], 10)"))
```

**Translation:** The second aggregate (SinglePartitioned) requires `Hash([ts@0])` but receives `Hash([region@0, ts@0], 10)` from the ProjectionExec.

---

## Investigation Findings

### 1. Debug Output from EnforceDistribution

When the `EnforceDistribution` rule runs, it sees:

```
DEBUG: Checking hash requirement
  Plan: AggregateExec
  Child: AggregateExec
  Child partitioning: UnknownPartitioning(2)
  Requirement: HashPartitioned([region, ts])
  Satisfies: false
  Multi-partitions: true
  → Will insert repartition ✓

DEBUG: Checking hash requirement
  Plan: AggregateExec  
  Child: AggregateExec
  Child partitioning: UnknownPartitioning(1)
  Requirement: HashPartitioned([ts])
  Satisfies: true  ← Single partition satisfies any hash requirement
  Multi-partitions: false
  → Will NOT insert repartition ✗
```

**Key Observation:** When processing the second aggregate:
- Child is reported as `AggregateExec` (not `ProjectionExec`)
- Child partitioning is `UnknownPartitioning(1)` (not `Hash([region, ts], 10)`)
- Since `partition_count() == 1`, `satisfy()` returns `true` automatically
- No repartition is inserted

### 2. Final Plan vs. What EnforceDistribution Saw

| Phase | Second Aggregate's Child | Partitioning |
|-------|-------------------------|--------------|
| **During EnforceDistribution** | AggregateExec | UnknownPartitioning(1) |
| **Final Plan (Error)** | ProjectionExec | Hash([region@0, ts@0], 10) |

**This mismatch proves the plan structure changes AFTER EnforceDistribution runs.**

---

## Why the Plan Changes

### Hypothesis 1: Sort Operators Removed During Pruning

The logical plan includes Sort operators:
```
Sort: metrics.ts
  Aggregate: groupBy=[ts]
    Sort: metrics.region, metrics.ts  ← This gets removed
      Aggregate: groupBy=[region, ts]
```

During the "pruning" phase of `EnforceDistribution`, unnecessary distribution-changing operators (including sorts) are removed. This causes the plan structure to simplify, potentially causing the optimizer to not see the final structure.

**Status:** Partially explains it, but doesn't explain the ProjectionExec

### Hypothesis 2: Projection Inserted by Later Optimizer Rule

Looking at the optimizer pipeline:
```rust
Arc::new(EnforceDistribution::new()),
Arc::new(CombinePartialFinalAggregate::new()),  ← Runs AFTER
Arc::new(EnforceSorting::new()),                ← Runs AFTER
Arc::new(ProjectionPushdown::new()),             ← Runs AFTER
```

The `ProjectionPushdown` or `CombinePartialFinalAggregate` rules might insert/modify projections AFTER distribution enforcement.

**Status:** Needs verification - search codebase for where ProjectionExec gets inserted between aggregates

### Hypothesis 3: Aggregate Physical Planning Inserts Projection

When creating physical aggregates, the planner might insert a projection to align schemas between Partial/Final aggregate modes. This would happen BEFORE optimizers run, but the projection's partitioning might be computed incorrectly.

**Status:** Most likely - need to check aggregate physical planning code

---

## Why satisfy() Returns True Incorrectly

From `datafusion/physical-expr/src/partitioning.rs:153-220`:

```rust
pub fn satisfy(&self, required: &Distribution, ...) -> bool {
    match required {
        Distribution::UnspecifiedDistribution => true,
        Distribution::SinglePartition if self.partition_count() == 1 => true,
        // ⚠️ ANY partitioning with count=1 satisfies hash requirements
        Distribution::HashPartitioned(_) if self.partition_count() == 1 => true,
        Distribution::HashPartitioned(required_exprs) => {
            match self {
                Partitioning::Hash(partition_exprs, _) => {
                    physical_exprs_equal(required_exprs, partition_exprs)
                    // Falls back to equivalence class normalization
                }
                _ => false,
            }
        }
        _ => false,
    }
}
```

**The Bug:** When `partition_count() == 1`, ANY partitioning (including `UnknownPartitioning(1)`) automatically satisfies hash requirements. This makes sense semantically (single partition = all data together), but it causes the optimizer to skip repartitioning even when the plan will later be modified to have multiple partitions.

---

## Attempted Fixes and Why They Failed

### Fix 1: Add aggregate-specific re-enforcement
**Code:** Lines 1307-1321 in `enforce_distribution.rs`  
**Status:** Removed (didn't help)  
**Why it failed:** Runs before the plan structure is finalized, so still sees `UnknownPartitioning(1)`

### Fix 2: Check satisfy() before marking hash_necessary
**Code:** Lines 1150-1156 in `enforce_distribution.rs`  
**Status:** Currently in code  
**Why it fails:** The satisfy check happens too early, before the final plan structure exists

### Fix 3: Update alignment logic to respect satisfy() checks
**Code:** Lines 1169-1184 in `enforce_distribution.rs`  
**Status:** Currently in code  
**Why it fails:** Same timing issue - the plan changes after this check

---

## The Real Problem: Timing and Plan Stability

The fundamental issue is an **architectural assumption violation**:

```rust
// From optimizer.rs:107-108
// The EnforceDistribution rule is for adding essential repartitioning to satisfy distribution
// requirements. Please make sure that the whole plan tree is determined before this rule.
```

**This assumption is FALSE.** The plan tree is NOT fully determined when EnforceDistribution runs. Evidence:

1. Sorts are removed during the pruning phase within EnforceDistribution itself
2. Projections appear/disappear between when EnforceDistribution checks and when SanityCheckPlan validates
3. Partitioning metadata changes (from `UnknownPartitioning(1)` to `Hash([...], 10)`)

---

## Proposed Solutions

### Solution A: Two-Pass Distribution Enforcement (Conservative)

**Approach:** Run EnforceDistribution twice:
1. First pass: After initial physical planning
2. Second pass: After all plan-modifying optimizers complete

**Pros:**
- Catches any distribution violations introduced by intermediate optimizers
- Minimal code change
- Safe and conservative

**Cons:**
- Performance overhead (running optimizer twice)
- Still doesn't fix root cause

**Implementation:**
```rust
// In optimizer.rs, add second pass:
Arc::new(EnforceDistribution::new()),
Arc::new(CombinePartialFinalAggregate::new()),
Arc::new(EnforceSorting::new()),
Arc::new(ProjectionPushdown::new()),
Arc::new(CoalesceBatches::new()),
Arc::new(EnforceDistribution::new()),  // ← Second pass
Arc::new(OutputRequirements::new_remove_mode()),
```

### Solution B: Make Projection-Aware (Targeted)

**Approach:** Detect when a Projection drops partitioning columns and force repartitioning

**Implementation:**
```rust
// In ProjectionExec::compute_properties():
pub fn project(&self, mapping: &ProjectionMapping, ...) -> Self {
    if let Partitioning::Hash(exprs, part) = self {
        let normalized_exprs = ...;
        
        // Check if any partitioning expression became UnknownColumn
        let has_unknown = normalized_exprs.iter().any(|e| {
            e.as_any().is::<UnknownColumn>()
        });
        
        if has_unknown {
            // Partitioning is no longer valid - drop to Unknown
            return Partitioning::UnknownPartitioning(*part);
        }
        
        Partitioning::Hash(normalized_exprs, *part)
    } else {
        self.clone()
    }
}
```

**Pros:**
- Fixes the specific case (projection dropping columns)
- Localized change
- Makes partitioning metadata more accurate

**Cons:**
- Only fixes projection-related issues
- Doesn't address general plan instability

### Solution C: Defer Projection Insertion (Structural)

**Approach:** Ensure ProjectionExec is inserted BEFORE EnforceDistribution runs, not after

**Requires:** Finding where projections are inserted and reordering optimizer pipeline

**Pros:**
- Fixes the root cause
- Plan is stable when EnforceDistribution runs

**Cons:**
- Requires understanding complex aggregate planning logic
- May have unintended side effects
- Higher risk

### Solution D: Smarter SanityCheckPlan (Pragmatic)

**Approach:** Make SanityCheckPlan automatically insert missing repartitions instead of just failing

**Implementation:**
```rust
// In sanity_checker.rs:
if !satisfies_distribution {
    // Instead of returning error, insert repartition
    let fixed_child = add_repartition_to_satisfy(child, requirement)?;
    return plan.with_new_children(vec![fixed_child]);
}
```

**Pros:**
- Catches ALL distribution violations, regardless of cause
- Defensive programming - fail-safe instead of fail-hard
- No timing issues

**Cons:**
- Hides bugs in earlier optimizer passes
- May insert suboptimal repartitions
- Philosophical: should a "sanity checker" modify the plan?

---

## Recommended Action Plan

### Phase 1: Quick Fix (Solution B + Logging)
1. Implement projection-aware partitioning (Solution B)
2. Add warning logs when projections drop partitioning columns
3. Test with `multi_partition_aggregate.rs` example
4. **Timeline:** 1-2 days

### Phase 2: Investigation (Parallel to Phase 1)
1. Trace exactly where ProjectionExec is inserted in aggregate planning
2. Identify all optimizer rules that modify plan structure after EnforceDistribution
3. Document findings
4. **Timeline:** 2-3 days

### Phase 3: Comprehensive Fix (Based on Phase 2 findings)
Choose one of:
- **If projections are the only issue:** Keep Solution B, add regression tests
- **If multiple optimizers modify plans:** Implement Solution A (two-pass)
- **If structural issue is deeper:** Implement Solution C (reorder pipeline)
- **If problems persist:** Implement Solution D as last resort
- **Timeline:** 3-5 days

---

## Test Cases Needed

### Immediate
1. ✅ `repartitions_between_sorted_aggregates` - Currently passes
2. ❌ `multi_partition_aggregate.rs` example - Currently fails
3. ❌ Projection between aggregates with different group keys
4. ❌ Multiple projections in aggregate pipeline
5. ❌ Aggregate after join with projection

### Comprehensive
6. Window functions with projections
7. Unions with different partitioning
8. CTEs with aggregates and projections
9. Nested aggregates (3+ levels)
10. Aggregates with complex expressions in group-by

---

## Code Locations

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| Main optimizer loop | `physical-optimizer/src/enforce_distribution.rs` | 1196-1207 | `ensure_distribution()` entry point |
| Pruning phase | `physical-optimizer/src/enforce_distribution.rs` | 1230-1236 | `prune_distribution_changing_nodes()` |
| Enforcement phase | `physical-optimizer/src/enforce_distribution.rs` | 1240-1495 | `enforce_required_repartitions()` |
| Hash necessary check | `physical-optimizer/src/enforce_distribution.rs` | 1150-1166 | Where `satisfy()` is called |
| Alignment logic | `physical-optimizer/src/enforce_distribution.rs` | 1169-1184 | Overrides hash_necessary flags |
| Partitioning satisfy | `physical-expr/src/partitioning.rs` | 153-195 | `satisfy()` method |
| Projection partitioning | `physical-expr/src/partitioning.rs` | 198-220 | `project()` method |
| Sanity checker | `physical-optimizer/src/sanity_checker.rs` | 140-170 | Where failure occurs |

---

## Open Questions

1. **Where exactly is the ProjectionExec inserted?**
   - Physical planning during aggregate creation?
   - `CombinePartialFinalAggregate` optimizer?
   - `ProjectionPushdown` optimizer?
   - Some other location?

2. **Why does partitioning change from `UnknownPartitioning(1)` to `Hash([...], 10)`?**
   - Is this the projection computing its output partitioning?
   - Is another optimizer updating partitioning metadata?
   - Is the aggregate mode changing?

3. **Why do the integration tests pass but the example fails?**
   - Different query patterns?
   - Different configuration?
   - Different aggregate modes (Partial/Final vs SinglePartitioned)?

4. **Should single-partition inputs really satisfy hash requirements?**
   - Current behavior in `satisfy()` assumes yes
   - But if the plan will later have multiple partitions, this is wrong
   - Is there a way to detect "will become multi-partition later"?

---

## References

- Original issue reproduction: `datafusion-examples/examples/multi_partition_aggregate.rs`
- Integration tests: `datafusion/core/tests/physical_optimizer/repartition_for_aggregates.rs`
- Optimizer pipeline: `datafusion/physical-optimizer/src/optimizer.rs`
- Distribution enforcement: `datafusion/physical-optimizer/src/enforce_distribution.rs`
- Partitioning logic: `datafusion/physical-expr/src/partitioning.rs`

---

## Next Steps

1. **Immediate:** Implement Solution B (projection-aware partitioning)
2. **Short-term:** Complete Phase 2 investigation to understand projection insertion
3. **Medium-term:** Implement comprehensive fix based on investigation findings
4. **Long-term:** Add architectural documentation about optimizer phase ordering and plan stability guarantees

