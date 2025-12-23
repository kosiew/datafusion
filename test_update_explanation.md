# Test Result Updates - Distribution Satisfaction Refactoring

## Overview

This commit updates test results to reflect refined repartitioning logic introduced in the distribution satisfaction refactoring (commits `bbe54ad6f` through `a306d68ca`).

## Background: The Issue #18989 Panic

The changes were motivated by [issue #18989](https://github.com/apache/datafusion/issues/18989), which involved a panic when executing chained aggregations on an **empty, multi-partitioned MemTable**. The reproducer case (`datafusion-examples/examples/issue_18989.rs`) demonstrates the problem:

```rust
// create an empty but multi-partitioned MemTable
let mem_table = MemTable::try_new(schema.clone(), vec![vec![], vec![]]).unwrap();

// aggregate and sort twice - this would panic
let data_frame = ctx
    .table("metrics").await.unwrap()
    .aggregate(vec![col("region"), col("ts")], vec![count_udaf().call(vec![col("value")])])
    .unwrap()
    .sort(vec![col("region").sort(true, true), col("ts").sort(true, true)])
    .unwrap()
    .aggregate(vec![col("ts")], vec![count_udaf().call(vec![col("count(metrics.value)")])])
    .unwrap()
    .sort(vec![col("ts").sort(true, true)])
    .unwrap();
```

The panic occurred because the optimizer's previous logic for determining when hash repartitioning was "necessary" (`hash_necessary` flag) was based on whether the input already had multiple partitions. For empty multi-partitioned inputs, this heuristic failed to insert proper repartitioning between chained aggregates, leading to inconsistent partition distributions and runtime panics.

## Root Cause

The refactoring removed the flawed `hash_necessary` flag logic and introduced a new `DistributionSatisfactionResult` struct with a `requires_repartition()` method that more precisely determines when repartitioning is needed.

### The Flawed `hash_necessary` Logic (before `bbe54ad6f`)

The old code used a `hash_necessary` flag that was set to `true` when:
1. The requirement was hash partitioning, AND
2. The input already had more than one partition (`multi_partitions = output_partitioning.partition_count() > 1`)

```rust
let hash_necessary = is_hash && multi_partitions;
```

This logic had a critical flaw: **it would skip repartitioning for single-partition inputs even when the target partition count was higher**. For empty multi-partitioned inputs, this could lead to inconsistent distributions between chained operations.

Additionally, there was an "alignment" mechanism that would set `hash_necessary = true` for all hash requirements if *any* child needed alignment, but this was triggered by heuristics (`multi_partitions || roundrobin_sensible`) that didn't account for all cases where repartitioning should occur.

### Previous Logic (immediate predecessor of `bbe54ad6f`)

```rust
let needs_repartition = if allow_subset_satisfy_partitioning {
    !satisfaction.is_satisfied()
} else {
    !satisfaction.is_satisfied() 
        || n_target > input.plan.output_partitioning().partition_count()
};
```

This had a subtle bug: when `allow_subset_satisfy_partitioning` was enabled, it would **not** repartition even when `target_partitions > current_partitions`, missing opportunities to increase parallelism.

### Current Logic (after `a306d68ca`)

```rust
pub fn requires_repartition(&self, allow_subset: bool, target_partitions: usize) -> bool {
    (match self.satisfaction {
        PartitioningSatisfaction::Exact => false,
        PartitioningSatisfaction::Subset => !allow_subset,
        PartitioningSatisfaction::NotSatisfied => true,
    }) || (!allow_subset && target_partitions > self.output_partitioning.partition_count())
}
```

The refined logic now correctly:
1. **Never repartitions** when distribution is exactly satisfied
2. **Conditionally repartitions** for subset satisfaction (only when `allow_subset=false`)
3. **Always repartitions** when not satisfied
4. **Additionally repartitions** when `target_partitions > current_partitions` and subset satisfaction is disabled

## Why Non-Empty RecordBatches Are Also Affected

While the issue was discovered through an **empty multi-partitioned MemTable**, the fix correctly applies to **all cases**, including non-empty record batches. Here's why:

### The Core Problem Was General, Not Specific to Empty Data

The old `hash_necessary` logic made repartitioning decisions based on:
- Whether the current input already had multiple partitions
- Heuristic alignment rules

But the **correct** decision should be based on:
- Whether the current partitioning **satisfies** the required distribution
- Whether we need to increase parallelism to reach `target_partitions`

This applies equally to empty and non-empty data.

### Example: Single-Partition Input Requiring Hash Distribution

Consider a query with a window function that needs partitioning by column `a`:

**Why does a single partition NOT satisfy `HashPartitioned([a])`?**

A single partition means `SinglePartition` distribution (all data in one place). This is fundamentally different from `HashPartitioned([a])`, which requires:
- Data distributed across **multiple** partitions
- Each row placed in a partition determined by `hash(a) % partition_count`
- Rows with the same `a` value guaranteed to be in the same partition

A single partition doesn't provide this guarantee structure—it's just "everything together," not "things with same hash together in specific partitions." This matters for algorithms that rely on the hash distribution property.

**Old behavior (with `hash_necessary`):**
- Input: `DataSourceExec: partitions=1` (distribution: `SinglePartition`)
- Requirement: `HashPartitioned([a])`
- `hash_necessary = false` (because `partition_count() = 1`, not `> 1`)
- **No repartitioning inserted** → single-threaded execution

**New behavior (with `requires_repartition()`):**
- Input: `DataSourceExec: partitions=1` (distribution: `SinglePartition`)
- Requirement: `HashPartitioned([a])`
- `satisfaction = NotSatisfied` (single partition ≠ hash partitioned)
- `requires_repartition() = true` (not satisfied, and target=4 > current=1)
- **Repartitioning inserted** → `RepartitionExec: partitioning=Hash([a], 4)` → parallel execution with proper hash distribution

### Why Test Plans Changed

The refined logic now properly inserts repartitioning to achieve parallelism for operations like:
1. **Window functions** - Now correctly partition by the `PARTITION BY` columns
2. **Joins** - Now ensure both sides are properly hash-partitioned
3. **Aggregates** - Now use `FinalPartitioned` mode with proper hash distribution

These improvements apply to **all data**, not just empty tables. The test updates (commit `f6b132742`) reflect the optimizer now generating **correct, more parallel plans** for non-empty record batches that benefit from distributed execution.

### Summary

The fix for issue #18989 was not a narrow patch for empty tables—it was a **fundamental correction** to how DataFusion decides when repartitioning is needed. The test updates reflect the optimizer now making better decisions for all queries, empty or not, that can benefit from parallel execution.

## Detailed Impact on Test Results

The corrected logic results in the following observable changes:

### 1. New RepartitionExec Nodes

Operations that benefit from parallel execution (window functions, joins, aggregates) now show explicit `RepartitionExec` nodes with hash partitioning:

```diff
-03)----SortExec: expr=[a@0 ASC NULLS LAST], preserve_partitioning=[false]
-04)------DataSourceExec: partitions=1, partition_sizes=[1]
+03)----SortExec: expr=[a@0 ASC NULLS LAST], preserve_partitioning=[true]
+04)------RepartitionExec: partitioning=Hash([a@0], 4), input_partitions=1
+05)--------DataSourceExec: partitions=1, partition_sizes=[1]
```

### 2. Preserve Partitioning Flag Changes

Sort operators now correctly set `preserve_partitioning=[true]` to maintain the hash partitioning created by upstream repartitioning operations.

### 3. Mode Changes for Aggregates

Aggregates that previously ran in `SinglePartitioned` mode now properly use `FinalPartitioned` mode with explicit repartitioning in the `Partial` phase:

```diff
-03)----AggregateExec: mode=SinglePartitioned, gby=[1 as Int64(1)], aggr=[]
-04)------PlaceholderRowExec
+03)----AggregateExec: mode=FinalPartitioned, gby=[Int64(1)@0 as Int64(1)], aggr=[]
+04)------RepartitionExec: partitioning=Hash([Int64(1)@0], 4), input_partitions=1
+05)--------AggregateExec: mode=Partial, gby=[1 as Int64(1)], aggr=[]
+06)----------PlaceholderRowExec
```

## Files Affected

The following test files have updated expected plans to match the new, correct behavior:

- `count_star_rule.slt` - Window functions with partition by
- `cte.slt` - Semi joins
- `explain_tree.slt` - Join visualization
- `limit.slt` - Grouped aggregates
- `qualify.slt` - Window functions with qualify
- `sort_merge_join.slt` - Sort-merge joins with filters
- `union.slt` - Union with aggregates

## Verification

All updated plans show:
1. Proper hash partitioning to leverage parallelism
2. Correct preservation of partitioning through sort operators
3. Appropriate use of partitioned aggregate modes

These changes represent the **correct** behavior - the optimizer now properly inserts repartitioning to achieve parallelism when beneficial, which was previously missed due to the flawed conditional logic.
