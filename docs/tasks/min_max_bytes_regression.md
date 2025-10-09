# MinMaxBytesAccumulator Regression Analysis

## Regression Summary
Commit `325bafa73` introduced a `locations` scratch buffer inside `MinMaxBytesState::update_batch`. The buffer was resized to `total_num_groups` on every batch, even though each batch only touched a subset of groups. As `total_num_groups` grows with every new group, later batches pay to allocate and zero ever-larger vectors, yielding quadratic work and memory churn. This explains the across-the-board slowdown observed in the Criterion benchmarks, especially on dense/high-cardinality workloads.

## Root Cause Details
* The new implementation constructs `locations = vec![MinMaxLocation::ExistingMinMax; total_num_groups]` for every batch. This immediately scales with the lifetime number of groups, not the number of rows in the current batch.
* During iteration, the code only ever populates entries for groups seen in the batch, leaving the rest unused. However, the upfront allocation cost dominates once `total_num_groups` becomes large.
* Because the vector was rebuilt per batch, no reuse occurred between batches, so the cost repeated for every incoming batch.

## Proposed Tasks
1. **Introduce Reusable Scratch Structures** – Replace the per-batch `locations` allocation with reusable scratch state (e.g., epoch-tagged dense arrays or sparse hash maps) that only touches groups observed in the current batch.
2. **Track Batch Density** – Add lightweight statistics (unique groups, max group index) so the accumulator can choose between dense and sparse update strategies without scanning the full group domain each time.
3. **Extend Benchmarks & Tests** – Augment Criterion benchmarks and regression tests to cover high-cardinality multi-batch scenarios, ensuring future changes cannot regress back to quadratic allocations.
4. **Update Size Accounting & Documentation** – Adjust `size()` / memory accounting to include any new scratch structures and document the adaptive strategy so future contributors understand the trade-offs.
