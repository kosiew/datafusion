# MinMaxBytesAccumulator Performance Issue Triage

## 1. Issue Analysis
- **Summary:** `MinMaxBytesAccumulator::update_batch` allocates a `locations` buffer sized to `total_num_groups` for every batch processed. Because `total_num_groups` grows with the number of distinct groups seen so far, later batches allocate increasingly large vectors, causing quadratic work and memory churn during high-cardinality aggregations (for example `MIN`/`MAX` of `Utf8` columns).
- **Actual vs. Expected:** Currently, throughput drops sharply as more groups are encountered because each batch re-allocates and zeroes the ever-growing `locations` vector. Expected behavior is near-linear scaling, only touching rows that appear in the current batch instead of all historical groups.
- **Reproducibility:** Clear. Running `SELECT l_orderkey, l_partkey, MIN(l_comment) FROM lineitem GROUP BY l_orderkey, l_partkey` on a large dataset shows the slowdown.

## 2. Codebase Scope
- **Primary Modules:** `datafusion/functions-aggregate/src/min_max/min_max_bytes.rs` implements `MinMaxBytesAccumulator` and `MinMaxBytesState`. The hot path is `MinMaxBytesState::update_batch`, which resizes internal storage and constructs the per-batch `locations` vector.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L414-L486】
- **Supporting Utilities:** Uses `apply_filter_as_nulls` from `datafusion_functions_aggregate_common`, the `GroupsAccumulator` trait, and Arrow string/binary array conversions.
- **Recent Changes:** No recent regressions identified; the quadratic allocation pattern has been present since the current implementation of `update_batch` was introduced.

## 3. Classification
- **Type:** Bug (performance defect).
- **Severity:** Major — can make grouped `MIN`/`MAX` queries unacceptably slow on large cardinalities.
- **Scope:** Single component (aggregate string min/max accumulator) but impacts any plan that uses it.
- **Priority:** High, due to the measurable runtime degradation and memory pressure on realistic workloads.

## 4. Resolution Plan
1. Replace the grow-to-`total_num_groups` temporary buffer with a structure sized to the number of groups touched in the current batch (e.g., hash map from group id to `MinMaxLocation`).
2. Alternatively (or additionally) maintain a reusable scratch buffer inside `MinMaxBytesState` that can be reset per batch without repeated allocations.
3. Benchmark the new implementation against the provided repro query to verify linear scaling and ensure no regressions for small group counts.
4. Add regression tests or benchmarks to guard against reintroducing quadratic behavior.

## 5. Next Steps
- Recommend implementing the fix (no further clarification needed) and validating with targeted benchmarks.

## 6. Fix Location
- Fix in this repository. The problematic allocation lives in `MinMaxBytesState::update_batch`, and the DataFusion project already hosts the relevant code paths and abstractions.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L414-L486】
