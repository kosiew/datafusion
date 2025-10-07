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

---

## Regression Analysis (Commit `a75b763e4`)

- **Observed impact:** Criterion shows severe slowdowns for dense workloads (`+105%` to `+107%` mean time) even though sparse cases improved modestly.
- **Root cause:** The refactor now performs a full pre-pass over every batch, materialising all `Option<&[u8]>` inputs into `batch_values` and inserting each group id into a `HashSet` just to estimate density. This duplicates the hot-path work and adds an `O(batch_len)` hash lookup for every row before the actual aggregation runs, which overwhelms any benefit of enabling the dense scratch table earlier.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L498-L575】
- **Secondary effect:** The new `dense_candidate_this_batch` flag forces the dense scratch path to allocate eagerly, so dense batches now pay for both the hash pre-pass and the dense scratch zeroing during the same call, compounding the regression.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L526-L575】

## Tasks to Address the Regression

1. **Eliminate the pre-pass hash scan.** Track `unique_groups` and `max_group_index` inside the main update loop (where we already discover first-touch events through `scratch_group_ids`) so that we avoid building `batch_values`/`HashSet` and regain streaming processing.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L543-L613】
2. **Restore amortised dense allocation.** Re-evaluate when to flip `scratch_dense_enabled`: gate it on the cheaper metrics gathered in the main loop so dense scratch only initialises after we confirm re-use, preventing simultaneous hash and dense overheads on the first batch.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L526-L575】
3. **Add dense benchmark coverage.** Extend the Criterion suite (or unit tests) to assert dense workloads do not regress, catching future attempts to add expensive pre-processing before the aggregation loop.
