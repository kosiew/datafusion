# Min/Max Bytes Accumulator Issue Analysis

## 1. Issue Analysis
- **Summary:** `MinMaxBytesAccumulator::update_batch` resizes its internal `min_max` vector to the total group count and then allocates a temporary `locations` buffer sized to `total_num_groups` on every call. When cardinality grows across batches, later iterations repeatedly allocate and zero increasingly large buffers, turning an intended linear update into quadratic work.
- **Actual vs. Expected Behavior:** Actual executions revisit every historical group each batch, incurring repeated buffer allocation and comparison even for groups untouched by the current input, which severely degrades throughput during high-cardinality `MIN`/`MAX` aggregations. Expected behavior is that each batch only examines the groups referenced by the batch's `group_indices`, maintaining near-linear scaling as more groups are seen.
- **Reproduction Steps Clarity:** The reported SQL query `SELECT l_orderkey, l_partkey, MIN(l_comment) FROM lineitem GROUP BY l_orderkey, l_partkey` exercises the pathological path and provides clear, end-to-end reproduction guidance.

## 2. Codebase Scope
- **Primary Modules:**
  - `datafusion/functions-aggregate/src/min_max/min_max_bytes.rs` defines `MinMaxBytesAccumulator` and `MinMaxBytesState`, including the problematic per-batch `locations` buffer and `min_max` resizing logic.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L46-L138】【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L214-L311】
  - `datafusion/functions-aggregate/src/min_max.rs` wires the accumulator into the public aggregate factory, so any fix must stay compatible with these entry points.【F:datafusion/functions-aggregate/src/min_max.rs†L338-L626】
- **Dependencies:** The accumulator depends on Arrow string/binary array builders, DataFusion's group accumulator traits, and `apply_filter_as_nulls` from `datafusion_functions_aggregate_common`, so changes must preserve those interfaces.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L17-L102】【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L138-L184】 No external crates beyond the existing workspace are implicated.
- **Recent Changes:** `git log` shows only documentation-oriented updates touching this file (`Improve documentation for ordered set aggregate functions (#17744)`), suggesting the performance regression is longstanding rather than recently introduced.【4f2046†L1-L6】

## 3. Classification
- **Type:** Bug (performance defect in existing functionality).
- **Severity:** Major — the quadratic behavior can stall realistic high-cardinality queries.
- **Scope:** Single component (the bytes min/max accumulator) with localized impact on affected aggregates.
- **Priority:** High, because it directly impacts core aggregate performance for common workloads.

## 4. Resolution Plan
1. Refactor `MinMaxBytesState::update_batch` to avoid allocating a `locations` vector proportional to `total_num_groups`; track only the groups seen in the current batch (e.g., via small vector + hash map or by caching locations in a reusable buffer indexed by batch-local positions).
2. Ensure the accumulator skips touching untouched groups by early filtering with `group_indices` and reuses per-accumulator buffers to avoid repetitive allocations.
3. Add focused benchmarks or regression tests that simulate high-cardinality `MIN`/`MAX` workloads to verify the fix and guard against future regressions.
4. Validate memory accounting (`total_data_bytes`) remains correct after the refactor and adjust any size reporting if necessary.

## 5. Next Steps
- **Chosen Path:** Patch directly — the bug is well-understood, reproducible, and isolated to this implementation.

## 6. Fix Location
- **Decision:** Fix in this repository. The problematic logic lives entirely within `MinMaxBytesState::update_batch`, so the remediation can be implemented locally without upstream Arrow changes. Any further optimization ideas (e.g., specialized Arrow kernels) can be explored later if needed.
