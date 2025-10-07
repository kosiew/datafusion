# Min/Max bytes dense-mode regression follow-up

## Observed regression
Criterion detected statistically significant slowdowns in the "min bytes dense first batch" (~+3%) and "min bytes dense reused accumulator" (~+6%) benchmarks after the dense-inline mode changes.

## Root cause
The new dense-inline path collects per-batch density statistics inside `update_batch_dense_inline_impl`, updating `dense_inline_marks`, `unique_groups`, and `max_group_index` for every row so it can drive the mode switching heuristics in `record_batch_stats`. For dense workloads that already hit every group id, this extra bookkeeping happens on the hot path without changing the outcome, which adds overhead compared with the pre-change implementation that simply staged updates in a scratch vector and wrote them back once per batch.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L677-L741】【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L858-L934】

## Tasks
- [ ] Profile `update_batch_dense_inline_impl` on the dense benchmarks to measure how much time is spent updating `dense_inline_marks`, `unique_groups`, and `max_group_index`, and confirm they are the new hotspots.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L705-L725】
- [ ] Optimize the dense-inline bookkeeping so dense workloads avoid the redundant per-row tracking (e.g. reuse the old contiguous scratch approach or add a fast path when the batch touches every group sequentially) while still supplying the data that `record_batch_stats` needs.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L705-L734】【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L873-L934】
- [ ] Extend the benchmarks to cover the optimized path and guard against regressions in both dense and sparse scenarios once the fix lands.【F:datafusion/functions-aggregate/benches/min_max_bytes.rs†L58-L118】
