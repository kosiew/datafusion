# Tasks to Resolve `MinMaxBytesState` Performance Regression

## Root Cause Summary

`MinMaxBytesState::update_batch` now buffers every value from the input iterator
into a temporary `Vec` before the main loop begins so that it can replay the
current element whenever the dense scratch table is toggled mid-iteration.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L520-L561】

This eagerly materialised `values` buffer doubles the amount of work the kernel
performs on each batch: we traverse the Arrow array once to populate the buffer
and then a second time to perform the comparisons. Dense workloads (where every
row participates in the aggregation) pay the full cost of the extra pass yet do
not benefit from the sparse-path optimisations, leading to the 30–70% slowdowns
observed in the dense Criterion benchmarks.

## Remediation Tasks

1. **Eliminate the extra pass over the batch.**
   * Refactor `update_batch` so that it can retry the current element when the
     dense scratch table is enabled without materialising the entire iterator up
     front (for example by storing just the in-flight value or by restructuring
     the control flow to avoid the re-entry loop).【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L520-L672】
   * Confirm the rewrite still avoids allocating buffers proportional to
     `total_num_groups` for sparse workloads.
2. **Restore dense-path performance coverage.**
   * Augment the Criterion benchmarks (or add micro-benchmarks) so we have
     explicit coverage for dense-first and dense-reused scenarios, ensuring the
     refactor delivers regressions-free throughput before merging.【F:datafusion/functions-aggregate/benches/min_max_bytes.rs†L1-L204】
   * Add a unit test that exercises a single dense batch without prior state to
     catch future regressions in the activation path.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L861-L937】
3. **Document the streaming contract.**
   * Update the inline comments near the dense/sparse switching logic to record
     that the iterator must remain streaming (single-pass) so that future
     changes do not accidentally reintroduce the buffered pass.【F:datafusion/functions-aggregate/src/min_max/min_max_bytes.rs†L520-L736】
