source: aggmetrics-02-23570a
# Replace parallel aggregate accumulator timer vectors with phase-indexed storage

## Problem

`AggregateAccumulatorMetrics` in `datafusion/physical-plan/src/aggregates/group_values/metrics.rs` stores one optional `Vec<Time>` for each phase: `update_times`, `merge_times`, `state_times`, and `evaluate_times`. Construction repeats phase-membership checks for every field; `time` repeats the phase-to-field routing in a four-arm match.

The representation duplicates the `AccumulatorPhase` vocabulary in storage, construction, and lookup. Adding or renaming a phase requires synchronized edits across all three places, and a missed edit can silently omit metrics in release builds because an unavailable phase produces no timer after the debug assertion is compiled out.

## Why it matters

Per-aggregate metrics are a user-visible observability contract. The current representation makes phase coverage harder to audit and raises the cost and risk of extending the metric set. It also obscures the key distinction between:

- a phase not applicable to an aggregate mode, for which no metric must exist; and
- a phase invoked by execution but not initialized, which is a correctness bug.

This refactor should reduce mechanical duplication without allocating metrics for phases the execution mode does not perform or changing metric names, labels, timing boundaries, or visibility.

## Invariant / desired behavior

For each `AccumulatorPhase`, an aggregate accumulator metric is either:

1. absent because that phase is not initialized for the aggregate mode; or
2. present once per aggregate expression, partition, and metric set, named `agg_expr_{index}_{phase}_time` and labelled with the existing `aggregate` label.

A phase invoked through `AggregateAccumulatorMetrics::time` must have been initialized for that execution path. The storage abstraction must preserve the current lazy/selected-phase metric visibility: it must not register zero-value timers for phases that are not applicable.

## Proposed direction

Replace the four parallel optional vectors with one private phase-indexed container owned by `AggregateAccumulatorMetrics`. Keep `AccumulatorPhase` as the canonical key and make construction and lookup use the same mapping rather than separate field-specific logic.

The representation may use an array indexed by a private `AccumulatorPhase` index, an enum-keyed map, or an equivalent fixed-size private type. Prefer a fixed, allocation-light representation because lookup runs in per-aggregate/per-batch execution paths. Keep the public-to-crate API narrow: callers should continue to request timing by `(aggregate_index, phase)` and should not manipulate storage directly.

This work depends on resolving the separate spill-path correctness gap: final and single aggregate spill paths call `take_state_batch`, so their configured phase sets must include `State`. Do not encode the current omission as an intentional absent phase.

## Scope

### In

- Refactor `AggregateAccumulatorMetrics` construction and lookup in `datafusion/physical-plan/src/aggregates/group_values/metrics.rs` to use one private phase-indexed representation.
- Preserve current per-aggregate metric names, `aggregate` labels, partition association, and timer creation at table/stream construction time.
- Preserve absence of metric families for phases not executed by the selected aggregate mode.
- Retain debug assertions that identify invalid aggregate indexes and invoked-but-uninitialized phases.
- Extend focused metrics tests for every aggregate mode and its present/absent phase families.

### Out

- Adding new accumulator phases or changing aggregate execution semantics.
- Changing `GroupsAccumulator`, `Accumulator`, or other public trait APIs.
- Registering all four phase metrics for every aggregate mode.
- Renaming metric families, changing labels, or changing `EXPLAIN ANALYZE` display policy.
- Unifying hash and ordered aggregate execution flows.

## Acceptance criteria

- [ ] `AggregateAccumulatorMetrics` has one canonical private phase-indexed storage path; no field-per-phase storage or separate phase-routing match remains.
- [ ] For every supported aggregate mode, registered metric families exactly match the phases that mode can execute, including spill-driven `State` materialization where applicable.
- [ ] A mode that does not execute a phase does not register that phase's zero-value metric family.
- [ ] Existing metric names (`agg_expr_{index}_{phase}_time`), aggregate labels, partition labels, and timer timing boundaries are unchanged.
- [ ] Same-function aggregates over different inputs remain separately labelled and timed.
- [ ] Debug assertions catch an invalid aggregate index or an invoked-but-uninitialized phase; normal supported paths do not trigger them.
- [ ] Release behavior does not silently lose timing for a phase that a supported execution path invokes once the spill-path correctness prerequisite is resolved.

## Tests / verification

- Unit tests in or adjacent to `datafusion/physical-plan/src/aggregates/group_values/metrics.rs` covering construction and lookup for `Update`, `Merge`, `State`, and `Evaluate`.
- Table-driven test for `Partial`, `PartialReduce`, `Final`, `FinalPartitioned`, `Single`, and `SinglePartitioned`, asserting expected present and absent metric families.
- Regression tests for the separate spill-path prerequisite must force hash and ordered final/single aggregation to spill and assert `agg_expr_*_state_time` exists after collection.
- Run focused tests:
  ```bash
  cargo test -p datafusion-physical-plan aggregates::group_values::metrics
  cargo test -p datafusion-physical-plan aggregates
  ```

## Dependencies / blockers

- Resolve the final/single spill-path `State` phase initialization gap before relying on mode coverage as the timer-storage contract.
- Keep `accumulator_phases` in `datafusion/physical-plan/src/aggregates/aggregate_hash_table/mod.rs` as the canonical mode-to-phase selection boundary; storage must not duplicate that responsibility.
- Prefer a fixed-size representation if it avoids map allocation and hashing in the timing hot path, subject to project style.
