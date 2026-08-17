source: aggmetrics-02-23570a
# Centralize migrated hash and ordered aggregate metric construction

## Problem

Migrated hash and ordered aggregate tables independently build the same aggregate metric set:

- `AggregateHashTable::new_with_filters` in `datafusion/physical-plan/src/aggregates/aggregate_hash_table/common.rs`
- `OrderedAggregateTableMetrics::new` in `datafusion/physical-plan/src/aggregates/aggregate_hash_table/common_ordered.rs`

Each path derives aggregate labels from `AggregateExec::aggr_expr`, creates `GroupByMetrics`, creates `AggregateArgumentMetrics`, selects accumulator phases, and creates `Arc<AggregateAccumulatorMetrics>`. The ordered path must preserve this ownership because metric instances are shared when a hash table transitions to ordered replay.

Duplicated setup risks metric drift: a new label policy, metric group, phase selection input, or construction-time contract can be updated in one table family but omitted in the other.

## Why it matters

Hash and ordered aggregation have intentionally different accumulation, spill, and output lifecycles, but equivalent aggregate operations should expose equivalent metric definitions. Construction is the correct narrow shared boundary: it determines metric identity and ownership without coupling execution behavior.

A small shared constructor makes the metric contract auditable in one place while preserving the existing `Arc` sharing needed during hash-to-ordered spill/replay. Broadly merging table implementations would create unnecessary semantic risk.

## Invariant / desired behavior

For the same `AggregateExec` and partition, hash and independently constructed ordered aggregate tables register equivalent:

- `GroupByMetrics`;
- per-aggregate argument timers;
- per-aggregate accumulator timers for the same selected phases;
- aggregate labels, metric names, and partition association.

When an ordered table is built from or reuses a hash table during spill/replay, both tables must share the same accumulator metric instances rather than creating duplicate metric families. Metric timers must still be created once during table setup, never in a per-batch or per-row path.

## Proposed direction

Introduce a small crate-private aggregate-table metric factory/value type near the aggregate hash-table metric boundary. It should accept the existing narrow inputs (`AggregateExec`, partition, and canonical mode-to-phase selection) and return the three metric groups with ownership suitable for both consumers.

Make both direct hash-table and direct ordered-table construction call this factory. Preserve `OrderedAggregateTableMetrics::from_hash_table` (or an equivalent explicit transfer helper) as the ownership boundary for replay: it must clone ordinary metric handles and `Arc::clone` accumulator metrics.

Do not share table execution code, accumulator buffers, or spill mechanics. This issue is only metric creation and metric-handle transfer.

## Scope

### In

- Extract duplicated label collection and construction of `GroupByMetrics`, `AggregateArgumentMetrics`, and `AggregateAccumulatorMetrics` from `common.rs` and `common_ordered.rs`.
- Define clear private ownership for the returned metric groups, including `Arc<AggregateAccumulatorMetrics>` where sharing is required.
- Route direct hash and ordered table construction through the same factory.
- Preserve the hash-to-ordered replay path's sharing behavior.
- Add focused tests for metric equivalence and metric-instance reuse through replay where feasible.

### Out

- Merging `AggregateHashTable` and `OrderedAggregateTable` implementations.
- Changing ordered early-emission behavior, hash spilling, sorting, or output materialization.
- Changing phase-selection rules; use the existing canonical `accumulator_phases` helper.
- Renaming, adding, removing, or changing visibility of metrics.
- Replacing `Arc` sharing with copied metric sets.

## Acceptance criteria

- [ ] Direct hash and ordered table metric construction use one crate-private construction path for aggregate labels and the three metric groups.
- [ ] Equivalent direct hash and ordered plans register the same metric names, `aggregate` labels, partitions, and applicable phase families.
- [ ] Hash-to-ordered replay reuses accumulator metric instances via `Arc` sharing and does not register duplicate `agg_expr_*` metrics.
- [ ] Metric objects are created only during table/stream setup; no new metric registration occurs in batch processing or output/spill loops.
- [ ] The refactor does not alter hash or ordered aggregation results, schemas, spill behavior, or output ordering.
- [ ] The shared construction boundary remains small and does not expose table execution internals.

## Tests / verification

- Extend aggregate metrics tests to compare name/label sets from direct hash and direct ordered aggregate paths for the same expressions and mode.
- Add a replay/spill regression test that drives hash-to-ordered transition and asserts each expected per-aggregate metric family is registered once, not once per table instance.
- Retain result/schema assertions for representative partial, final, and single aggregate modes.
- Run focused tests:
  ```bash
  cargo test -p datafusion-physical-plan aggregates::group_values::metrics
  cargo test -p datafusion-physical-plan aggregates
  ```

## Dependencies / blockers

- `OrderedAggregateTableMetrics::from_hash_table` makes the sharing contract explicit: `GroupByMetrics` and `AggregateArgumentMetrics` are cloned while accumulator metrics use `Arc::clone`. Preserve this distinction unless the factory can express it more clearly without changing behavior.
- Do not make this refactor depend on a storage redesign within `AggregateAccumulatorMetrics`; either representation may be used as long as metric construction and sharing contracts remain intact.
