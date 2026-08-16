source: aggmetrics-01-23570a
# Instrument grouped TopK aggregate argument evaluation per aggregate expression

## Problem

`GroupedTopKAggregateStream` evaluates `aggregate_arguments` only inside the compatibility timer `aggregate_arguments_time` (`datafusion/physical-plan/src/aggregates/grouped_topk_stream.rs`). It does not create or use `AggregateArgumentMetrics`.

`AggregateExec::execute_typed` selects this stream for grouped aggregates with supported `limit_options` (including compatible grouped `MIN`/`MAX` query shapes with an upstream sort/limit). Consequently, the planner can select a valid grouped aggregate implementation whose `EXPLAIN ANALYZE` metrics omit the newly added indexed metrics:

- `agg_expr_0_arguments_time`
- `agg_expr_1_arguments_time`
- etc.

Other grouped paths construct `AggregateArgumentMetrics` once from `aggr.aggr_expr`, label each metric with the aggregate expression, and time expression evaluation by aggregate index. The TopK path currently exposes only the aggregate-wide compatibility metric.

## Why it matters

Aggregate argument metrics are a user-visible observability feature. Query shape and planner choice must not silently remove the per-aggregate timing breakdown.

Without this coverage, users cannot distinguish argument-evaluation cost for individual aggregates on the optimized TopK path. It also creates an inconsistent `AggregateExec` metrics contract: a query may show `SUM(a)` / `SUM(b)`-style indexed metrics on one grouped implementation but not on another solely because of an optimization choice.

## Invariant / desired behavior

Every reachable grouped `AggregateExec` path that evaluates aggregate arguments must expose the same per-aggregate argument metric family:

- one `agg_expr_{idx}_arguments_time` timer per aggregate expression;
- an `aggregate` label identifying that expression, using the established display/label convention; and
- the existing aggregate-wide `aggregate_arguments_time` timer retained for compatibility.

Timers must be registered once when the stream is constructed, then selected by aggregate index during batch evaluation. Instrumentation must not alter TopK aggregation results, schemas, selection behavior, or per-batch allocation characteristics.

## Proposed direction

Extend `GroupedTopKAggregateStream` to own an `AggregateArgumentMetrics` value constructed from the `AggregateExec` aggregate expressions during `new`.

At the `evaluate_many(&self.aggregate_arguments, ...)` boundary, retain the existing aggregate-wide timer and also time each aggregate argument evaluation with its matching indexed timer. Reuse the existing helper and metric naming/label behavior from `datafusion/physical-plan/src/aggregates/group_values/metrics.rs`; do not introduce a TopK-specific metric vocabulary.

Add a regression that constructs or plans a grouped TopK-compatible `MIN`/`MAX` aggregate, executes it, and verifies the TopK path reports the indexed metric name and aggregate label.

## Scope

### In

- Add `AggregateArgumentMetrics` construction and storage to `GroupedTopKAggregateStream`.
- Record per-aggregate argument-evaluation timing in the TopK stream.
- Preserve `aggregate_arguments_time`.
- Add focused regression coverage for a planner/execution shape selecting `GroupedTopKAggregateStream`.
- Assert exact indexed metric name and aggregate label, not merely that some timer exists.

### Out

- Per-aggregate timing for accumulator update, merge, state, or final evaluation phases.
- Changes to global/non-grouped aggregation metrics.
- Changes to TopK planning criteria, `LimitOptions`, aggregate semantics, or output ordering.
- Renaming existing metric families or changing label policy.
- Broader legacy/migrated grouped aggregate metric coverage beyond this missing TopK path.

## Acceptance criteria

- [ ] `GroupedTopKAggregateStream` registers `AggregateArgumentMetrics` from the aggregate expressions once per stream/partition.
- [ ] TopK aggregate argument evaluation records `agg_expr_{idx}_arguments_time` for each evaluated aggregate expression.
- [ ] Each indexed metric has the established `aggregate` label for its corresponding aggregate expression.
- [ ] `aggregate_arguments_time` remains present and continues to cover the aggregate-argument evaluation phase.
- [ ] A grouped TopK-compatible `MIN`/`MAX` regression proves `GroupedTopKAggregateStream` is the executed path and exposes an indexed timer with its expected label.
- [ ] The regression proves the indexed timer records evaluation work, rather than merely being registered.
- [ ] Aggregate results, schema, and TopK behavior are unchanged.

## Tests / verification

- Add a unit/regression test near existing aggregate metric tests or the TopK aggregate tests.
- Build a deterministic physical plan or SQL/planner case with `limit_options` that selects `GroupedTopKAggregateStream`; assert the selected stream variant/type before executing the metric assertions.
- Execute the plan and inspect `AggregateExec::metrics()`.
- Assert at least one exact `(metric name, aggregate label)` pair, e.g. `("agg_expr_0_arguments_time", "MIN(value)")` or the corresponding `MAX` expression used by the test, and prove its elapsed value reflects executed argument evaluation.
- Assert the compatibility `aggregate_arguments_time` metric still exists.
- Run targeted tests, including:

  ```bash
  cargo test -p datafusion-physical-plan aggregates::group_values::metrics --lib
  cargo test -p datafusion-physical-plan grouped_topk --lib
  ```

## Notes / open questions

- The TopK stream currently asserts one aggregate input in its `MIN`/`MAX` branch. The regression need only cover the supported one-aggregate shape; the implementation should still use the shared indexed-metric API rather than hard-code a separate metric name.
- Confirm the most stable existing TopK test helper/query shape before choosing whether this regression belongs in `grouped_topk_stream.rs`, `aggregates/mod.rs`, or `group_values/metrics.rs`.
