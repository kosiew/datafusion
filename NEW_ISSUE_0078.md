source: pr-22752_a
# Extract a skip-aggregation aggregate test fixture/helper

## Summary

The skip-partial-aggregation tests in `datafusion/physical-plan/src/aggregates/mod.rs` repeat the same setup for schemas, group-by expressions, aggregate expressions, `TestMemoryExec`, `AggregateExec`, and session config. This makes each new boundary/regression test longer than the behavior it is trying to verify.

Extract a small local test helper/fixture so skip-aggregation tests can focus on input batches, threshold config, and expected output/metrics.

## Motivation

Recent fixes around `datafusion.execution.skip_partial_aggregation_probe_ratio_threshold` need careful boundary coverage:

- ratio greater than threshold: skip should activate
- ratio equal to threshold: skip should not activate
- threshold `1.0`: feature should be effectively disabled
- multi-batch behavior: skip decision affects subsequent batches

The current tests duplicate boilerplate around physical aggregate construction. This increases maintenance cost and makes it easier for future tests to drift in subtle ways, such as using different aliases, nullable fields, count expression setup, or config value types.

## Current state

Nearby tests such as:

- `test_skip_aggregation_after_first_batch`
- `test_skip_aggregation_after_threshold`
- `test_skip_aggregation_disabled_at_threshold_one`

all build a similar physical plan:

1. create schema with `key` and `val`
2. create `PhysicalGroupBy::new_single`
3. create `COUNT(val)` aggregate expression
4. create `TestMemoryExec`
5. create `AggregateExec` in `AggregateMode::Partial`
6. configure skip-aggregation thresholds in `SessionConfig`
7. execute and collect output or metrics

Most of this setup is incidental to the actual behavior under test.

## Proposed change

Add a small helper in the aggregate test module, for example:

```rust
fn make_skip_aggregation_exec(
    input_data: Vec<RecordBatch>,
) -> Result<(Arc<AggregateExec>, Arc<Schema>)> {
    // create common schema, group_by, COUNT(val), TestMemoryExec,
    // and partial AggregateExec
}

fn skip_aggregation_session_config(
    probe_rows_threshold: i64,
    probe_ratio_threshold: f64,
) -> SessionConfig {
    SessionConfig::default()
        .set(
            "datafusion.execution.skip_partial_aggregation_probe_rows_threshold",
            &ScalarValue::Int64(Some(probe_rows_threshold)),
        )
        .set(
            "datafusion.execution.skip_partial_aggregation_probe_ratio_threshold",
            &ScalarValue::Float64(Some(probe_ratio_threshold)),
        )
}
```

If useful, add another helper for common batch creation:

```rust
fn int32_batch(schema: Arc<Schema>, keys: Vec<i32>, vals: Vec<i32>) -> RecordBatch
```

Keep helpers local to the test module unless broader reuse appears.

## Acceptance criteria

- Existing skip-aggregation tests keep the same behavior and expected output.
- Repeated aggregate setup is reduced in touched tests.
- Test names and assertions remain behavior-focused.
- Helper does not obscure important differences between tests, especially threshold values and batch boundaries.
- `cargo test -p datafusion-physical-plan skip_aggregation` passes.

## Non-goals

- Do not change runtime skip-aggregation behavior.
- Do not introduce a public test utility API.
- Do not refactor unrelated aggregate tests unless they naturally use the same helper.
- Do not replace behavior assertions with only helper-level tests.

## Risk / considerations

- Avoid over-generalizing the helper. A small fixture for the common `COUNT(val) GROUP BY key` partial aggregation case is enough.
- Keep config values explicit at call sites so threshold-boundary tests remain easy to inspect.
- Preserve output aliases and schema shape used by current snapshot tests.

## Suggested validation

Run:

```bash
cargo test -p datafusion-physical-plan skip_aggregation
```
