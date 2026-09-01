# Extra Agent Guidelines for Apache DataFusion

## Workflow / CI

- Migrated Rust CI steps are defined in `xtask/src/ci_steps.rs` and invoked by
  Actions and locally through `cargo xtask ci step <step> <target>`. Use
  `--explain` to inspect the exact command; update `xtask` rather than duplicating
  those CI commands in workflows.
- The workspace MSRV is Rust `1.94.0`; keep workflow/toolchain/docs changes
  aligned with `Cargo.toml`.
- GitHub Actions workflow Rust tool installs should use `taiki-e/install-action`, not
  `cargo install`. Validate workflow edits with:

```bash
ci/scripts/check_no_cargo_install_in_workflows.sh
```

- The main Rust CI test job records coverage with `cargo llvm-cov` and uploads
  `target/codecov.json` to Codecov.
- Dev builds use line-table debug info. Use `CARGO_PROFILE_DEV_DEBUG=2` when
  interactive debugging needs local variables. Doc-generator binaries require
  `--features docs_generation`; the `dev/update_*_docs.sh` scripts set it.

## Benchmarks

- SQL benchmarks use `benchmark_runner`; suite commands put the suite name before
  its options (`--list` needs no suite).
  Use `--list` to inspect suites, `--criterion` for Criterion mode,
  `--output <file>` for simple-run JSON, and `--save-baseline <name>` only with
  `--criterion`.
- Each discoverable suite needs `<suite>/<suite>.suite` TOML metadata. Keep
  option precedence CLI > environment > metadata default; use `--dry-run` to
  validate resolved options and path replacements without loading data or SQL.
- `--path` only works for suites declaring a `DATA_DIR` path replacement.
  Keep ClickBench setup aligned with the single-file vs partitioned registration
  notes in `benchmarks/README.md`.
- `parquet_row_filter_skip` measures fully matched Parquet RowFilter skipping;
  use its `skip` / `control` subgroups and `PRED_ROWS` / `RG_SIZE` knobs.
- `dfbench statistics` compares planned estimates with runtime operator metrics for
  Parquet tables and SQL files. It stores branch/query reports in
  `target/dfbench/statistics`; compare with the prior branch run or `--compare`.

## Architecture Notes

- Catalog/table contract traits now live in `datafusion-session`; `datafusion-catalog`
  re-exports them. Custom `Session` implementations must provide
  `catalog_list()` and can use `EmptyCatalogProviderList` when no catalog exists.
- Physical expression planning state for scalar subqueries and lambda-variable
  scope is explicit in `PhysicalPlanningContext`. Custom `PhysicalPlanner` /
  `ExtensionPlanner` implementations must accept and forward it when creating
  physical expressions; lambda bodies extend it to preserve nested shadowing.
- Generator-style streams live in `datafusion/execution/src/async_stream.rs` as
  `async_stream` / `async_try_stream`; prefer them over hand-written poll state
  machines when they fit.
- Physical plans can self-serialize with `ExecutionPlan::try_to_proto` and
  companion `try_from_proto` helpers in `datafusion/physical-plan/src/proto.rs`.
  Keep `datafusion-physical-plan` dependent only on `datafusion-proto-models`,
  not `datafusion-proto`.
- Spill backends can be supplied through
  `DiskManagerBuilder::with_temp_file_factory`; do not assume spills are local
  filesystem paths.
- `AsOfJoinExec` is a bounded, left-preserving broadcast join: it collects a
  single-partition right input and requires both inputs ordered by equality keys
  then the match key. Match/equality expressions must be deterministic and float
  equality keys are unsupported.
- `SortMergeJoinExec::projection` indexes its unprojected join schema. Preserve
  it through construction, child replacement, properties, execution, and serde.
- `ExecutionPlan::apply_expressions` is required and visits expressions owned
  and evaluated by the node. Replace children through
  `with_new_children_if_necessary`; implementations use `replace_children` and
  preserve properties only when replacement children have the same ones.
- Protobuf-model conversions between generated types and common types live in
  `datafusion-proto-models`; owner crates host conversions when the orphan rule
  permits. Keep `datafusion-proto` as conversion orchestration/compatibility.

## API / Contract Changes

- `CREATE EXTERNAL TABLE` supports multiple `LOCATION` paths. Logical and SQL
  structs use `locations: Vec<String>`; all paths must share schema and object
  store.
- `GroupsAccumulator::convert_to_state` is required; the
  `supports_convert_to_state` method and FFI field are removed.
- `FFI_Partitioning` preserves `Partitioning::Range` via
  `FFI_RangePartitioning`; FFI consumers must use fallible conversion back to
  native `Partitioning`.
- Projection pushdown into file scans must not duplicate volatile or non-trivial
  CSE'd expressions referenced more than once; only leaf-pushable expressions
  such as `get_field` / scan-metadata UDFs may still merge.
- `TableProvider::scan` projections are `Option<&[usize]>`; callers with an
  `Option<Vec<usize>>` pass `.as_deref()`. `None` requests all columns.
- Compatible range-partitioned hash joins support dynamic-filter pushdown.
  `ExprType::RangeExpr` is an additive physical-protobuf variant; exhaustive
  downstream matches must handle it.

## Additional Agent Notes

These notes capture change-sensitive conventions not yet obvious from the
contributor docs. Keep entries tied to real code/workflow behavior.

## Architecture

- Catalog, table, planner, and physical optimizer contracts live in
  `datafusion-session`; older public paths re-export them for compatibility.
  Custom planners should take `&dyn Session`, not `&SessionState`.
- Real `Session` implementations that plan queries must override
  `query_planner`, `optimize`, `physical_optimizers`, and
  `statistics_registry`. Defaults are intentionally minimal:
  unsupported planner, no logical optimization, no physical rules, no stats
  registry.
- File-scan leaf protobuf conversions for `FileRange`, `PartitionedFile`, and
  `FileGroup` are owned by `datafusion-datasource/src/proto.rs`. Treat
  `datafusion-proto` conversions for those types as thin shims.
- Physical plan / expression serde is moving to owner-side `try_to_proto` /
  `try_from_proto` hooks using encode/decode contexts. `DataSource` and
  `FileSource` implementations own their source nodes; `FileScanConfig` owns
  the shared file-scan spine. Avoid adding `datafusion-proto` dependencies to
  `datafusion-physical-plan`.
- `MERGE INTO` dispatches to `TableProvider::merge_into`. Providers receive the
  source execution plan, target-then-source qualified `DFSchema`, `ON`, and
  `WHEN` clauses; the default reports unsupported.
- Substrait correlated references serialize as `OuterReference` field references.
  Producers and consumers must push enclosing schemas at subquery boundaries and
  preserve the `steps_out` depth.

## Behavior contracts

- `PhysicalPlanningContext` must be forwarded through custom physical and
  extension planners so scalar subqueries share the same subtree state.
- `ExprProperties::preserves_lex_ordering` is non-strict monotonicity;
  `strictly_order_preserving` is required when replacing a sort key while
  preserving suffix ordering.
- `unnest_outer` returns one `NULL` row for `NULL` or empty input lists and
  cannot be mixed with `unnest` in the same `SELECT`.
- `array_distance` supports only one-dimensional arrays; multidimensional
  inputs are planning errors.
- `datafusion.execution.parquet.max_in_list_size` caps `IN (...)` pruning
  rewrites (default `20`); larger lists skip file, row-group, and page pruning,
  and `0` disables this pruning. New callers use `PruningPredicateBuilder`;
  `PruningPredicate::try_new` is deprecated.
- `WindowTopN` must run after `FilterPushdown` but before
  `EnsureRequirements` and `ProjectionPushdown`; preserve the documented
  physical optimizer rule order.
- `datafusion-cli --spark` opt-in enables the Spark dialect, expression planner,
  and Spark scalar, aggregate, and window functions. Default CLI behavior does
  not enable or register them.
- File scans derive single-column key distinct counts from primary/unique
  constraints when row counts are known; composite keys do not imply per-column
  distinct counts. Invalid `default_filter_selectivity` and `duration_format`
  values fail when set, not later during planning or display.
- Parquet `bytes_processed` measures resolved scan-range bytes. It credits read
  and pruned bytes, and remaining bytes on a terminal early exit; it differs from
  object-store-only `bytes_scanned`.

## Benchmarks

- When using `BenchmarkRun`, call
  `set_memory_pool(&ctx.runtime_env().memory_pool)` after building each runtime.
  Cases then emit `pool_peak_bytes` when a memory limit installs the recording
  pool.
