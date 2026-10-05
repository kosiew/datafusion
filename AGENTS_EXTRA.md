# Extra Agent Guidelines for Apache DataFusion

## Workflow / CI

- Migrated Rust CI steps are defined in `xtask/src/ci_steps.rs` and invoked by
  Actions and locally through `cargo xtask ci step <step> <target>`. Use
  `--explain` to inspect the exact command; update `xtask` rather than duplicating
  those CI commands in workflows.
- The workspace MSRV is Rust `1.95.0`; the pinned default toolchain is
  Rust `1.99.0`. Keep workflow/toolchain/docs changes aligned with
  `Cargo.toml` and `rust-toolchain.toml`.
- `./dev/rust_lint.sh` requires `python3` with PyYAML; run it through
  `uv run` when the current environment lacks PyYAML. It also checks generated
  config/function docs, the examples README, links, security, dependency
  cycles/unused dependencies, large files, and ASF status checks; `--write`
  updates the generated docs and README.
- GitHub Actions workflow Rust tool installs should use `taiki-e/install-action`, not
  `cargo install`. Validate workflow edits with:

```bash
ci/scripts/check_no_cargo_install_in_workflows.sh
```

- The main Rust CI test job records coverage with `cargo llvm-cov` and uploads
  `target/codecov.json` to Codecov. On direct upstream `main` pushes, Rust CI
  skips only the queue-verified jobs listed in
  `ci/scripts/check_asf_yaml_status_checks.py`; that script validates the exact
  skip list and Cargo-check artifact guards. Update it with `rust.yml` changes.
- `dev/update_config_docs.sh` and `dev/update_function_docs.sh` accept
  `--output-dir <dir>` (relative to repo root), enabling generated-doc checks
  without replacing the checked-in pages.
- Dev builds use line-table debug info. Use `CARGO_PROFILE_DEV_DEBUG=2` when
  interactive debugging needs local variables. Doc-generator binaries require
  `--features docs_generation`; the `dev/update_*_docs.sh` scripts set it.
- `rust_lint.sh` builds the HTML documentation via
  `ci/scripts/check_docs_html.sh`; it fails on Sphinx warnings and needs `uv`,
  `cargo`, `cargo-depgraph`, Graphviz `dot`, and `make` (installs none).
- `datafusion-examples` must depend on the umbrella `datafusion` crate, not
  DataFusion subcrates, except `datafusion-proto` and `datafusion-substrait`.
  `ci/scripts/check_examples_datafusion_crates.py` enforces this.
- Contributors without write access may have at most three open, non-draft PRs.
  Wait for merges or close PRs before opening another.

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
  Keep `datafusion-physical-plan` dependent on `datafusion-proto-models` and
  `datafusion-proto-common`, not `datafusion-proto`.
- `ParquetFileSchemaProvider` derives a file-only Arrow schema lazily after its
  Parquet footer loads. `PartitionedFile::arrow_schema` takes precedence;
  provider errors fail opening; plans using one require a custom serde codec.
- Spill backends can be supplied through
  `DiskManagerBuilder::with_temp_file_factory`; do not assume spills are local
  filesystem paths. Native `AsyncSpillWriter`s persist sequential owned `Bytes`
  in order, account retained buffers separately, and treat `finish` as the
  commit boundary; `abort` cleans up uncommitted uploads and may be retried.
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
  Exhaustively destructure serializable configs and reject behavior with no
  wire representation rather than silently dropping it.

## API / Contract Changes

- `CREATE EXTERNAL TABLE` supports multiple `LOCATION` paths. Logical and SQL
  structs use `locations: Vec<String>`; all paths must share schema and object
  store.
- `GroupsAccumulator::convert_to_state` is required; the
  `supports_convert_to_state` method and FFI field are removed. Accumulators
  and group accumulators may receive optional `AggregateMetrics` immediately
  after construction; aggregate-owned subphase identifiers must be stable and
  recorders must be installed before use.
- `FFI_Partitioning` preserves `Partitioning::Range` via
  `FFI_RangePartitioning`; FFI consumers must use fallible conversion back to
  native `Partitioning`.
- Projection pushdown into file scans must not duplicate volatile or non-trivial
  CSE'd expressions referenced more than once; only leaf-pushable expressions
  such as `get_field` / scan-metadata UDFs may still merge.
- `TableProvider::scan` projections are `Option<&[usize]>`; callers with an
  `Option<Vec<usize>>` pass `.as_deref()`. `None` requests all columns. A
  pushed-down `limit` is an upper bound on filtered rows, not a minimum.
- Compatible range-partitioned hash joins support dynamic-filter pushdown.
  `ExprType::RangeExpr` is an additive physical-protobuf variant; exhaustive
  downstream matches must handle it. Null-aware joins require equi-join keys;
  planning must reject a keyless null-aware join rather than choose an executor
  that lacks null-aware semantics.
- A `TableProvider` may push down offset only by overriding `scan_with_args` to
  honor `ScanArgs::skip()` exactly and returning true from
  `supports_skip_pushdown()`. Scan order is filters, skip, limit, projection;
  otherwise DataFusion keeps the skip above the scan. Build `TableScan` with
  `TableScanBuilder`, which carries its `skip` field and boxed statistics set.
- `FileScanConfig` derives equivalences only from `FileSource::exact_filter`,
  never the generic `filter`. A source returns it only for predicates every
  output row satisfies; pruning-only filters return `None`.

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
  `WHEN` clauses; the default reports unsupported. `MergeIntoOp` is
  non-exhaustive and uses `new(target_qualifier, on, clauses)`; qualify target
  fields by the SQL-visible alias, not the provider identity.
- `ensure_distribution_with_stats` takes a shared `StatisticsContext` for one
  bottom-up traversal. Its cache retains plan nodes, so rewrites need no cache
  reset for correctness; reset only at lifecycle boundaries to bound memory.
  `ensure_distribution` is deprecated.
- Correlated filters may move above only logical-plan nodes explicitly known to
  preserve their semantics. When adding a `LogicalPlan` variant, classify it in
  `PullUpCorrelatedExpr` rather than relying on a wildcard: scope boundaries,
  row-order/row-selection nodes, and nodes holding outer references stop the
  pull-up.
- `EnsureRequirements` normalizes `InterleaveExec` to `UnionExec` top-down,
  then re-derives interleaving from final child partitioning. Do not retain
  interleaves across child rewrites. A child rewrite may temporarily produce a
  non-interleavable `InterleaveExec`: report `UnknownPartitioning` and reject
  it only at the `Executable` invariant level, after requirements can repair it.
- Substrait correlated references serialize as `OuterReference` field references.
  Producers and consumers must push enclosing schemas at subquery boundaries and
  preserve the `steps_out` depth.

## Behavior contracts

- `PhysicalPlanningContext` must be forwarded through custom physical and
  extension planners so scalar subqueries share the same subtree state.
- `GroupSelection` preserving reads validate the stored group count and retain
  selected order and duplicates. `GroupColumn::values_preserving` and
  `GroupsAccumulator::{evaluate,state}_preserving` must not alter values,
  logical state, or group indices.
- Physical filter pushdown resolves columns by position. Use `from_child` only
  for identity-position schemas; projections/reorders use
  `from_child_with_column_mapping`. The allowed-indices helper is deprecated
  because duplicate child names make name lookup ambiguous.
- `ExprProperties::preserves_lex_ordering` is non-strict monotonicity;
  `strictly_order_preserving` is required when replacing a sort key while
  preserving suffix ordering.
- `unnest_outer` returns one `NULL` row for `NULL` or empty input lists and
  cannot be mixed with `unnest` in the same `SELECT`.
- `array_distance` supports only one-dimensional arrays; multidimensional
  inputs are planning errors.
- `datafusion.execution.parquet.max_in_list_size` caps min/max `IN (...)`
  pruning rewrites (default `20`); `0` disables them. Lists over the cap skip
  min/max rewriting; within the cap, large literal string lists use compact
  pruning for `IN` and `NOT IN`, including NULL members. Bloom-filter pruning
  remains available. New callers use `PruningPredicateBuilder`;
  `PruningPredicate::try_new` is deprecated.
- `AggregateUDFImpl::distinct_handling` declares duplicate behavior. Return
  `Insensitive` only for idempotent merges; the optimizer removes `DISTINCT`
  before `SingleDistinctToGroupBy`. `Sensitive` (default) owns deduplication;
  `Unsupported` requires planning-time deduplication or rejection.
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
- `AggregateMode::PartialReduce` is best-effort: output may be partially reduced
  or unchanged, and group keys may repeat across batches. Consumers merge it as
  `Partial` output.
- `enable_nlj_coordinated_fallback` defaults true. Distributed engines that run
  right probe partitions in separate tasks/processes must set it false: the
  coordinated LEFT/SEMI/ANTI/MARK/FULL fallback requires one process and would
  otherwise stall; under memory pressure those joins then fail exhausted.
- `StatisticsContext` is the statistics walk. `StatisticsRegistry::compute*` and
  `DefaultStatisticsProvider` are deprecated; an empty/delegating registry falls
  back to `statistics_from_inputs`. Providers should implement `matches` and
  return one valid `child_stats_requests` entry per child.
- `udaf_default_*` display/schema helpers are deprecated; UDAFs use the
  `Udaf*Builder` builders. `with_dynamic_filter_expr` on `SortExec`,
  `AggregateExec`, and `HashJoinExec` is deprecated/serde-only; normal planning
  creates dynamic filters through pushdown paths.
- Common proto option and constraint conversions are fallible `TryFrom`; reject
  out-of-range `usize` values and missing `constraint_mode` rather than casting
  or panicking.
- `ForeignSession::create_physical_plan` is deliberately unsupported to avoid
  re-entering an installed foreign planner. Export and retain the original
  `FFI_QueryPlanner` before installation, then invoke it directly.
- `datafusion.sql_parser.recursion_limit` defaults to `51`; explicit shallow
  limits may need raising after the sqlparser upgrade.
- One-argument `from_unixtime` uses `datafusion.execution.time_zone`; an
  explicit timezone overrides it. Reject timezone-shifted values outside
  `NaiveDateTime` range before formatting can panic.
- Parquet page-level limit pruning is valid only after every conjunct yields an
  invertible page selection. Intersect it with any existing row selection and
  replace the access plan only once fully matched rows satisfy the limit.
- Multiple-file output also rotates at
  `datafusion.execution.soft_max_bytes_per_output_file` (default
  `4_294_967_295` bytes), based on asynchronously reported encoded bytes. It is
  a soft target and does not apply to single-file output; buffered batches and
  file metadata can exceed it.
- `ExecutionOptions::time_zone` is `Option<ConfigTimeZone>`: parse direct
  assignments and use `as_str()` to read it. Invalid values fail at `SET` time.
  For mixed timezone-aware/naive timestamp comparison or subtraction, the
  configured session zone interprets the naive operand; without one, use the
  aware operand's zone. Use `comparison_coercion_with_session_timezone` for
  non-binary comparison contexts (`IN`, `BETWEEN`, `CASE`, subqueries, ANY/ALL).
- `datafusion.execution.parquet.row_group_range_assignment` assigns row groups
  in split files by `start_offset` (default) or Spark-compatible `midpoint`.
  `RowGroupAccessPlanFilter::prune_by_range` requires that assignment. Preserve
  `RowSelection`'s bitmap representation through Parquet access-plan splitting
  and filter toggles; do not force it into selectors.
- Substrait producers must set `output_type` from the derived expression field
  for scalar and window function calls, including nested `not(like(...))`.
- Per-partition fetch operators report overall bounds as
  `min(input_rows, fetch * partition_count)` and inexact unless an exact input
  proves no rows are dropped. Use `with_per_partition_fetch` for that case.
- `approx_distinct` supports `RunEndEncoded` input, including nested dictionary
  encoding. Spark `shuffle` preserves NULL lists without creating placeholder
  child values. `date_trunc` scalars truncate in their own timestamp unit, like
  arrays, and must not first convert out-of-range values to nanoseconds.

## SQLLogicTests

- `# configMatrix: <key>=<v1>,<v2>` runs an SLT once per value combination;
  repeated keys merge values and distinct keys form a Cartesian product. It
  applies settings like `SET`; default and Substrait runners support it, Postgres
  ignores it, and `--complete` rejects files declaring it.

## Benchmarks

- When using `BenchmarkRun`, call
  `set_memory_pool(&ctx.runtime_env().memory_pool)` after building each runtime.
  Cases then emit `pool_peak_bytes` when a memory limit installs the recording
  pool.
