# Extra Agent Guidelines for Apache DataFusion

## Developer Documentation

- [AI Policy](docs/source/contributor-guide/ai-policy.md)
- [Security Policy](SECURITY.md)

## Before Starting Work

Review any existing open PR for the issue before opening another; identify it
and explain why you are not contributing to it in your PR description.

## AI-Assisted Work

Follow the [AI Policy](docs/source/contributor-guide/ai-policy.md). Authors must
personally read the entire PR, understand its core ideas end-to-end, and disclose
unknowns and assumptions. Review comments and replies must be reviewed by the
human posting them and expressed in their own words; do not post raw agent output.
Follow ASF Generative Tooling Guidance, including disclosure of third-party code.

For suspected vulnerabilities, follow [SECURITY.md](SECURITY.md); do not file a
public issue. Ordinary crashes, hangs, and resource exhaustion are bugs unless
exploitable for arbitrary code execution or process-memory information disclosure.

## Workflow / CI

- Optional Nix setup uses `nix develop` with `flake.nix` for pkg-config, OpenSSL,
  and protobuf. `.envrc` loads the flake only when Nix is available, then loads
  ignored `.envrc.local` overrides; keep machine-specific settings there.
- Docs deployment uses `rsync --delete` but excludes `/versions/`; preserve
  published versioned docs when changing `.github/workflows/docs.yaml`.
  Root docs track `main`; `docs_versions.yaml` builds release tags listed in
  `docs/source/_static/versions.json` (picker) and `old_versions.json` (older
  releases) with `docs/scripts/build_versions.sh` and main's
  `docs/scripts/release/conf.py` overlay. To publish a release, add its manifest
  entry and move `preferred`; retain older releases in `old_versions.json` to
  preserve links. CI mirrors both manifests to `asf-site/versions/` with
  `rsync --delete`, removing releases absent from both.
  Both publishing workflows share non-cancelling `publish-asf-site` concurrency
  and `.github/actions/setup-docs-builder`. Preview over HTTP before merging;
  see `dev/release/README.md`.
- Release cleanup keeps the latest major release and subsequent releases in
  that major; inspect generated SVN deletion commands before running them.
  See `dev/release/README.md`.

- `main` uses the `.asf.yaml` merge queue ruleset: squash merges, `ALLGREEN`,
  up to five concurrent builds, one entry per merge, and a 360-minute required
  check timeout. Release branches retain ordinary branch protection.
- `merge_queue_cleanup.yml` scans on merge groups, every 15 minutes, and manual
  dispatch using trusted default-branch code. Cancellation requires
  `MERGE_QUEUE_CLEANUP_ENABLED=true`; manual runs default to dry-run. It cancels
  obsolete merge-group runs, not runs merely failing checks; active queue refs,
  commits already in the target branch, and reruns are protected.
- Migrated Rust CI steps are defined in `xtask/src/ci_steps.rs` and invoked by
  Actions and locally through `cargo xtask ci step <step> <target>`. Use
  `--explain` to inspect the exact command and prerequisites; update `xtask`
  rather than duplicating those CI commands in workflows. Shared execution,
  environment resolution, and prerequisite checks live in `xtask/src/command.rs`.
  `test cli` supplies its storage-integration environment itself and needs Docker.
- `test workspace` covers ordinary tests; `test extended` selects the
  `extended_tests` target. `test fuzz` selects the separate `fuzz` target in
  unpublished `datafusion-fuzz`. Both extended steps select `datafusion` and
  `datafusion-fuzz` with matching features to reuse compiled dependencies.
  Gate complete slow-test targets with `required-features = ["extended_tests"]`,
  never individual tests with `#[cfg(feature = "extended_tests")]`; `xtask`
  enforces this. `extended.yml` also runs `test fuzz-legacy`: routed hash lookup
  and key-range bitmap union still run 10,000 iterations each.
- Large fuzzing unit tests needing a driver and over 10 seconds implement
  `test_utils::fuzz::FuzzTest` (`test-utils/src/fuzz.rs`); register suites in
  `datafusion/fuzz/tests/fuzz.rs`. Small bounded fuzzers may stay ordinary unit
  tests. Separate deterministic dataset preparation from seeded cases; replay
  with `test.reproduce(dataset_seed, case_seed).await` using the same fixed test
  configuration. `RunConfig` requires a positive case limit or a time budget;
  time limits are soft, include preparation, and never cancel active work.
  The first complete case always runs. The migrated spill-pool suite currently
  uses seed 0 and 1,000 cases. See the contributor testing guide.
- `ci/scripts/rust_clippy.sh` runs both the standard
  `avro,integration-tests,extended_tests` feature set and `--all-features` with
  warnings denied. All-features builds need CMake for `snmalloc-sys`.
  `.github/actions/setup-rust-runtime` disables incremental compilation through
  `CARGO_INCREMENTAL=0`, not `-C incremental=false`.
- Semver CI compares against `HEAD^1` of GitHub's test merge commit, not the
  current base-branch tip; checkout depth 2 supplies that baseline.
  `ci/scripts/changed_crates.sh` shares dependency builds in the workspace
  target directory's `semver-checks-shared` subdirectory.
- `draft_inactive_prs.yml` runs daily on upstream and supports manual dry runs
  (default). After 14 idle days, open non-draft PRs become drafts only for
  conflicts, failed/timed-out CI (or failure/error commit statuses), or a
  changes-requested review on the current head. Merge-queue PRs are excluded;
  cancelled, pending, skipped, or missing checks do not qualify. The helper
  paginates checks/reviews and rechecks head/update identity before mutation;
  run only trusted default-branch code. Address the notification's reasons,
  then mark Ready for review; its comment restarts the stale-PR timer.
- The workspace MSRV is Rust `1.95.0`; the pinned default toolchain is
  Rust `1.99.1`. Keep workflow/toolchain/docs changes aligned with
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

- The main Rust CI test job records coverage with `cargo llvm-cov`:
  `test workspace` writes `target/codecov.json` (`all-tests`), and
  `test slt-coverage` writes `target/codecov-slt.json` (`sqllogictests`).
  `.github/codecov.yml` keeps reports separate without carryforward; SLT statuses
  are informational. `check_asf_yaml_status_checks.py` validates
  required ASF checks; `ci/scripts/check_rust_workflow_skips.py` validates Rust
  skip lists for upstream `main` pushes and fork PRs to `main`, dependencies,
  and Cargo-check artifact guards. Keep these validators aligned with workflows.
- Fork PRs to `main` run Rust CI on the fork except macOS, which runs only
  upstream, including in the merge queue; upstream retains fast lint jobs.
  `fork_ci_status.yml` mirrors fork runs as `Rust (fork)` and reconciles missed
  events with scheduled polling. Its `pull_request_target` path reads metadata
  and writes checks, never executes fork code. The merge queue remains the gate.
- CI workflow regression tests run in `asf-yaml-check` and `rust_lint.sh`.
  Use Python with PyYAML and Node.js 24:

```bash
python3 ci/scripts/test_rust_workflow_skips.py
node --test ci/scripts/test_fork_ci_status.cjs
node --test ci/scripts/test_cancel_merge_queue_runs.cjs
```
- `dev/update_config_docs.sh` and `dev/update_function_docs.sh` accept
  `--output-dir <dir>` (relative to repo root), enabling generated-doc checks
  without replacing the checked-in pages.
- Dev builds use line-table debug info. Use `cargo build --profile dbg` or
  `cargo test --profile dbg` for full debug info and local-variable inspection
  (`Cargo.toml`). Doc-generator binaries require
  `--features docs_generation`; the `dev/update_*_docs.sh` scripts set it.
- `rust_lint.sh` builds the HTML documentation via
  `ci/scripts/check_docs_html.sh`; it fails on Sphinx warnings and needs `uv`,
  `cargo`, `cargo-depgraph`, Graphviz `dot`, and `make` (installs none).
- `datafusion-examples` must depend on the umbrella `datafusion` crate, not
  DataFusion subcrates, except `datafusion-proto` and `datafusion-substrait`.
  `ci/scripts/check_examples_datafusion_crates.py` enforces this.
- Small CI jobs use `ubuntu-slim` only well within one vCPU, 5 GB memory,
  and 15 minutes, without Docker, job containers, or service containers.
  Cargo jobs use `.github/actions/setup-rustup`, which installs the pinned
  toolchain/components and configures the Rust runtime environment.
- Circular dependency checking lives in `.github/workflows/rust.yml`
  (`depcheck`), with the upstream-main/fork-PR skip policy; `dependencies.yml`
  retains unused-dependency checking. `ci/scripts/check_circular_dependencies.sh`
  now runs `cargo publish --workspace --dry-run --no-verify --allow-dirty`;
  `dev/depcheck` is removed. The check reads the crates.io index and packages
  publishable crates without building/uploading, catching cycles (including
  dev-dependencies) and other packaging blockers.
  Semver changed-crate discovery excludes unpublished crates and
  `datafusion-sqllogictest`, even though it is published.
- `cargo xtask ci step test microbenchmarks` runs Criterion bench targets once
  with `--test` under `ci-bench` (CI profile plus `opt-level = 1`, retaining debug
  assertions/overflow checks). It excludes examples, CLI, and the SQL benchmark
  harness and sets `CLICKBENCH_DATA_PATH` to the small checked-in fixture.
  Full execution is local-only (about 10 minutes); no hosted CI job or required
  microbenchmark check remains. It catches panics, not performance regressions.
- Contributors without write access may have at most three open, non-draft PRs.
  Wait for merges or close PRs before opening another.

- License checks use `licenserc.toml` across code, docs, scripts, config files,
  and `.slt.part` fixtures. New covered files need all three ASF SPDX tags;
  run `ci/scripts/license_header.sh`. `.ai/skills/**` is excluded to preserve
  YAML frontmatter; hand-written proto-common/proto-models generated-module
  wrappers remain checked. Follow configured exclusions, not directory-wide
  assumptions about generated code.

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
- The `sort_pushdown` SQL suite runs 30 cases across `baseline`, `sorted`,
  `inexact`, `inexact_unsorted`, and `inexact_overlap`. From `benchmarks/`,
  prepare data with `./bench.sh data sort_pushdown` and
  `./bench.sh data sort_pushdown_inexact` (PyArrow needed for overlap data).
  The suite does not generate data; `--path` names the datasets' parent directory.
- Benchmark session settings use `SessionConfig::from_env()` and standard
  `DATAFUSION_*` variables; only explicitly supplied CLI options override them.
  Old `PARTITIONS`, `BATCH_SIZE`, `SORT_SPILL_RESERVATION_BYTES`,
  `PREFER_HASH_JOIN`, and `HASH_JOIN_BUFFERING_CAPACITY` aliases are ignored.
  See `benchmarks/sql_benchmarks/README.md` for supported settings.
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
  companion `try_from_proto` helpers in `datafusion/physical-plan/src/proto/`.
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

- Substrait consumers forward `&SubstraitConsumerContext` through relation and
  expression callbacks and helpers. Derive child contexts for outer schemas
  and lambda parameters; no shared scope push/pop. Parents and siblings stay
  unchanged on errors or cancellation. Direct top-level helper calls start with
  a default context; nested calls keep the current context. See
  `substrait/src/logical_plan/consumer/substrait_consumer.rs` and the 57 guide.
- `GroupingEquivalenceClass` describes complete grouping tuples contiguous over
  each partition's entire stream, not just a batch. Tuple expression order is
  irrelevant; `[a, b]` does not imply clustering by `[a]`. Preserve these facts
  through projection and union; see `physical-expr/src/equivalence/`.

## API / Contract Changes

- `CREATE FUNCTION` uses `OperateFunctionArg::field` and
  `CreateFunction::return_field` (`FieldRef`), replacing bare data types.
  Preserve extension metadata through SQL planning, factories, and protobuf;
  DESCRIBE and information-schema columns display extension names. See
  `expr/src/logical_plan/{ddl,information_schema}.rs`.
- Implicit struct coercion requires matching field-name sets recursively,
  allowing field reordering. Different names fail planning; list-item/map-entry
  labels remain positional. UNION builders infer common types before analysis;
  coercion casts preserve column qualifiers. See
  `expr-common/src/type_coercion/binary.rs`, `expr/src/type_coercion/other.rs`,
  and `expr/src/expr_rewriter/mod.rs`.
- Null-aware `SortMergeJoinExec` supports only uncorrelated one-key `LeftAnti`,
  no filter, `NullEqualsNothing`, NULLS FIRST, and single-partition inputs.
  Preserve validation/distribution constraints and protobuf `null_aware`;
  tuple/correlated cases need a supported alternative. See
  `physical-plan/src/joins/sort_merge_join/exec.rs`.
- Opt-in `optimizer.enable_row_number_to_aggregate` (default false) rewrites
  partitioned `row_number` top-1 filters to ordered `first_value` aggregation;
  tied rows may differ while remaining valid. Opt-in
  `optimizer.single_distinct_prefer_native` (default false) requires one target
  partition and specialized support for every DISTINCT aggregate. Native
  per-group distinct sets must fit memory; enabling it can fail queries the
  default two-phase rewrite can spill. See
  `optimizer/src/{replace_filter_top1,single_distinct_to_groupby}.rs`.

- `ExprPlanner::plan_scalar` receives `&DFSchema`; custom planners and their
  forwarding wrappers must accept and pass the input schema.
- Aggregate grouping uses `GroupClusteringMode::{None, Partial, Full}` and
  `AggregateExec::group_clustering_mode()`, replacing aggregate input-order
  modes. Clustering describes when groups can be emitted, not sort direction.
- Functional dependency helpers take input field indices, not display names.
  Use `passthrough_field_index` for column references and `None` for computed
  expressions; a cast named like a key does not inherit that key's dependencies.
- `CastExpr::cast_options()` returns a borrowing value. Child-only rewrites use
  `with_new_children` to preserve options and type-only vs explicit-field targets;
  use `with_new_expr_and_target_field` when replacing an explicit target too.

- `OptimizerRule::supports_rewrite` is removed; implement `rewrite` directly.
  `TypeSignature::get_example_types` is deprecated; use
  `get_representative_types` for representative accepted `NativeType` inputs.
- Bundled `operator_statistics` providers and
  `StatisticsRegistry::default_with_builtin_providers` are deprecated. Default
  estimates belong in each operator's `statistics_from_inputs`; the registry
  remains the extension point for user-defined providers.
- Tuple `IN`/`NOT IN` coercion is pairwise and nullability includes all tuple
  elements and subquery columns. Null-aware joins record
  `null_aware_value_keys`: leading keys are tuple values, remaining keys are
  correlation scope. Preserve the count and key order through rewrites and
  serde. Missing protobuf counts decode as `1`; older readers must not consume
  multi-column null-aware plans, since they misinterpret value keys as scope.
- `ProjectionExec` serializes explicit field/schema metadata overrides. Such
  overrides prevent projection elimination/merging; serialize the output schema
  when metadata cannot be derived from expressions and input.

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
- Explicit, uncorrelated `AS MATERIALIZED` CTEs use the built-in
  `LogicalPlan::MaterializedCte(MaterializedCtePlan)` variant, not extension
  nodes. `Cte` has body then continuation inputs and outputs the continuation
  schema; `Scan` is an ID-bound leaf. Exhaustive logical-plan matches must
  handle both. `MaterializedCteScan::find_in` also searches subquery expressions.
  Preserve the shared body's schema and optimization fence: per-reference
  filters, limits, and projections cannot enter it. `MaterializedCteExec` runs the body once on demand,
  accounts buffered batches, and spills while preserving per-partition order.
  Default / `NOT MATERIALIZED` CTEs and correlated bodies remain inlined.
- Extension plans can implement `ExtensionPlanFromProto` and encode with
  `ExecutionPlanEncodeCtx::extension_node::<Self>`. Libraries compose a
  shared `ProtoDecoderRegistry` through
  `config.update_extension(MyExec::register_proto)?` (import
  `ExtensionPlanFromProto`);
  libraries add decoders there instead of replacing it with `with_extension`.
  Updates create a default extension if absent and clone shared extensions before
  mutation, preserving previously cloned configs. Namespace names
  by owning crate; registration is idempotent for the same type and rejects
  different-type collisions within a kind. Unnamed/unregistered plans use the
  codec fallback; errors from a matched decoder are fatal. The registry lives
  in `proto-models`, its plan facade in `physical-plan/src/proto/registry.rs`.
  Extension expressions implement `ExtensionExprFromProto`, encode with
  `extension_expr_node::<Self>`, and register through `MyExpr::register_proto`
  from `ExtensionExprFromProto` in `datafusion-physical-expr/src/proto.rs`.
  The free `register_execution_plan` / `register_physical_expr` functions are
  removed. Plans and expressions share the registry with kind-scoped names and the same fallback/error policy; UDF
  payloads still need codecs. `plan_name` and `expr_name` are binary additive
  but unknown to older JSON readers.
- `MERGE INTO` dispatches to `TableProvider::merge_into`. Providers receive the
  source execution plan, target-then-source qualified `DFSchema`, `ON`, and
  `WHEN` clauses; the default reports unsupported. `MergeIntoOp` is
  non-exhaustive and uses `new(target_qualifier, on, clauses)`; qualify target
  fields by the SQL-visible alias, not the provider identity.
- Physical optimizer rules share one `StatisticsContext` per optimization run.
  Custom rules use `PhysicalOptimizerContext::compute_statistics`; traversals use
  `datafusion_session::with_statistics_context`. Build custom shared contexts
  from the session's statistics registry. The cache retains plan nodes, so
  rewrites need no reset for correctness; reset at lifecycle boundaries to bound
  memory. Provider `Exact` values must be guaranteed: aggregate-statistics and
  limit rewrites can replace execution based on them. Use
  `pushdown_limit_helper_with_stats`; `pushdown_limit_helper` and
  `ensure_distribution` are deprecated.
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

- `DataFrame::select` inserts a global aggregate when aggregate expressions
  remain after columnization and there are no windows, wildcards, or input
  columns outside aggregates. Scalar wrappers/literals are allowed; existing
  aggregate outputs stay projections. Mixed selections do not imply GROUP BY,
  and duplicate output names remain errors. See `core/src/dataframe/mod.rs`.
- Filter pushdown must not cross pure leaf-extraction projections, which would
  cause oscillation with `ExtractLeafExpressions`. `PushDownLeafProjections`
  yields to `OptimizeProjections`: do not split a projection unless the
  extraction projection can move below its input. Window filters may move only
  when non-volatile and structurally dependent solely on every window's partition
  keys; subqueries and outer references block movement.
- The hash-join integer prefilter and its `enable_join_integer_prefilter` /
  `join_integer_prefilter_*` configuration options are removed.
- Parquet `enable_rle_to_dictionary` defaults false and promotes eligible
  top-level string/binary dictionary-page columns only during schema inference,
  not for user-supplied schemas. Schema unification preserves value-type families
  and never narrows dictionary keys; mixed plain/narrow-key files need at least
  `Int32` keys. Preserve field/schema metadata during coercion.
- Parquet `write_row_group_number_distinct_values` defaults false; enabling it
  writes per-column row-group NDV at the cost of hashing every non-null value.
- `date_bin` scalar/array paths preserve the source unit and use wide arithmetic
  when nanosecond intermediates overflow; out-of-range results become NULL.
  All strides, including month strides, must be strictly positive; reject zero
  and negative strides.
- SQL/DataFrame queries are executable code; files, serialized plans, and
  extension implementations are trusted inputs under `SECURITY.md`. Embedding
  applications own validation/sandboxing. `SQLOptions` is not a sandbox and the
  query memory pool is not a hard limit on total process memory.

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
  limits may need raising after the sqlparser upgrade. CLI statement execution
  and interactive validation both honor the configured limit.
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

- Arrow IPC writers accept compression (`uncompressed`, `lz4_frame`, `zstd`),
  optional ZSTD-only compression level, and alignment (`8`, `16`, `32`, `64`).
  Defaults are LZ4_FRAME and alignment `64`; unsupported options fail.
- Parquet `coerce_int96` is `Option<DFTimeUnit>`, accepting `ns`, `us`, `ms`,
  or `s` case-insensitively. `coerce_int96_tz` is `Option<ConfigTimeZone>` and
  validates even when INT96 coercion is disabled. Writer `encoding` is
  `Option<DFParquetEncoding>` with case-insensitive parsing. Invalid settings
  fail at configuration time without mutating the prior value.
- `AT TIME ZONE` uses `timezone(zone, expr)`: naive timestamps become aware
  instants in the requested zone; aware timestamps become naive wall-clock
  values there. `ExprPlanner::plan_at_time_zone` receives `[zone, expr]` and
  must defer type-dependent decisions until after coercion. Timestamp literal
  unparsing must preserve the zone, not just the displayed wall-clock value.

## Optimizer / Execution Contracts

- Implement `ExecutionPlan::limit_behavior`, not deprecated
  `supports_limit_pushdown`: distinguish stopping, copying early-stop hints,
  moving exact row-preserving limits, merging partitions, and window lookahead.
  Copying retains the enforcing limit above the node; partition-reducing nodes
  must not turn a global bound into independent per-input limits.
  Limit/window rewrites share `physical-optimizer/src/row_bound.rs`; use the
  `LimitPushdown` rule, not deprecated one-node `pushdown_limit_helper` /
  `GlobalRequirements`.
- `LogicalPlan::map_expressions` re-derives changed projection schemas. If new
  expressions reference a replacement input, rewrite them directly and construct
  the projection after installing that input, not against the old schema.
- Preserve volatile alias evaluation through projection merging and HAVING /
  QUALIFY: predicates using such aliases run after projection, before DISTINCT.
  DISTINCT ON and deferred predicates with fresh aggregate/window expressions
  or non-selected columns are rejected; use a subquery. COALESCE must not
  duplicate volatile non-final arguments in CASE expansion; its fallback
  evaluates each argument once. See `volatile_evaluations.slt`.
- `common::metadata::coerced_fields_metadata` merges result-source metadata for
  COALESCE / NVL2, ignoring untyped NULL fields and preferring earlier
  keys. Different extension names error; extension parameters stay paired with
  the first supplying name. Parquet `skip_metadata` retains Arrow extension
  keys. Literal substitution needing field metadata uses
  `replace_columns_with_literals_keeping_metadata`.
- Aggregate nullability belongs in `AggregateUDFImpl::return_field`;
  `is_nullable` is deprecated, but keep existing overrides consistent while
  legacy callers remain.

- Float comparison/hash/grouping boundaries use
  `common::float_semantics`: equality normalizes signed zero (including nested
  leaves), hashing uses `hash_input`, and normalization leaves NaN payload bits
  unchanged. Ordering still uses IEEE totalOrder (`-0.0` before `+0.0`); keep
  grouped MIN/MAX extremes
  aligned with that order.
- Cast predicate preimages belong in `expr-common/src/casts.rs`; logical
  rewrites live in `optimizer/src/simplify_expressions/cast_preimage.rs`.
  Many-to-one casts need source-domain ranges, not same-operator unwrapping;
  IN lists accept only exact singleton preimages. Do not duplicate volatile
  inputs; preserve source timestamp unit, timezone, and dictionary wrapping.
- `Partitioning::keeps_keys_local` proves grouping-key locality, not cross-input
  co-partitioning. Range analysis requires disjoint images at concrete splits;
  monotonicity alone is insufficient. UDF/type combinations opt in through
  `supports_range_key_locality_analysis` only after audit for same-direction
  ordering and exact nullness preservation for the types and literal arguments.
  Forward its per-argument literal values (`None` means non-literal, not NULL);
  default/FFI behavior denies it.
- Retained batches that can be released independently use
  `RecordBatchMemoryTracker`: shared buffers are charged once by capacity and
  released at the last reference. Uncount before dropping buffers (addresses
  identify them); pair array-overhead count/uncount methods consistently.
  `RecordBatchMemoryCounter` remains for append-only accounting.
- `PartitionEvaluator::evaluate_ranges` batches one frame range per output row
  for frame-using window functions; default delegates to `evaluate`. Overrides
  must preserve output cardinality/order and evaluator memoization state.
- GROUP BY dependency helpers require `is_valid_across_nulls`: nullable UNIQUE
  determinants do not justify adding/removing grouping expressions, since
  GROUP BY puts NULL keys together. See `common/src/functional_dependencies.rs`.

- Physical optimization is an ordered single pass. Rules preserve results and
  plan validity; `OutputRequirements` and `EnsureRequirements` have
  compatible-order exceptions. `SanityCheckPlan` is deprecated and removed from
  default rules: the physical planner always checks the final optimized plan,
  including custom pipelines, and diagnoses the rule breaking requirements.
  Standalone optimizer callers use `check_plan_tree_sanity`. Prefer default
  order plus configuration toggles; custom pipelines need end-to-end result and `EXPLAIN`
  tests. Optimization coverage is not guaranteed for noncanonical shapes.
- `EliminateConstantJoin` runs after `AggregateStatistics`, before
  `EnsureRequirements`. It replaces inner nested-loop/cross joins against a
  provable single literal row with filter/projection, preserving output metadata;
  multi-partition placeholders and nested-loop lambda filters are not eligible.
- Sort pushdown must not change rows retained by limits. Across a local/global
  limit, only refine a directly underlying compatible sort; do not introduce
  sorting below unordered fetches. Limit `required_ordering` accessors are
  deprecated/no-op; protobuf fields are ignored on read and no longer written.
- Scans advertising output ordering preserve file order even without explicit
  `preserve_order`; disable sibling work stealing and limit file pruning there.
- Leaf extraction preserves aliases and aligns UNION inputs by position. Join
  projection pushdown must distinguish lambda-local indices from input columns
  and rebase lambda parameters when the join-filter input width changes.
- UNNEST preserves list-element field metadata in expression inference and plan
  schemas. Remap functional-dependency indices after struct expansion; expanded
  lists invalidate single-row dependencies and are not determined by input keys.
- `LazyBatchGenerator::try_to_proto` opts into built-in generate-series serde
  only when reset-state rows, batch boundaries, and schema match. Encode initial
  arguments, never the current cursor. `LazyMemoryExec` supports one generator;
  unsupported generators/multiple generators need a plan codec; errors are fatal.
  Series conversions live in `functions-table/src/generate_series/proto.rs`
  behind the `proto` feature.
- Preserved Hive file partitions use contiguous key ranges, not hash buckets.
  `FileGroup::group_by_partition_values_with_split_points` sorts with default
  sort/null options and returns routing boundaries; `ListingTable` declares
  matching `Partitioning::Range` unless statistics regrouping invalidates it.
  Use `output_range_partitioning_from_split_points`; the old hash-producing
  `output_partitioning_from_partition_fields` is deprecated. Preserve split
  points, projection mapping, and file-group count through serde.
- Window decorrelation may extend every window's PARTITION BY with correlated
  columns only for non-volatile column equalities with no float leaf and only
  identity-preserving casts (`common::nested_struct::is_injective_cast`). A
  pulled-up scalar aggregate blocks this extension; preserve window output names.
  See `optimizer/src/decorrelate.rs`.
- Positive struct/tuple IN lists can supply necessary per-column pruning
  guarantees via an immutable `struct_field_mapping` accessor. Keep the original
  predicate for tuple correlation; do not derive these guarantees for NOT IN,
  or for a field with a NULL member. Duplicate `named_struct` names disable its
  mapping. See `physical-expr/src/utils/guarantee.rs`.
- Join estimates include unmatched outer rows and rescale null counts for
  repeated/dropped rows, NULL padding, and `NullEquality`. Row-count-refining
  statistics providers rescale null counts too; estimates must not become exact
  merely because input counts were exact. See `physical-plan/src/joins/utils.rs`
  and `physical-plan/src/operator_statistics/mod.rs`.
- Hash-join build allocation failures retain all collected rows for fallback.
  Filter coalescing compacts external `Utf8View` / `BinaryView` buffers rather
  than retaining whole input buffers for sparse selected rows.

## Recent Runtime / Integration Contracts

- Logical filter pushdown keys replacements by qualified `Column`, not flat
  names: qualified `t.a` and an unqualified field literally named `t.a` differ.
  DataFrame fill methods likewise preserve identifiers rather than parsing names.
  See `optimizer/src/push_down_filter.rs` and `core/src/dataframe/mod.rs`.
- `EliminateFilter` may drop NULL members from positive IN lists and reject
  NOT IN lists containing NULL only under filter semantics; general expressions
  must retain three-valued results. See `optimizer/src/eliminate_filter.rs`.
- `SingleDistinctToGroupBy` casts outer projection expressions back to original
  declared field types when partial/final aggregation changes them, preserving
  decimal precision/scale and output schema. See
  `optimizer/src/single_distinct_to_groupby.rs`.
- Generic custom-node sort pushdown probes rebuilt children with recomputed
  properties; prove input ordering, distribution/co-partitioning, and output
  ordering. `maintains_input_order` alone is insufficient. Fetch boundaries
  only refine directly underlying sorts while preserving all selecting keys;
  never add blocking sorts to unordered unbounded inputs. See
  `physical-optimizer/src/ensure_requirements/enforce_sorting/sort_pushdown.rs`.
- Partial aggregation must not advertise unique grouping keys. Fully clustered
  partial/final/single streams release table capacity after emitting completed
  groups under memory pressure; final/single without spill may emit completed
  groups before failing, but must fail when none can complete and memory is
  exhausted. See `physical-plan/src/aggregates/`.
- `MaterializedCteScanExec` carries all-partition body statistics supplied by the
  planner; multi-partition per-partition requests stay unknown. Preserve the
  shared body's optimization fence. See `physical-plan/src/materialized_cte.rs`.
- `DynamicFilterPhysicalExpr::update_from` requires matching expression IDs;
  self/shared state is a no-op and completed targets ignore compatible updates.
  Copy expression/completion state, increment the target generation, invalidate
  cache, and notify waiters. See `physical-expr/src/expressions/dynamic_filters/`.
- Spark `from_utc_timestamp` uses Spark-compatible timezone parsing in
  `spark/src/function/datetime/timezone.rs`. Conditional timezone evaluation
  requires explicit `SparkFromUtcTimestampExpr` construction: computed zones
  evaluate only for non-null timestamps; ordinary UDF arguments remain eager.
  Preserve documented Chrono-range and post-2099 region-rule limitations.

- `datafusion.catalog.partition_information_schema` (default `false`) replaces
  `system_catalog`: when enabled, information-schema views are catalog-local;
  otherwise they describe all catalogs. There is no named virtual system catalog.
  `SHOW` / `DESCRIBE` use `LogicalPlan::Show` / `ShowKind` instead of
  `DescribeTable` or SQL expansion; SHOW TABLES remains cross-catalog.
  Shared batch/schema builders live in `expr/src/logical_plan/information_schema.rs`;
  catalog providers and the physical planner consume them.
- CSV execution honors `CsvOptions::null_regex` for string and numeric fields;
  compile it when constructing the reader and report invalid patterns as
  execution errors. See `datasource-csv/src/source.rs`.
- Ordered file scans may rebalance files by size only when the candidate's
  validated output ordering equals the original. Declared `output_partitioning`
  still forbids file repartitioning; preserve partition-to-file-group mapping.
- `FFI_AggregateUDF::distinct_handling` now forwards through
  `FFI_DistinctHandling`; unknown future native variants fall back to Sensitive.
  The new function-pointer field changes the ABI layout; rebuild consumers and
  providers together, and do not backport the layout change to patch branches.

- `datafusion-ffi` is an rlib; the unpublished
  `datafusion-ffi-integration-tests` crate owns the test cdylib. Cross-library
  cases live in `datafusion/ffi/integration-tests/tests/cases/`, registered in
  `cases/mod.rs`; constructors/exports remain in `datafusion/ffi/src/tests/`.
  Run `cargo test -p datafusion-ffi -p datafusion-ffi-integration-tests --features integration-tests`
  for ABI changes (also covered by `cargo xtask ci step test ffi`).
- `SQLOptions::with_allow_copy(false)` rejects COPY independently of other DML;
  disabling DML still rejects COPY. Both default to allowed; these controls
  are not a sandbox.
- Parquet multi-file inference widens nullability for fields missing from some
  containing schemas, including nested structs through List/LargeList. A missing
  parent does not widen its required children or list-element nullability;
  preserve field and schema metadata. Other nested types retain Arrow's exact
  type-equality requirements. See `datasource-parquet/src/file_format.rs`.
- File-ordering bounds include possible NULLs according to direction and
  `nulls_first`; non-null min/max alone cannot prove file groups ordered.
  Exact null counts override nullable-field uncertainty; all-null files need
  NULL bounds. See `datasource/src/statistics.rs`.
- `percentile_cont` and `approx_percentile_cont` accept column-referenced
  percentiles, resolved from non-null Float32/Float64 values and constant per
  aggregation across batches/merged states. Reject differing or out-of-range
  values; preserve the percentile in accumulator state. Descending ordering
  applies `1-p`; empty inputs return NULL. Run the aggregate `state_compat` test.
- Fully clustered multi-key grouping supports temporal, interval, decimal,
  and nonnegative-width fixed-size binary keys in addition to prior scalar
  keys. Floats, nested/encoded types, and invalid time units remain unsupported;
  retain hash-grouping parity across batches, NULLs, emission, and clear.
- Correlated aggregate decorrelation groups by whole subquery-side join
  expressions (e.g. casts), not merely their referenced columns. EXISTS count
  compensation must model matched groups failing HAVING and unmatched default
  aggregate values; fall back to uncompensated decorrelation when HAVING cannot
  be safely pulled up, filters read aggregate outputs, or keys are not bare columns.
- `EliminateGroupByConstant` rebuilds projections from retained aggregate output
  positions, preserving aliases/qualifiers rather than re-planning input
  expressions. Never remove every grouping key: empty-input semantics differ.
- Spark `stack(n, ...)` returns a non-null List of nullable structs with nullable
  `col0`, `col1`, ... fields; callers unnest it to produce rows. Require a positive
  INT literal `n`, enforce per-column Spark type compatibility, pad missing cells
  with NULLs, and cap generated rows at `1 << 24` per batch.

## SQLLogicTests

- `ARRAY_AGG(DISTINCT x)` has a native groups accumulator sharing encoded
  values across groups. Ordered DISTINCT accepts only ORDER BY the argument;
  reversed execution restores requested sort/null options. Without ORDER BY,
  compare list results as multisets in `state_compat`, not by element order.
  Preserve FILTER/IGNORE NULLS and cross-accumulator state compatibility.
- Materialized CTE behavior lives in `cte_materialized.slt`; accumulator state
  changes also run `cargo test -p datafusion-functions-aggregate --test state_compat`.
  Accumulator, groups-accumulator, and `convert_to_state` outputs must be mutually
  mergeable; extend the compatibility harness's cases/exclusions when needed.

- Substrait `--substrait-optimize` requires `--substrait-round-trip` and optimizes
  before serialization, covering optimizer-produced shapes. CI runs
  `cargo xtask ci step test substrait` (`limit.slt`) and
  `cargo xtask ci step test substrait-optimized` (`joins_conditionless.slt`);
  the full Substrait SLT suite is not yet supported.
- `# configMatrix: <key>=<v1>,<v2>` runs an SLT once per value combination;
  repeated keys merge values and distinct keys form a Cartesian product. It
  applies settings like `SET`; default and Substrait runners support it, Postgres
  ignores it, and `--complete` rejects files declaring it.

## Benchmarks

- `--criterion-namespace` / `CRITERION_NAMESPACE` implies Criterion mode;
  simple-only options still conflict. SQL benchmark `--debug` prints logical,
  optimized, physical plans and post-execution metrics on the first successful
  run, including preparation/warm-up; it affects that run's timing.
- `--simulate-latency` now models remote reads with per-request latency,
  per-connection bandwidth, bounded connections, and paginated LIST, not just
  sleeps. Writes remain unthrottled. The `pwmj` SQL suite covers classic
  piecewise merge joins; ClickBench query parity is checked by
  `benchmarks/tests/clickbench_parity.rs`.
- When using `BenchmarkRun`, call
  `set_memory_pool(&ctx.runtime_env().memory_pool)` after building each runtime.
  Cases then emit `pool_peak_bytes` when a memory limit installs the recording
  pool.
