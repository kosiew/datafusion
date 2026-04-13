# Repository Guidelines for Codex Agents

This repository uses Rust and contains documentation in Markdown and TOML configuration files. Our focus is on delivering robust, maintainable, and high-quality solutions. Please follow these guidelines when contributing.

## Engineering principles
- The code works, and we know it works through tests and validation.
- Solve the right problem first; avoid premature optimization or unnecessary feature creep.
- Handle error cases gracefully and predictably, with actionable error messages.
- Keep implementation simple and minimal; do only what is needed.
- Code should be well-covered by tests as regression protection.
- Practice red/green TDD: fail fast, implement incrementally, and refactor with confidence.
- Document behavior and update docs when behavior changes.
- Design changes to afford future evolution without complexity bloat.
- Maintain relevant non-functional qualities: reliability, maintainability, security, observability.

---

## Toolchain & Workspace Notes

* The workspace pins the Rust toolchain to `1.93.0` via `rust-toolchain.toml` while declaring an MSRV of `1.88.0` in `Cargo.toml`. Use `rustup override set 1.93.0` if you are not already using the bundled toolchain.
* Prefer crate-scoped builds/tests first for faster feedback (`cargo build -p <crate>`, `cargo test -p <crate>`), then expand to workspace-wide checks when changes cross crate boundaries.
* Python tooling now uses a `uv` workspace (`pyproject.toml`, `uv.lock`) for `docs`, `dev`, and `benchmarks`. Use `uv sync` to install/sync Python dependencies.
* CI validates several feature combinations (`--no-default-features` plus feature toggles) across major crates. Preserve feature-gated builds when changing Cargo features.
* Substrait coverage is now feature-gated in `datafusion-sqllogictest`; enable `--features substrait` when running Substrait round-trip tests or matching CI's broader sqllogictest coverage.
* The `ci` and `ci-optimized` Cargo profiles both have `debug-assertions = true`; code paths gated by `debug_assert!` are active in CI even for release-profile builds, so invariants that would be compiled away locally are enforced in CI.
* Recent engine work is concentrated in `datafusion/physical-plan`, `datafusion/functions*`, `datafusion/sqllogictest`, `datafusion/spark`, and `datafusion/ffi`.
* DataFusion is organised as a large Cargo workspace. Key crates and directories include:

  * `datafusion/core`, `datafusion/execution`, and `datafusion/sql` for the core logical planning, runtime configuration, and SQL planning layers.
  * `datafusion/optimizer`, `datafusion/physical-optimizer`, `datafusion/pruning`, and `datafusion/physical-plan` for logical/physical rewrite rules, partition pruning, execution plans, and operator implementations (including the metrics subsystem under `physical-plan/src/metrics`).
  * `datafusion/common`, `datafusion/common-runtime`, and `datafusion/macros` for shared types, error handling, async helpers (`JoinSet`, `SpawnedTask`), tracing hooks, and macro utilities.
  * `datafusion/catalog` plus `datafusion/catalog-listing` for catalog APIs and listing-table helpers.
  * `datafusion/datasource` and the `datafusion/datasource-*` crates for shared datasource utilities and connectors (Avro, CSV, JSON, Parquet).
  * `datafusion/physical-expr`, `datafusion/physical-expr-common`, and `datafusion/physical-expr-adapter` for physical expression evaluation, property tracking, and schema rewriting.
  * `datafusion/functions-*` crates covering scalar, aggregate, window, nested, and table functions together with shared utilities.
  * `datafusion/proto`, `datafusion/proto-common`, and `datafusion/substrait` for protocol buffers, Flight SQL/CLI integration, and Substrait representations, alongside `datafusion/doc` for developer-focused documentation builds.
  * `datafusion-cli`, `datafusion-examples`, `benchmarks`, and `test-utils` for the CLI binary, runnable examples, benchmarking harnesses, and reusable test helpers.

## Quick Policy Checklist

- Use crate-scoped feedback first (`cargo build -p <crate>`, `cargo test -p <crate>`), then expand scope when needed (see [Toolchain & Workspace Notes](#toolchain--workspace-notes)).
- For major changes, run `./dev/rust_lint.sh` before proposing updates (see [Optional Tooling Checks](#6-optional-tooling-checks)).
- If planner/executor/expr behavior changes, add both unit tests and SQL Logic Tests (see [Testing & Validation](#3-testing--validation)).
- Prefer SQL Logic Tests under `datafusion/sqllogictest/test_files/` over snapshots for SQL/engine tests; if snapshots are necessary, include PR justification (see [Testing & Validation](#3-testing--validation)).
- When SQL setup is insufficient for an SLT case, register Rust-built tables in `datafusion/sqllogictest/src/test_context.rs` and dispatch them by `.slt` file name before the test runs.
- Keep functions focused and modules cohesive; document public APIs with `///` comments and examples (see [Maintainability](#maintainability)).
- Use implementation, documentation, and contextual comments to capture intent and invariants (see [Commenting guidance](#commenting-guidance)).

## 1. Solution-First Approach

* **Understand the Problem:** Before writing code, ensure you clearly understand the requirements and desired outcomes. Ask clarifying questions if anything is ambiguous.
* **Design Thoughtfully:** Sketch out the architecture or data flow. Consider performance, scalability, and readability.
* **Provide Examples:** Include usage examples or code snippets demonstrating how your feature or fix works.

## 2. Code Quality & Best Practices

### Idiomatic Rust

* Prefer `snake_case` for variables, functions, and modules; `CamelCase` for types and enums.
* Use pattern matching, `Option<T>`, and `Result<T, E>` for clear and safe handling of optionality and errors.
* **Think of `Option` as a computation pipeline, not just "value or no value."** Stop treating `Option` merely as presence/absence—view it as a lazy transformation stream. Combinators like `.map()`, `.and_then()`, `.filter()`, and `.ok_or()` turn error-handling into declarative data flow. Once you internalize absence as a first-class transformation, you can write entire algorithms that never mention control flow explicitly—yet remain 100% safe and compiler-analyzable. This paradigm shift transforms imperative conditionals into composable, chainable operations.
* Prefer `Option` combinator pipelines over imperative `if let` extraction chains when transforming optional values.
* Leverage iterators and functional constructs for concise, efficient code.
* **Refactor imperative traversals** (e.g. using `.apply()` with mutable flags) **into declarative expressions** (e.g. using `.exists()`, `.any()`, or `.find()`) wherever possible. This makes the intent clearer, eliminates boilerplate, and improves maintainability.

  > ✅ *Example:* Convert verbose recursion with `mut` flags into `.exists()` + `.unwrap()` patterns for simple boolean checks.
* **Simplify match arms by directly binding variables** (e.g. `LogicalPlan::Foo(x)` instead of `LogicalPlan::Foo(_)` + `if let`). This reduces redundant re-matching and enhances clarity.

  > ✅ *Example:* Collapse a redundant `if let` on a known variant by binding it directly in the `match` arm.
  > Also take the opportunity to **clarify comments or add TODOs**, especially around complex control flows like CTEs or plan rewriting.

#### Ergonomic function signatures (use `Into` / `AsRef` / `IntoIterator`)

When writing or refactoring functions—especially builder setters and public APIs—prefer signatures that are ergonomic for callers while preserving performance and clarity.

Use these patterns deliberately:

* **Own a value:** accept `impl Into<T>` and convert internally (e.g. `host: impl Into<String>`, `path: impl Into<PathBuf>`).
* This lets callers pass `String`/`&str`, `PathBuf`/`&Path`, etc., without manual `.to_string()` / `.into()`.
* **Borrow read-only inputs:** accept `impl AsRef<str>` / `impl AsRef<Path>` when you do not need ownership to avoid allocations.
* **Optional inputs:** accept `impl Into<Option<T>>` for ergonomic option-like parameters (`x`, `Some(x)`, or `None`).
* **Collections / iterables:** accept `impl IntoIterator<Item = T>` (or `Item = impl Into<T>` when appropriate) to support `Vec`, sets, arrays, and iterators.

Guidance:
* Prefer `AsRef` for hot paths where allocation would be wasteful; prefer `Into` when the function stores/owns the value.
* Do not use generic conversion bounds when they reduce clarity or introduce ambiguity—favor explicit types where precision matters.

### Rust mental models — think in types and proofs

The following mental models help write safer, clearer, and more idiomatic Rust. These are conceptual shifts — small syntax changes but large design wins.

- 🔥 Ownership → Compile-Time Resource Graph

  > Stop seeing ownership as “who frees memory.”
  > See it as a **compile-time dataflow graph of resource control**.

  Every `let`, `move`, or `borrow` defines an edge in a graph the compiler statically verifies — ensuring linear usage of scarce resources (files, sockets, locks) **without a runtime GC**. Once you see lifetimes as edges, not annotations, you’re designing **proofs of safety**, not code that merely compiles.

---

- ⚙️ Borrowing → Capability Leasing

  > Stop thinking of borrowing as “taking a reference.”
  > It’s **temporary permission to mutate or observe**, granted by the compiler’s capability system.

  `&mut` isn’t a pointer — it’s a **lease with exclusive rights**, enforced at compile time. Expert code treats borrows as contracts:

  * If you can shorten them, you increase parallelism.
  * If you lengthen them, you increase safety scope.

---

- 🧩 Traits → Behavioral Algebra

  > Stop viewing traits as “interfaces.”
  > They’re **algebraic building blocks** that define composable laws of behavior.

  A `Trait` isn’t just a promise of methods; it’s a **contract that can be combined, derived, or blanket-implemented**. Once you realize traits form a behavioral lattice, you stop subclassing and start composing — expressing polymorphism as **capabilities, not hierarchies**.

---

- 🧠 `Result` → Explicit Control Flow as Data

  > Stop using `Result` as an error type.
  > It’s **control flow reified as data**.

  The `?` operator turns sequential logic into a **monadic pipeline** — your `Result` chain isn’t linear code; it’s a dependency graph of partial successes. Experts design their APIs so every recoverable branch is an **encoded decision**, not a runtime exception.

---

- 💎 Lifetimes → Static Borrow Slices

  > Stop fearing lifetimes as compiler noise.
  > They’re **proofs of local consistency** — mini type-level theorems.

  Each `'a` parameter expresses that two pieces of data **coexist safely** within a bounded region of time. Experts deliberately model relationships through lifetime parameters to **eliminate entire classes of runtime checks**.

---

- 🪶 Pattern Matching → Declarative Exhaustiveness

  > Stop thinking of `match` as a fancy switch.
  > It’s a **total function over variants**, verified at compile time.

  Once you realize `match` isn’t branching but **structural enumeration**, you start writing exhaustive domain models where every possible state is named, and every transition is **type-checked**.

### Maintainability

* Keep functions focused (ideally under 40 lines) and modules cohesive.
* Name functions and variables descriptively; avoid generic names like `foo` or `temp`.
* Group related types and functions and document public APIs with `///` comments and examples.

### API Design Principles

When designing structs that carry arguments or configuration (e.g., `AccumulatorArgs`, `PartitionEvaluatorArgs`, `ScalarFunctionArgs`), follow these principles:

#### 1. **Clarity and Simplicity Over Cleverness**

Make contracts explicit. Prefer straightforward data fields over conditional logic, runtime synthesis, or smart defaults that require deep understanding to use correctly.

* ✅ **Good:** `pub input_fields: &'a [FieldRef]` — explicitly provides pre-computed field information
* ❌ **Avoid:** Synthesizing data on-demand based on whether other fields are empty, forcing users to understand complex fallback logic

**Why:** Users should not need to understand `Cow` semantics, conditional schema synthesis, or when to use `schema.field()` vs helper methods. Self-documenting APIs reduce cognitive load and onboarding time.

#### 2. **Consistency with the Ecosystem**

Align new APIs with existing patterns in the codebase. If scalar and window functions receive pre-evaluated `FieldRef`s, aggregate functions should follow a similar model wherever feasible.

* ✅ **Good:** Provide `input_fields: &[FieldRef]` alongside `exprs: &[Arc<dyn PhysicalExpr>]` to match the ergonomics of `ScalarFunctionArgs` and `PartitionEvaluatorArgs`
* ❌ **Avoid:** Forcing aggregate UDAFs to call `exprs[i].return_field(schema)?` when other function types receive fields directly

**Why:** Consistency reduces the mental context switch when moving between different parts of the codebase. Developers familiar with one function type can transfer that knowledge directly.

#### 3. **Optimize for Performance Through Pre-Computation**

Pre-compute expensive or frequently accessed information during construction rather than on every access or method call.

* ✅ **Good:** Compute `input_fields` once when building `AggregateFunctionExpr` and store them
* ❌ **Avoid:** Synthesizing schemas or calling `return_field()` multiple times across `create_accumulator()`, `groups_accumulator_supported()`, `create_groups_accumulator()`, etc.

**Why:** Aggregate expressions are constructed once but may create many accumulators (one per group in hash aggregations). Pre-computation amortizes costs and eliminates redundant work.

#### 4. **Mechanical Changes Beat Conditional Logic**

When faced with a choice between:
- (A) Adding a field to a struct and updating all call sites mechanically
- (B) Adding conditional logic that synthesizes or computes values at runtime

Choose (A). Mechanical changes are easier to review, verify, and maintain.

* ✅ **Good:** Add `input_fields` field, update 18 call sites with straightforward additions
* ❌ **Avoid:** Adding `args_schema()` method with `Cow<Schema>` that conditionally synthesizes based on whether the schema is empty

**Why:** Conditional logic introduces edge cases and potential bugs. Mechanical changes can be reviewed quickly and validated with simple grep/IDE searches. The compiler enforces that all call sites are updated.

#### 5. **Future-Proofing Through Explicit Structure**

Design data structures that can accommodate future features without breaking changes.

* ✅ **Good:** Having explicit `input_fields` makes it easy to add per-field caching, lazy evaluation, or metadata decorators
* ❌ **Avoid:** Tightly coupling field access to schema lookup patterns that make future optimizations require API redesign

**Why:** Explicit fields provide clear extension points. Adding features like "cache field metadata" or "lazily compute field nullability" becomes straightforward without changing the public API.

#### 6. **Self-Documenting APIs Over Extensive Documentation**

If your API requires 80+ lines of documentation to explain when and why fields behave differently in different contexts, consider redesigning the API.

* ✅ **Good:** `input_fields` is self-documenting — it contains the fields corresponding to input expressions
* ❌ **Avoid:** Requiring users to understand: "when schema is empty we synthesize from literals, but when non-empty we use it directly, and you can use either `schema.field()` or `input_field()` depending on..."

**Why:** Documentation gets outdated, misread, or overlooked. Self-documenting APIs encode contracts in types that the compiler verifies.

#### Summary Checklist for Argument Structs

When adding or modifying argument structs like `AccumulatorArgs`:

- [ ] Are all frequently accessed fields pre-computed during construction?
- [ ] Is the API consistent with similar structs (`ScalarFunctionArgs`, `PartitionEvaluatorArgs`)?
- [ ] Can users access what they need without calling helper methods or understanding conditional logic?
- [ ] Would the API be clear to someone seeing it for the first time?
- [ ] Does adding this field require mostly mechanical changes to call sites?
- [ ] Will this structure accommodate future features without breaking changes?

### Use `take_function_args` when possible

When implementing or refactoring function/aggregate/window argument handling, look for opportunities to replace manual argument-count checks and iterator extraction with the helper `take_function_args`. This reduces boilerplate and makes the code clearer and less error-prone.

Example: replace this pattern

  -        if args.len() != 3 {
  -            return plan_err!("nvl2 must have exactly three arguments");
  -        }
  -
  -        let mut args = args.into_iter();
  -        let test = args.next().unwrap();
  -        let if_non_null = args.next().unwrap();
  -        let if_null = args.next().unwrap();

with the concise and safer form

  +        let [test, if_non_null, if_null] = take_function_args(self.name(), args)?;

`take_function_args` validates the argument count and returns a fixed-size array for pattern matching. Use it when the arity is known at compile time.

If you answer "no" to multiple questions, consider simplifying the design.

### Error Handling & Context

* Use the `?` operator to propagate errors with context. Avoid silent failures.
* When you encounter `.map_err` converting one error type to another, evaluate whether implementing `From` (and hence `?`) would be cleaner; a dedicated `From` impl often reduces boilerplate and improves composability.
* Implement custom errors via the `thiserror` crate when appropriate.
* Provide clear messages to aid debugging and user feedback.

### Performance Considerations

* Favor zero-copy patterns: use `&str` over `String` and `Arc<T>` for shared data.
* Avoid unnecessary heap allocations; minimize cloning.
* Benchmark critical paths when performance is a concern.
* Optimizations should be focused on bottlenecks — those steps that are repeated millions of times in a query; otherwise, prefer simplicity.
* Pre-compute expensive or frequently reused values during construction instead of recomputing on each call.

* Prefer multiple simple code paths over a single complex adaptive path. Optimize for the common case first and keep that path fast and easy to reason about; handle rare or complex edge cases with separate, well-tested branches or fallbacks. This often yields clearer, faster, and more maintainable code than trying to build one highly adaptive, catch-all implementation.

### Benchmark Pattern

When creating or modifying benchmarks in `benchmarks/`, follow this standard pattern:

- **Setup phase (outside loop):** Create a `SessionContext` once per case using `create_context(...) -> Result<SessionContext>`. This builds the plan once and is excluded from timing.
- **Benchmark loop (timed):** Inside the bench loop, run:
  - `ctx.sql(...)` — parse SQL and create logical plan
  - `df.create_physical_plan()` — convert to physical plan
  - `collect(plan, ctx.task_ctx())` — execute and collect results

The benchmark timing includes **execution**, but the setup phase (context creation and initial planning) happens only once **outside the loop** to measure execution performance accurately.

- `dfbench` now falls back to `DATAFUSION_RUNTIME_MEMORY_LIMIT` when `--memory-limit` is not provided; use this env var for scripted benchmark runs that need runtime memory pressure settings.

### Clone & Copy: Precision over Reflex

* **Treat `Clone` as a precision instrument, not a reflex.** Implement custom `Clone` to achieve amortized O(1) behavior where possible — clone only references or indices, not deep data structures, then perform explicit duplication at controlled boundaries.
* **Reserve `Copy` for plain-old-data only.** Restrict `Copy` to types where move and copy are semantically and computationally indistinguishable: integers, coordinates, short fixed-size math vectors, and similar primitives.
* **Make ownership boundaries visible.** For everything else, use intentional `Clone` APIs so that duplication shows up clearly in code reviews and profiling. Explicit `.clone()` calls document where data is being duplicated and help identify optimization opportunities.
* **Design for shared ownership.** Prefer `Arc<T>` and reference-counted patterns over deep cloning when multiple owners need access to the same data. Clone the `Arc`, not the underlying data.

### Area-specific guidance

#### Functions and Spark compatibility

* Generic SQL functions live in `datafusion/functions*`.
* Spark-compatible behavior lives in `datafusion/spark`.
* New function behavior should include correctness tests, preferably SLT coverage under `datafusion/sqllogictest/test_files/`.
* `as_any()` has been removed from `ScalarUDFImpl`, `AggregateUDFImpl`, `WindowUDFImpl`, `ExecutionPlan`, `TableProvider`, `SchemaProvider`, `CatalogProvider`, and `CatalogProviderList`; do not implement or call it for these traits. Use trait-object helpers (`is::<T>()`, `downcast_ref::<T>()`) for downcasting instead.
* `arrow_try_cast(value, type)` is a new built-in function in `datafusion/functions/src/core/` that casts a value to the target Arrow type and returns `NULL` on failure instead of erroring; prefer it over `arrow_cast` when cast failures should produce `NULL`.
* `cast_to_type(expr, reference)` and `try_cast_to_type(expr, reference)` are built-ins that cast to the runtime type of `reference`; use them when the target type is data-driven rather than fixed in SQL text.
* `split_part(string, delimiter, position)` now supports negative `position` values to split from the right, and matches PostgreSQL's empty-delimiter behavior (`1`/`-1` return the full string, other positions return `''`, and `0` remains an error). Preserve these semantics in planner, function, and SLT coverage.
* Ordered-set aggregates now accept both `WITHIN GROUP (...)` and the inline built-in form `agg(value, percentile ORDER BY value)` for functions such as `percentile_cont` and `quantile_cont`; preserve both planner paths when extending ordered-set aggregate support.

#### Optimizer and expression simplification

* `SimplifyExpressions` now includes `rewrite_multiple_linear_aggregates` to canonicalize repeated linear aggregate patterns (for example `SUM(x + 1), SUM(x + 2)`), enabling common-subexpression elimination opportunities; preserve this behavior when extending aggregate simplification.
* `LogicalPlan::Aggregate::aggr_expr` must remain top-level aggregate expressions (`Expr::AggregateFunction` or aliased aggregate expressions). If simplification yields non-aggregate wrappers, keep the Projection-above-Aggregate rewrite pattern intact.
* `SimplifyContext::with_schema()`, `with_config_options()`, and `with_query_execution_start_time()` are deprecated since 54.0.0; use `SimplifyContext::builder()` (returning a `SimplifyContextBuilder`) to set non-default context values.

#### Physical plan and execution

* Changes in `datafusion/physical-plan` often require planner/optimizer wiring updates in `datafusion/core` and `datafusion/physical-optimizer`.
* `ExecutionPlan::apply_expressions` is required. Custom plan nodes must explicitly visit all owned top-level `PhysicalExpr`s, and multi-expression nodes should use `TreeNodeRecursion::visit_sibling` to preserve short-circuit semantics.
* Unnest now supports `ListView` and `LargeListView` in both logical and physical paths; keep datatype matching and recursion behavior aligned across `datafusion/expr` and `datafusion/physical-plan` when extending unnest support.
* Keep `PhysicalOptimizer::new` rule ordering intentional: `LimitPushPastWindows`, `HashJoinBuffering`, `LimitPushdown`, and `TopKRepartition` have ordering dependencies that affect correctness and performance.
* Physical join selection now supports pluggable cross-operator statistics via `StatisticsRegistry`; wire custom registries through `SessionStateBuilder::with_statistics_registry(...)` and gate registry-based join reordering via `datafusion.optimizer.use_statistics_registry`.
* `PushdownSort` now inserts a bounded `BufferExec` when it eliminates a partition-preserving `SortExec` directly under `SortPreservingMergeExec`; preserve this buffering behavior and the `datafusion.execution.sort_pushdown_buffer_capacity` config when changing sort-pushdown or TopK execution.
* `RepartitionExec` now serializes/deserializes `preserve_order` through `datafusion/proto`; when touching repartition planning or serde, keep this flag round-trippable and covered by tests.
* `SortMergeJoinExec` now routes semi/anti/mark joins through a specialized `bitwise_stream` path instead of the pair-materializing stream. When changing SMJ behavior, validate both the specialized and materializing code paths, including pending-boundary and spill scenarios.
* Validate both correctness and performance when touching joins, aggregation, sorting, spilling, or memory tracking.
* `AggregateExec::partition_statistics` now estimates grouped output rows from NDV, null counts, grouping sets, and TopK limits, and propagates group-column distinct counts into output statistics. Preserve these heuristics unless you are deliberately changing aggregate statistics behavior and updating tests accordingly.
* `FilterExec` statistics now collapse equality predicates to single-value intervals and set `distinct_count` to `Exact(1)` for those columns. Keep this behavior aligned with interval analysis and downstream statistics consumers.
* `ExplainAnalyzeLevel` has been renamed to `MetricType` in `datafusion/common/src/format.rs`; the old `MetricType::SUMMARY` (SCREAMING_SNAKE_CASE) variant is now `MetricType::Summary` (CamelCase). A new `MetricCategory` enum (Rows, Bytes, Timing) and `ExplainAnalyzeCategories` filter type enable per-category metric filtering in `EXPLAIN ANALYZE` output; tag new metrics with `with_category(MetricCategory::...)` via `MetricBuilder` so they surface correctly in filtered output.
* `EXPLAIN ANALYZE` datasource reporting now includes `output_rows_skew`; if you add new scan metrics, categorize them correctly and keep the explain docs/tests aligned with the emitted metric names and formatting.

#### Datasource and custom scans

* `DataSource` and `FileSource` implementations must provide `apply_expressions` so scan-time expressions (filters, projections, dynamic filters) remain discoverable by optimizer and execution rewrites.
* `DataSource::with_new_state` is the runtime state handoff hook; return `None` when a state payload is not applicable so plan-level propagation can continue.
* `DataSource::partition_statistics` now returns `Arc<Statistics>`; preserve shared ownership semantics and avoid regressing partition-statistics propagation in join/filter optimizations.
* `FileStream::new` is deprecated; construct file streams with `FileStreamBuilder`, and keep file-stream metrics wiring (`time_elapsed_*`, file error counters, processed/opened counters) intact when extending datasource scan execution.
* Morsel-driven scan APIs (`Morselizer`, `MorselPlanner`, `MorselPlan`) are now the execution pattern for staged scan CPU/I/O pipelines; for Parquet, keep `ParquetMorselizer` state-machine behavior aligned with this contract (CPU work in planning steps, I/O in pending planner futures).
* `ExtensionPlanner` can now plan `LogicalPlan::TableScan` for custom `TableSource` implementations. Ensure produced `ExecutionPlan` schemas match the logical projected schema.
* Newline-delimited JSON range scans now use `AlignedBoundaryStream` to align partition boundaries lazily at newline terminators instead of precomputing byte ranges with extra object-store probes. Preserve this boundary-aligned streaming path when changing NDJSON repartitioned reads; JSON array format still does not support range-based scanning.
* Parquet row-filter pushdown now resolves `get_field` paths to specific struct leaves; preserve leaf-level projection behavior (do not regress to projecting entire struct roots unless field paths are non-literal/unknown). When struct columns are present, `build_projection_read_plan` uses leaf-level `ProjectionMask` entries instead of root-level entries.
* Parquet row-filter predicate reordering now prioritizes required compressed bytes; avoid reintroducing legacy sorted-vs-unsorted partitioning heuristics without measured benefit.
* `PreparedAccessPlan` moved under `datafusion/datasource-parquet/src/access_plan.rs`; construct it via `ParquetAccessPlan::prepare(...)` and preserve reverse-scan behavior through `PreparedAccessPlan::reverse(...)`.
* `compute_file_group_statistics` / `compute_all_files_statistics` now intentionally skip expensive statistics merging when `collect_stats` is `false`; preserve that fast path and only rely on row counts for limit-driven file selection in that mode.
* `PruningStatistics::row_counts` is now container-level and takes no `Column` argument. New pruning implementations should expose row counts once per container, and pruning callers should not model row counts as column-specific metadata.
* Parquet writer config now supports `datafusion.execution.parquet.use_content_defined_chunking` with nested `min_chunk_size`, `max_chunk_size`, and `norm_level` options. CDC-enabled writes must validate chunk sizes, disable parallel writing, and round-trip through both `datafusion/proto-common` and logical-plan file-format serde.
* `datafusion/datasource-avro` now delegates all Avro-to-Arrow conversion to the upstream `arrow-avro` crate; the internal `avro_to_arrow/` module has been removed. Extend Avro reading via `arrow-avro` APIs rather than any internal path.

#### Substrait consumer and correlated subqueries

* `datafusion/substrait` consumer now resolves correlated references via an outer-schema stack (`OuterReference` with `steps_out`).
* When adding subquery-consumption paths, ensure outer schemas are pushed/popped around relation consumption so nested correlated subqueries resolve correctly.
* Keep correlated-subquery coverage in integration tests under `datafusion/substrait/tests/cases/consumer_integration.rs` and plan fixtures under `datafusion/substrait/tests/testdata/test_plans/`.
* SQL planning now has explicit `LATERAL` coverage for correlated subqueries in the `FROM` clause. Use `LATERAL` for same-`FROM` outer references, keep current decorrelation limits in mind (notably outer references outside supported positions and `LEFT JOIN LATERAL` gaps), and add SLT coverage under `datafusion/sqllogictest/test_files/lateral_join.slt` for planner changes in this area.

#### Session state, extension types, and table functions

* `SessionState` now carries an extension type registry. Wire custom registries through `SessionStateBuilder::with_extension_type_registry(...)`, and keep any extension-type behavior aligned across `datafusion/common`, planner/session state, examples, and tests.
* DataFusion now includes an initial extension-type surface in `datafusion/common/src/types/`, including canonical UUID pretty-printing and runnable examples under `datafusion-examples/examples/extension_types/`; extend that system rather than inventing parallel hooks.
* Table UDTFs now receive `TableFunctionArgs` via `TableFunctionImpl::call_with_args(...)` instead of only a raw expression slice. Use `args.exprs()` for coerced arguments and `args.session()` when table creation depends on session/catalog state.
* SQL projection names must now be unique; the planner no longer auto-suffixes duplicate select-list names. Preserve this behavior and alias duplicate expressions explicitly in tests and examples.

#### FFI and memory integration

* FFI extension points are active surfaces: config extensions and table provider factory integration live in `datafusion/ffi`.
* FFI scalar UDF calls now carry `ConfigOptions`; preserve this plumbing when modifying `ScalarFunctionArgs`, FFI bridges, or UDF invocation paths.
* Preserve `ForeignTableProvider::scan` semantics for `projection=None` (meaning “all columns”) across the FFI boundary.
* Keep FFI changes covered by tests under `datafusion/ffi/src/tests` and `datafusion/ffi/tests`.
* `FFI_PhysicalOptimizerRule` in `datafusion/ffi/src/physical_optimizer.rs` allows custom `PhysicalOptimizerRule` implementations to cross the FFI boundary; test coverage must live in `datafusion/ffi/src/tests/physical_optimizer.rs` and `datafusion/ffi/tests/`.
* `datafusion/execution/src/memory_pool/arrow.rs` provides Arrow `MemoryPool` integration for DataFusion memory tracking; keep memory accounting behavior consistent with `MemoryReservation` semantics.

#### Proto and plan serialization

* `DefaultLogicalExtensionCodec` now handles built-in file-format factories (CSV/JSON/Arrow/Avro/Parquet when enabled). Extend this codec path deliberately when adding new built-in table/file formats.
* `ParquetOptions` / `TableParquetOptions` protobuf and logical-plan serde now include content-defined chunking (`CdcOptions`). Keep CDC fields round-trippable and preserve proto defaults for unset chunk sizes when modifying Parquet serialization.
* When changing plan serialization in `datafusion/proto`, add/update roundtrip tests in `datafusion/proto/tests/cases/` for both logical and physical plans.

## 3. Testing & Validation

* **Unit Tests:** Cover individual functions and edge cases with `cargo test`. Prefer `cargo test --workspace` when your change crosses crate boundaries, or target specific packages with `-p <crate>` for focused runs.
* **Planner/Executor/Expr changes:** If you modify planner, executor, or expression behavior, add both unit tests and SQL Logic Tests under `datafusion/sqllogictest/test_files/`.
* **Integration Tests:** Validate end-to-end behavior, especially for CLI commands or key modules.
* **Continuous Testing:** Ensure tests run reliably in CI. Use `cargo nextest run --workspace` for faster, parallel execution when configured.
* **SQLLogicTest Timing Triage:** Use `--timing-summary` (or `SLT_TIMING_SUMMARY=true`) to enable a per-file elapsed-time report sorted slowest-first; when enabled, periodic `Progress:` lines are suppressed. The multi-mode flags (`auto`/`off`/`top`/`full`), `--timing-top-n`, `SLT_TIMING_TOP_N`, and `SLT_TIMING_DEBUG_SLOW_FILES` have been removed in favor of this simpler boolean flag.
* **SQLLogicTest Config Hygiene:** SLT files that modify configuration via `SET` must restore original values before file end; the DataFusion SLT runner now reports dangling config mutations at teardown.
* **SQLLogicTest Reset Workflow:** Prefer `RESET <config>` to restore session options inside SLT files now that SQL-level config reset is documented and supported.
* **SQLLogicTest Rust-backed setup:** If an SLT needs tables that SQL cannot express directly, add a test-harness registration helper in `datafusion/sqllogictest/src/test_context.rs` and invoke it from `TestContext::try_new_for_test_file` based on the `.slt` file name. `metadata.slt` is the reference pattern for Arrow field/schema metadata.
* **Minimize Test Binaries:** Each test binary requires additional build time and disk space. Consolidate related tests into fewer test files rather than creating many small test binaries. Use `#[cfg(test)]` modules within `src/` files for unit tests, and reserve `tests/` directories for true integration tests that require a separate binary. Group logically related integration tests into shared test files with multiple `#[test]` functions.

* **Prefer SQL Logic Tests (SLT) over snapshots:** For new SQL/engine tests prefer adding SQL Logic Tests (SLT) under `datafusion/sqllogictest/test_files/` instead of creating or relying on snapshot (.snap) files. SLT files are easier to review and maintain. If a snapshot file is absolutely necessary, include a brief justification in the PR description.

---

### 3.1 macOS Profiling & Performance Triage (all performance investigations)

This section applies to **all performance investigations performed on macOS**. Always measure first and capture reproducible evidence before changing code.

* Start with crate-scoped runs for faster iteration:

  * `cargo test -p <crate> --release -- <args>`
  * or run the target binary directly from `target/release/`
* Record in your PR or issue:

  * Exact command line
  * Git commit hash

---

### Required macOS CPU profiling workflow (Instruments / Time Profiler)

All CPU-bound investigations on macOS must use **Xcode Instruments – Time Profiler** via `xcrun xctrace`.

1. Build with symbols (release + debug info):

```bash
RUSTFLAGS="-g" cargo build -p <crate> --release
```

2. Record a profile (launch preferred over attach):

```bash
mkdir -p profiling

xcrun xctrace record \
  --template "Time Profiler" \
  --output profiling/time-profiler.trace \
  --launch -- target/release/<binary> <ARGS>
```

3. Export profiling data for review:

```bash
mkdir -p profiling/export

xcrun xctrace export \
  --input profiling/time-profiler.trace \
  --output profiling/export
```

4. PR reporting requirements:

Include:

* Total wall-clock runtime
* Top 10 hottest functions or stack traces
* % CPU attributed to dominant subsystems (examples below)
* A short interpretation of the bottleneck

---

### What to Look For 

Identify which subsystem dominates CPU time. Common examples in DataFusion:

* Compression/decompression (zstd, parquet codecs)
* Arrow array construction or kernel execution
* Hashing / grouping (`hashbrown`, aggregation hot loops)
* Sorting (`sort`, `sort_preserving_merge`)
* Join execution (hash join build/probe imbalance)
* Memory allocation / excessive cloning
* Expression evaluation overhead (`PhysicalExpr` hot paths)
* Planner or optimizer recursion (unexpected planning hotspots)

Report the dominant subsystem explicitly, for example:

> "~32% CPU in hash aggregation build phase"
> "~18% CPU in parquet decode"
> "~27% CPU in expression evaluation during projection"

Avoid vague statements like "it seems slow" — always quantify.

---

### Performance Change Requirements

When proposing a performance optimization:

* Include **before/after numbers** using the same binary, arguments, and dataset.
* Measure in `--release` mode only.
* Do not mix unrelated refactors with performance changes.
* If optimizing test-only behavior, explicitly justify correctness and confirm no production-path regression.
* Store exact commands used in `profiling/commands.txt` for reproducibility.

Optimizations must target demonstrated bottlenecks. Avoid speculative micro-optimizations without profiling evidence.

## 4. Documentation & Examples

* Provide clear README updates for new features or changes.
* Use the Architecture Guide as a contributor reference: `docs/source/contributor-guide/architecture.md`.
* Include practical examples in code comments and the `examples/` directory.
* Update CLI help strings and guides to reflect enhancements.

## 5. Collaboration & Review

* **Pull Requests:** Provide a concise summary of changes, motivation, and how to test.
* **Code Reviews:** Offer constructive feedback focusing on clarity, correctness, and design.
* **Discussions:** Use issues to propose major changes or ask design questions.

### PR discussion & review response style

When responding in PR discussions or code reviews, use a conversational, approachable tone that explains the reasoning behind design decisions.

Structure responses with:

- **An opening statement** that directly addresses the core insight.
- **Concrete headings** that call out key reasons (for example: "Clear intent", "Easier to reason about").
- **Real code examples** that show how the design behaves in practice.
- **Practical benefits** tied to readability, maintainability, and developer experience.
- **Code references where possible** that point to relevant files and line numbers so reviewers can verify context quickly.

Focus on why a design choice is better instead of only asserting it is better. Keep responses engaging and persuasive without becoming overly formal.

Example framing for explicit state fields:

```rust
if self.has_grouping_set {
  // handle grouping-set specific behavior
}
```

Prefer explanations like: explicit checks such as `if self.has_grouping_set { ... }` are easier to scan and understand than repeatedly pattern-matching on `Option` values across the codebase.

## 6. Optional Tooling Checks

For major changes, run `./dev/rust_lint.sh` before proposing the change so formatting/lint/doc issues are caught early.

While linting, formatting, and spelling are important, they should not overshadow solution quality. Please run formatting, linting, and `typos` **after** finalizing your code when applicable:

```bash
# Optional but recommended:
./dev/rust_lint.sh    # Formats, lints, and checks docs
./pre-commit.sh       # Runs clippy and fmt for staged Rust files
uv sync               # Syncs Python workspace dependencies (docs/dev/benchmarks)
uv run --package datafusion-docs ./docs/build.sh  # Builds docs in uv workspace
prettier -w <path/to/file.md>  # Formats Markdown
taplo format --check  # Validates TOML
typos                 # Checks for spelling mistakes
```

Ensure these checks pass before merging (run `typos` when applicable), but prioritize delivering clear, well-designed code. `./dev/rust_lint.sh` bootstraps `taplo` automatically if it is missing so that formatting checks match CI.

## Useful Helper Functions

Below are helper modules and functions that simplify common tasks across the codebase:

* `datafusion/common/src/utils/string_utils.rs`

  * `string_array_to_vec` converts Arrow string arrays into `Vec<Option<&str>>` for easier Rust processing.
* `datafusion/common/src/hash_utils.rs`

  * `combine_hashes` merges two `u64` values and backs hash-related utilities for arrays.
* `datafusion/common/src/utils/aggregate.rs`

  * `precision_add` performs overflow-aware scalar precision addition without Arrow array round-trips and is preferred in statistics aggregation hot paths.
* `datafusion/common/src/test_util.rs`

  * `format_batches` pretty-prints `RecordBatch` collections.
  * Macros such as `assert_batches_eq`, `assert_batches_sorted_eq`, `assert_contains`, and `assert_not_contains` aid concise test assertions.
* `datafusion/expr/src/utils.rs`

  * `grouping_set_expr_count` counts unique grouping expressions, accounting for `GROUPING SETS`.
  * `merge_grouping_set` joins two grouping sets while enforcing size limits.
  * `cross_join_grouping_sets` builds the Cartesian product of grouping-set combinations.
* `datafusion/physical-optimizer/src/utils.rs`

  * `add_sort_above` and `add_sort_above_with_check` inject sorting into physical plans.
  * Helper predicates like `is_sort`, `is_window`, `is_union`, `is_sort_preserving_merge`, `is_coalesce_partitions`, `is_repartition`, and `is_limit` classify execution plan nodes.
* `datafusion/physical-expr-common/src/utils.rs`

  * `ExprPropertiesNode::new_unknown` initializes property tracking with unknown order and range for expression trees.
  * `scatter` performs mask-driven array scatter, filling nulls where the mask is false.
* `datafusion/functions-window/src/utils.rs`

  * `get_signed_integer`, `get_scalar_value_from_args`, and `get_unsigned_integer` handle window-function argument extraction.
* `datafusion/functions-aggregate-common/src/utils.rs`

  * `get_accum_scalar_values_as_arrays`, `ordering_fields`, and `get_sort_options` support aggregate computation internals, while helpers such as `DecimalAverager` and `Hashable<T>` keep decimal results precise and floating-point hashes stable.
* `datafusion/sqllogictest/src/util.rs`

  * Utilities like `setup_scratch_dir`, `value_normalizer`, `read_dir_recursive`, `df_value_validator`, and `is_spark_path` assist SQLLogicTest harnesses.
* `datafusion-cli/src/helper.rs`

  * `CliHelper` offers interactive SQL parsing, dialect switching, and the `split_from_semicolon` helper for multi-statement inputs.
* `datafusion/functions/src/utils.rs`

  * `make_scalar_function` wraps array-based logic for scalars and arrays, while macros like `utf8_to_str_type` and `utf8_to_int_type` derive optimal return types.
* `datafusion/functions-nested/src/utils.rs`

  * `check_datatypes` verifies all array arguments share compatible types, returning an error otherwise.
  * `make_scalar_function` adapts array-oriented closures to work with `ColumnarValue` inputs and preserves scalar outputs when possible.
  * `align_array_dimensions` pads nested list arrays so every argument reaches the same number of dimensions.
* `datafusion/functions-window-common/src/partition.rs`

  * `PartitionEvaluatorArgs` bundles window function arguments, fields, reversal flags, and `IGNORE NULLS` handling for custom evaluators.
* `datafusion/optimizer/src/utils.rs`

  * Utilities such as `has_all_column_refs`, `replace_qualified_name`, `is_restrict_null_predicate`, and `evaluates_to_null` support query optimizer rules.
* `datafusion/sql/src/utils.rs`

  * Helpers like `resolve_columns`, `rebase_expr`, and `check_columns_satisfy_exprs` aid SQL planning and validation.
* `datafusion/common-runtime/src/trace_utils.rs`

  * `set_join_set_tracer`, `trace_future`, and `trace_block` inject consistent instrumentation for asynchronous and blocking workloads.
* `datafusion/common-runtime/src/join_set.rs` and `datafusion/common-runtime/src/common.rs`

  * `JoinSet` and `SpawnedTask` wrap Tokio primitives with automatic tracing and cancellation to simplify task management.
* `datafusion/physical-plan/src/metrics`

  * `ExecutionPlanMetricsSet`, `MetricBuilder`, and `BaselineMetrics` streamline exposing counters, timers, and spill statistics from execution operators.
* `datafusion/physical-expr-adapter/src/schema_rewriter.rs`

  * `PhysicalExprAdapter` and `DefaultPhysicalExprAdapterFactory` rewrite physical expressions to match differing logical and physical schemas, handling casts, missing columns, and partition values.

## Commenting guidance

Use three complementary kinds of comments in the codebase to keep intent clear and the public API documented:

- Implementation Comments
  - Explains non-obvious choices and tricky implementations.
  - Serves as breadcrumbs for future developers when reasoning about why code is written in a particular way.

- Documentation Comments
  - Describes functions, types, traits, modules and their public behaviour and contracts.
  - Acts as the public interface documentation (prefer `///` Rust doc comments for public Rust items).

- Contextual Comments
  - Documents assumptions, preconditions, invariants, and non-obvious requirements.
  - Use these to record constraints that aren't enforced directly in code (e.g., expected input ranges, thread-safety considerations, or compatibility notes).

Keep comments short, factual, and up-to-date. Prefer code clarity and small helper functions over long explanatory blocks. When a comment becomes longer than a paragraph, prefer extracting intent into a well-named function or adding a `TODO` with a short plan.
