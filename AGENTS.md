# Repository Guidelines for Codex Agents

This repository uses Rust and contains documentation in Markdown and TOML configuration files. Our focus is on delivering robust, maintainable, and high-quality solutions. Please follow these guidelines when contributing.

---

## Toolchain & Workspace Notes

* The workspace pins the Rust toolchain to `1.90.0` via `rust-toolchain.toml` while declaring an MSRV of `1.87.0` in `Cargo.toml`. Use `rustup override set 1.90.0` if you are not already using the bundled toolchain.
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

## 1. Solution-First Approach

* **Understand the Problem:** Before writing code, ensure you clearly understand the requirements and desired outcomes. Ask clarifying questions if anything is ambiguous.
* **Design Thoughtfully:** Sketch out the architecture or data flow. Consider performance, scalability, and readability.
* **Provide Examples:** Include usage examples or code snippets demonstrating how your feature or fix works.

## 2. Code Quality & Best Practices

### Idiomatic Rust

* Prefer `snake_case` for variables, functions, and modules; `CamelCase` for types and enums.
* Use pattern matching, `Option<T>`, and `Result<T, E>` for clear and safe handling of optionality and errors.
* **Think of `Option` as a computation pipeline, not just "value or no value."** Stop treating `Option` merely as presence/absence—view it as a lazy transformation stream. Combinators like `.map()`, `.and_then()`, `.filter()`, and `.ok_or()` turn error-handling into declarative data flow. Once you internalize absence as a first-class transformation, you can write entire algorithms that never mention control flow explicitly—yet remain 100% safe and compiler-analyzable. This paradigm shift transforms imperative conditionals into composable, chainable operations.
* Leverage iterators and functional constructs for concise, efficient code.
* **Refactor imperative traversals** (e.g. using `.apply()` with mutable flags) **into declarative expressions** (e.g. using `.exists()`, `.any()`, or `.find()`) wherever possible. This makes the intent clearer, eliminates boilerplate, and improves maintainability.

  > ✅ *Example:* Convert verbose recursion with `mut` flags into `.exists()` + `.unwrap()` patterns for simple boolean checks.
* **Simplify match arms by directly binding variables** (e.g. `LogicalPlan::Foo(x)` instead of `LogicalPlan::Foo(_)` + `if let`). This reduces redundant re-matching and enhances clarity.

  > ✅ *Example:* Collapse a redundant `if let` on a known variant by binding it directly in the `match` arm.
  > Also take the opportunity to **clarify comments or add TODOs**, especially around complex control flows like CTEs or plan rewriting.

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

### Error Handling & Context

* Use the `?` operator to propagate errors with context. Avoid silent failures.
* Implement custom errors via the `thiserror` crate when appropriate.
* Provide clear messages to aid debugging and user feedback.

### Performance Considerations

* Favor zero-copy patterns: use `&str` over `String` and `Arc<T>` for shared data.
* Avoid unnecessary heap allocations; minimize cloning.
* Benchmark critical paths when performance is a concern.
* Optimizations should be focused on bottlenecks — those steps that are repeated millions of times in a query; otherwise, prefer simplicity.

* Prefer multiple simple code paths over a single complex adaptive path. Optimize for the common case first and keep that path fast and easy to reason about; handle rare or complex edge cases with separate, well-tested branches or fallbacks. This often yields clearer, faster, and more maintainable code than trying to build one highly adaptive, catch-all implementation.

### Clone & Copy: Precision over Reflex

* **Treat `Clone` as a precision instrument, not a reflex.** Implement custom `Clone` to achieve amortized O(1) behavior where possible — clone only references or indices, not deep data structures, then perform explicit duplication at controlled boundaries.
* **Reserve `Copy` for plain-old-data only.** Restrict `Copy` to types where move and copy are semantically and computationally indistinguishable: integers, coordinates, short fixed-size math vectors, and similar primitives.
* **Make ownership boundaries visible.** For everything else, use intentional `Clone` APIs so that duplication shows up clearly in code reviews and profiling. Explicit `.clone()` calls document where data is being duplicated and help identify optimization opportunities.
* **Design for shared ownership.** Prefer `Arc<T>` and reference-counted patterns over deep cloning when multiple owners need access to the same data. Clone the `Arc`, not the underlying data.

## 3. Testing & Validation

* **Unit Tests:** Cover individual functions and edge cases with `cargo test`. Prefer `cargo test --workspace` when your change crosses crate boundaries, or target specific packages with `-p <crate>` for focused runs.
* **Integration Tests:** Validate end-to-end behavior, especially for CLI commands or key modules.
* **Continuous Testing:** Ensure tests run reliably in CI. Use `cargo nextest run --workspace` for faster, parallel execution when configured.
* **Minimize Test Binaries:** Each test binary requires additional build time and disk space. Consolidate related tests into fewer test files rather than creating many small test binaries. Use `#[cfg(test)]` modules within `src/` files for unit tests, and reserve `tests/` directories for true integration tests that require a separate binary. Group logically related integration tests into shared test files with multiple `#[test]` functions.

## 4. Documentation & Examples

* Provide clear README updates for new features or changes.
* Include practical examples in code comments and the `examples/` directory.
* Update CLI help strings and guides to reflect enhancements.

## 5. Collaboration & Review

* **Pull Requests:** Provide a concise summary of changes, motivation, and how to test.
* **Code Reviews:** Offer constructive feedback focusing on clarity, correctness, and design.
* **Discussions:** Use issues to propose major changes or ask design questions.

## 6. Optional Tooling Checks

While linting, formatting, and spelling are important, they should not overshadow solution quality. Please run formatting, linting, and `typos` **after** finalizing your code when applicable:

```bash
# Optional but recommended:
./dev/rust_lint.sh    # Formats, lints, and checks docs
./pre-commit.sh       # Runs clippy and fmt for staged Rust files
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
