source: pr-23651_a
# Consolidate physical plan display option propagation

## Problem

Physical plan rendering options are threaded through several independent call paths with repeated builder chains. Recent statistics registry propagation required coordinated edits in:

- `datafusion/core/src/physical_planner.rs` for `EXPLAIN` physical plan rendering
- `datafusion/physical-plan/src/analyze.rs` for `EXPLAIN ANALYZE`
- `datafusion/physical-plan/src/display.rs` for `DisplayableExecutionPlan`

The repeated pattern looks like:

```rust,ignore
displayable(plan)
    .set_show_statistics(show_statistics)
    .set_statistics_registry(statistics_registry.clone())
    .set_show_schema(show_schema)
    .indent(verbose)
```

Each renderer decides locally which display options to apply. This makes it easy for a future display option to be added to one branch but missed in another. Some omissions are intentional, such as schema-only verbose branches not needing a statistics registry, but that intent is currently encoded by omission rather than by a named rendering mode.

## Why it matters

`EXPLAIN` and `EXPLAIN ANALYZE` are user-facing debugging APIs. Their output should reflect the same display configuration regardless of whether the plan is initial, optimized, final, verbose-with-stats, or rendered after execution.

Scattered option propagation increases risk that:

- `EXPLAIN` and `EXPLAIN ANALYZE` diverge for the same session config
- a new display option is silently ignored by some formats or branches
- reviewers have to audit repeated fluent chains instead of a single rendering contract
- future changes add local defensive patches instead of fixing the propagation boundary

## Invariant / desired behavior

For a given physical plan render mode, there should be one canonical place that defines which display options are applied.

Adding a new physical plan display option should require updating one local representation or helper, not searching every `displayable(...).set_*()` chain across planner, analyze, and display code.

Intentional differences between render modes, such as “schema-only verbose branch does not need statistics registry,” should be explicit in names or types rather than accidental missing setters.

## Proposed direction

Introduce a small, local display-options abstraction for physical plan rendering. Keep it narrow and behavior-preserving.

Possible shapes:

1. A private helper near the `EXPLAIN` indent rendering path:

```rust,ignore
fn render_physical_plan(
    plan: &dyn ExecutionPlan,
    options: &PhysicalPlanDisplayOptions,
    verbose: bool,
) -> String
```

2. A small options struct consumed by `DisplayableExecutionPlan`:

```rust,ignore
struct PhysicalPlanDisplayOptions {
    show_statistics: bool,
    statistics_registry: StatisticsRegistry,
    show_schema: bool,
    metric_types: Vec<MetricType>,
    metric_categories: Option<Vec<MetricCategory>>,
    tree_maximum_render_width: usize,
}
```

3. A method on `DisplayableExecutionPlan` that applies a grouped set of options:

```rust,ignore
impl DisplayableExecutionPlan<'_> {
    fn with_display_options(self, options: &PhysicalPlanDisplayOptions) -> Self
}
```

Prefer the smallest option that removes duplicated option plumbing without expanding public API unless there is a clear downstream need.

Also consolidate the default initialization in `DisplayableExecutionPlan::{new, with_metrics, with_full_metrics}` with a private constructor that takes `ShowMetrics`, so adding a new display field is not a three-site edit.

## Scope

### In

- Consolidate repeated physical plan display option setup in `datafusion/core/src/physical_planner.rs`.
- Consolidate repeated default field initialization in `datafusion/physical-plan/src/display.rs`.
- Ensure `AnalyzeExec` rendering receives the same relevant options as `EXPLAIN` rendering.
- Make intentional render-mode differences explicit, especially stats-only vs schema-only verbose branches.
- Preserve current output for existing `EXPLAIN` / `EXPLAIN ANALYZE` formats.
- Add or update snapshot/regression tests that cover statistics, schema, metrics, and verbose output combinations.

### Out

- No semantic changes to physical planning or optimization.
- No redesign of `ExplainFormat`.
- No new user-facing config options.
- No changes to statistics computation itself.
- No broad public API addition unless required to avoid worse duplication.

## Acceptance criteria

- [ ] There is a single local helper/representation for applying common physical plan display options.
- [ ] Adding a new display option to the common path requires changing one central location plus tests, not each `EXPLAIN` branch.
- [ ] `DisplayableExecutionPlan::{new, with_metrics, with_full_metrics}` no longer duplicate all default field initialization.
- [ ] Existing `EXPLAIN` and `EXPLAIN ANALYZE` output is unchanged except for intentional snapshot updates explained in the PR.
- [ ] Schema-only and stats-only verbose branches remain distinct and their option differences are explicit.
- [ ] The refactor does not make `StatisticsRegistry` mandatory for render paths that do not display statistics.

## Tests / verification

- Run focused physical plan display tests, including existing explain/analyze snapshots.
- Add or update SQLLogicTests or snapshot tests for:
  - plain `EXPLAIN` with `show_statistics`
  - `EXPLAIN VERBOSE` with stats enabled and disabled
  - `EXPLAIN` with schema output
  - `EXPLAIN ANALYZE` with metrics and statistics
- Run at minimum:

```bash
cargo test -p datafusion-physical-plan --lib display
cargo test -p datafusion --lib explain
```

Adjust exact test filters to match existing test names.

## Notes / open questions

- Decide whether the display-options abstraction should remain private to planner/display modules or become part of `DisplayableExecutionPlan`'s public builder API.
- If public API is changed, check semver impact and update docs accordingly.
