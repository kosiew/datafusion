# Fix Steps: Using Debug-Gated Timing + A/B Sweep Benchmarks

This guide describes a repeatable workflow for fixing `push_down_filter` performance issues using:

1. Debug-gated section timing instrumentation in `datafusion/optimizer/src/push_down_filter.rs`
2. The A/B comparison benchmark sweep in `datafusion/core/benches/sql_planner_extended.rs`

The goal is to identify the true hot section, implement a focused fix, and verify it across a parameter sweep.

## Prerequisites

- Use the repo toolchain (`rust-toolchain.toml`).
- Build at least once so benchmark iterations are not dominated by first-time compilation.

Suggested one-time compile check:

```bash
cargo check -p datafusion --benches
```

## Step 1: Capture Baseline on Base Branch (main)

Before testing your fix branch, collect baseline on a stable base (typically `main` or the target base commit).

Suggested flow:

```bash
# 1) switch to base
git checkout main

# 2) run A/B sweep and capture numbers
cargo bench -p datafusion --bench sql_planner_extended -- push_down_filter_case_heavy_left_join_ab

# 3) (optional) save criterion baseline for later direct comparison
cargo bench -p datafusion --bench sql_planner_extended -- \
  --save-baseline pushdown-main push_down_filter_case_heavy_left_join_ab
```

Then switch back to your feature branch for iterative work.

## Step 2: Establish A/B Status on Feature Branch

Run the case-heavy A/B benchmark group:

```bash
cargo bench -p datafusion --bench sql_planner_extended -- push_down_filter_case_heavy_left_join_ab
```

What to capture:

- Cases where `with_push_down_filter` is slower than `without_push_down_filter`
- Worst regression points by sweep dimensions:
  - `predicates={10,20,30,40,60}`
  - `case_depth={1,2,3}`

Output to record in notes:

- Benchmark ID (for example: `predicates=60,case_depth=3`)
- Relative slowdown (`with` vs `without`)

If you saved a Criterion baseline on `main`, compare directly on feature branch:

```bash
cargo bench -p datafusion --bench sql_planner_extended -- \
  --baseline pushdown-main push_down_filter_case_heavy_left_join_ab
```

## Step 3: Select a Single Worst-Case Repro

Pick one worst point from Step 2 and iterate on that single case first.

Use Criterion filtering to run a narrow subset:

```bash
cargo bench -p datafusion --bench sql_planner_extended -- \
  push_down_filter_case_heavy_left_join_ab/with_push_down_filter/predicates=60,case_depth=3
```

Use the matching `without_push_down_filter` ID as a local baseline for quick re-checks.

## Step 4: Enable Debug-Gated Timing Instrumentation

Run the same case with debug logging enabled for `push_down_filter`:

```bash
RUST_LOG=datafusion_optimizer::push_down_filter=debug \
cargo bench -p datafusion --bench sql_planner_extended -- \
  push_down_filter_case_heavy_left_join_ab/with_push_down_filter/predicates=60,case_depth=3
```

Look for logs like:

- `push_down_filter_timing: section=infer_join_predicates, elapsed_us=...`
- `push_down_filter_timing: section=simplify_predicates, elapsed_us=...`

These come from `with_debug_timing(...)` and identify which section dominates.

## Step 5: Form a Narrow Hypothesis

Based on Step 4 timing, write a single concrete hypothesis.

Examples:

- `infer_join_predicates` dominates due to repeated expression traversal on large conjunctions.
- `simplify_predicates` dominates due to expensive work on unchanged predicate structures.

Avoid broad refactors before a section-level hotspot is clear.

## Step 6: Implement a Focused Change

Apply the smallest code change that addresses the measured hotspot.

Typical fix styles:

- Hoist/reuse computed structures to avoid repeated work.
- Add early returns for no-op/cheap paths.
- Reduce unnecessary cloning/allocation in the hot section.

Keep behavior and correctness unchanged.

## Step 7: Validate the Hotspot Improvement

Re-run the same single-case benchmark with debug logs:

```bash
RUST_LOG=datafusion_optimizer::push_down_filter=debug \
cargo bench -p datafusion --bench sql_planner_extended -- \
  push_down_filter_case_heavy_left_join_ab/with_push_down_filter/predicates=60,case_depth=3
```

Confirm two things:

- Section timing decreased for the targeted hotspot.
- Criterion result improved for the same benchmark ID.

If only one improved, refine the hypothesis and repeat Steps 5-7.

## Step 8: Re-run Full A/B Sweep

Once single-case improvement is confirmed, validate across the full matrix:

```bash
cargo bench -p datafusion --bench sql_planner_extended -- push_down_filter_case_heavy_left_join_ab
```

Success criteria:

- Most or all prior regressions are reduced or eliminated.
- No obvious new regressions at other sweep points.

Also compare to the saved `main` baseline if available:

```bash
cargo bench -p datafusion --bench sql_planner_extended -- \
  --baseline pushdown-main push_down_filter_case_heavy_left_join_ab
```

## Step 9: Sanity + Compile Checks

Run at least crate-scoped checks before proposing the patch:

```bash
cargo check -p datafusion --benches
```

If the change is larger, run broader checks as needed (`cargo test -p datafusion`, then workspace scope if required).

## Step 10: Document Evidence in PR/Issue

Include:

- Baseline (`main`) vs feature-branch numbers for worst-case IDs
- Before/after debug timing for hotspot sections
- Brief explanation of why the fix helped

This keeps review objective and makes regressions easier to detect later.

## Recommended Iteration Loop

Use this loop until resolved:

1. Baseline on `main`.
2. Sweep A/B on feature branch to identify worst case.
3. Single-case run with debug timing to isolate hotspot.
4. Focused fix.
5. Single-case verify.
6. Full-sweep and (optionally) Criterion baseline verify.

## Why Keep Both Instruments

- A/B sweep benchmark: detects whether the optimizer rule is helping or hurting across realistic complexity gradients.
- Debug-gated section timing: explains where time is spent inside the rule when debug logs are enabled.

Together they provide both external impact (`with` vs `without`) and internal causality (which section regressed).
