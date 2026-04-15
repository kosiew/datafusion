stale
source: pr-22322_a
# Benchmark Measures Planning Despite Claiming Execution-Only Timing

## Summary
The benchmark module documentation states that queries are pre-planned and only execution is timed, but each Criterion iteration currently invokes `ctx.sql(sql)` before `df.collect()`. This includes SQL parsing, logical planning, and optimization in the timed section.

## Scope
- Component: DataFusion benchmarks
- File: `datafusion/core/benches/multi_group_by.rs`

## Current Behavior
- Timed function executes:
  1. `ctx.sql(sql)`
  2. `df.collect()`
- The planning step runs on every iteration.
- Planning cost can scale with column count and query complexity, confounding a benchmark intended to isolate execution performance.

## Expected Behavior
If benchmark intent is execution-only:
1. Build the logical/physical plan once during setup.
2. Time only repeated execution of the prepared plan.

If planning is intentionally included:
1. Update benchmark docs/comments/names to explicitly state end-to-end query timing.

## Why This Matters
- The current setup can blur the observed effect of grouping implementation changes.
- Increased column count may inflate planning overhead, making execution comparisons less reliable.
- Benchmark consumers may draw incorrect conclusions due to mismatch between docs and measured scope.

## Reproduction / Validation Notes
1. Read module docs in `datafusion/core/benches/multi_group_by.rs`.
2. Inspect timed query function and confirm `ctx.sql(sql)` is inside the benchmark iteration closure.
3. Compare planning+execution vs execution-only timing to observe impact.

## Proposed Fix Options
1. Refactor benchmark setup to pre-create DataFrame/plan outside `b.iter`.
2. Keep warmup for I/O/cache as needed, but separate from timed execution body.
3. If required, add separate benchmark groups:
   - `plan_plus_execute`
   - `execute_only`

## Acceptance Criteria
1. Timed closure for execution-only benchmark does not call `ctx.sql(sql)`.
2. Module docs accurately describe measured scope.
3. Benchmark names make timing scope explicit.
4. Benchmark compiles and runs with existing command.

## Suggested Labels
- `benchmark`
- `performance`
- `measurement`
