stale
source: pr-22322_a
# Benchmark Labels Use Theoretical Key-Space, Not Measured Distinct Groups

## Summary
Several benchmark names and comments present theoretical key-space sizes (for example `30^4 = 810K`, `100^4 = 100M`, `500^4 = 62.5B`) as if they were realized group counts. With roughly 1M input rows sampled independently, observed distinct groups can be substantially lower (or capped near row count), so labels are misleading for a benchmark focused on group-count thresholds.

## Scope
- Component: DataFusion benchmarks
- File: `datafusion/core/benches/multi_group_by.rs`

## Current Behavior
- Case labels/comments imply actual group counts equal full key-space size.
- Data generation uses random sampling over a large key-space with fixed row count.
- Realized NDV (number of distinct values/groups) is stochastic and often far below key-space.

## Expected Behavior
Benchmark output should reflect what is actually measured:
1. Either generate controlled data with deterministic target NDV per case.
2. Or label cases as key-space cardinality and also report measured NDV.

## Why This Matters
- The PR motivation references crossover by distinct-group count.
- Mislabeling key-space as realized groups can lead to incorrect threshold conclusions.
- Readers cannot accurately compare runs without understanding true NDV.

## Reproduction / Validation Notes
1. Inspect case names/comments in `datafusion/core/benches/multi_group_by.rs` for `810K`, `100M`, `62B` style labels.
2. Run a distinct-count query on generated data to measure realized NDV.
3. Compare measured NDV to theoretical key-space for the same case.

## Proposed Fix Options
1. Controlled NDV generation:
   - Construct key tuples so exact NDV is known and repeatable.
2. Clarified labeling:
   - Rename benchmarks to key-space terms (for example `keyspace_100M`).
   - Print/log measured NDV during setup and include in benchmark output.
3. Keep deterministic random seed and report row count plus measured NDV per case.

## Acceptance Criteria
1. Case names no longer imply realized NDV unless that NDV is actually controlled/verified.
2. Measured NDV is either explicitly reported or deterministically enforced.
3. Benchmark docs explain distinction between key-space and observed distinct groups.
4. Benchmark compiles and runs with existing command.

## Suggested Labels
- `benchmark`
- `performance`
- `measurement`
