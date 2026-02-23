# CI Investigation: `Run sqllogictests with the sqlite test suite`

## Question
Why did the CI task
`Datafusion extended tests / Run sqllogictests with the sqlite test suite (pull_request)`
increase from about 1 hour to about 2 hours after merge commit `76be0b64c`?

## Scope
Compared:
- pre-merge parent: `8f959bba6`
- merge result: `76be0b64c`

## Findings

### 1) CI workflow/job definition did not change in the merge
I diffed `.github/workflows/extended.yml` between `8f959bba6` and `76be0b64c` and found no changes.

Implication: the slowdown is not explained by a direct change to the job steps, image, or command in this merge.

### 2) The sqllogictest workload increased materially
`datafusion/sqllogictest/test_files` changes in this merge range:
- `34 files changed`
- `+2359 / -296` lines (net `+2063`)
- new files include:
  - `join_limit_pushdown.slt`
  - `spark/bitmap/bitmap_bit_position.slt`
  - `spark/bitmap/bitmap_bucket_number.slt`
  - `spark/json/json_tuple.slt`

Largest growth files:
- `sort_pushdown.slt`: `+748`
- `projection_pushdown.slt`: `+348/-191`
- `dynamic_filter_pushdown_config.slt`: `+301`
- `join_limit_pushdown.slt`: `+269`

Aggregate test corpus size in `datafusion/sqllogictest/test_files`:
- files: `459 -> 463`
- lines: `134122 -> 136189`
- runnable records (query/statement/skipif/onlyif markers): `14845 -> 15110` (`+265`)

Implication: the same CI command now executes more sqllogictest content than before.

### 3) Sqllogictest crate/dependency changes also landed from `main`
In `datafusion/sqllogictest/Cargo.toml`:
- `sqllogictest 0.29.0 -> 0.29.1`
- `clap 4.5.57 -> 4.5.60`

`Cargo.lock` in the merge range changed significantly (`+350/-185`), including new packages.

Implication: compile/setup time for the job can increase even if workflow YAML is unchanged.

### 4) Datafusion engine/query-planning code changed heavily in the merge range
This merge pulled many optimizer/execution changes from `main` (plus extensive sqllogictest updates). Even with "perf" commits, net runtime of this specific test corpus can still shift.

Implication: execution time of thousands of sqllogictest queries can change due to planner/executor behavior changes, not only due to test-count growth.

## Most likely explanation
The duration increase is most likely from **workload growth + dependency/build churn introduced from `main`**, not from a workflow definition change in commit `76be0b64c` itself.

In other words, `76be0b64c` is the integration point where many upstream changes became active on this branch.

## Confidence
- High confidence: no job YAML change in this merge, and sqllogictest corpus/deps grew.
- Medium confidence on exact split between "build-time increase" vs "test-runtime increase" because I could not fetch GitHub step timing logs in this environment.

## Limitation encountered
`gh auth status` shows the local GitHub token is invalid, so I could not inspect historical GitHub Actions step durations for direct Build-vs-Run timing attribution.

## Recommended next check (to confirm exact driver)
Compare step durations for two runs (before/after `76be0b64c`) for:
1. `Build sqllogictest binary`
2. `Run sqllogictest`

If Build step grew most: dependency/compile churn is primary.
If Run step grew most: test corpus / query execution behavior is primary.
