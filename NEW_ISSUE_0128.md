source: pr-23975_a
# Prevent incomplete statistics baselines from being used in compare mode

## Problem
The statistics runner persists report output after each statement by replacing statistics.json repeatedly during execution. If the run is interrupted or later input fails (for example unreadable SQL file), the resulting file may contain a successful prefix while still representing an incomplete run.

Current behavior in benchmarks/src/statistics.rs:
- After each statement, store_report writes the current prefix to the final report path.
- Comparison loading reads that report as authoritative if the file exists and deserializes.
- There is no completion marker or run status that distinguishes complete from partial outputs.

This allows compare mode to treat an incomplete prior run as a valid baseline.

## Why it matters
- Correctness: comparisons can be based on truncated data.
- Trust: benchmark regressions/improvements may be misreported due to missing tail queries.
- Operational safety: interruptions and incidental I/O failures silently poison later comparisons.

## Invariant / desired behavior
A report must be considered a valid comparison baseline only if it is marked complete for the full intended run.

Partial or interrupted outputs must be rejected (or ignored) as baselines.

## Proposed direction
Introduce explicit run-completion semantics with minimal persistence complexity:
- During execution, write progress to an in-progress artifact (or include completion metadata set to false).
- Publish or mark final report as complete only after all query files and statements have been processed.
- On compare load, reject incomplete artifacts with a clear warning/error and fall back to no-baseline behavior.

Two acceptable implementation shapes:
- Atomic publish model:
  - Write progress to a side file and only rename to final statistics file once complete.
- Status metadata model:
  - Extend report format with top-level status metadata (complete true/false, optional run id/timestamps) and validate on load.

Prefer the smallest change that preserves backward compatibility expectations for local tooling.

## Scope
### In
- Completion-aware baseline persistence for statistics runner.
- Baseline load validation that rejects incomplete reports.
- Tests for interruption-adjacent scenarios represented via partial files.

### Out
- Full durable checkpoint/restart framework.
- CI-specific policy changes for nonzero exit behavior.
- Large redesign of benchmark report schemas beyond required status metadata.

## Acceptance criteria
- [ ] Incomplete runs cannot overwrite or masquerade as complete comparison baselines.
- [ ] Compare mode rejects incomplete baseline artifacts with a clear diagnostic message.
- [ ] Successful full runs still produce a loadable baseline and compare behavior remains unchanged for valid reports.
- [ ] Regression tests cover both complete and incomplete baseline loading paths.

## Tests / verification
- Add focused tests near benchmarks/src/statistics.rs tests:
  - Loading an intentionally incomplete report is rejected (or treated as missing baseline).
  - Completed report remains accepted and comparison output works.
  - Interrupted-write simulation via partial artifact fixture.
- Run:
  - cargo test -p datafusion-benchmarks --lib statistics::tests -- --nocapture
  - cargo check -p datafusion-benchmarks --lib

## Notes / open questions
- Decide whether incomplete baseline should be a hard error or a warning plus no-baseline fallback.
- If status metadata is introduced, define compatibility behavior for legacy reports without metadata.
