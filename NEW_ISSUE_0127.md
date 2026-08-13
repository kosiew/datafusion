source: pr-23975_a
# Make SQL parser settings updates apply across query files in statistics runner

## Problem
The statistics runner snapshots sql_parser options once at startup and uses that snapshot to pre-parse every SQL file. If a suite executes session statements that change parser settings, later files are still parsed with stale settings.

Current behavior in benchmarks/src/statistics.rs:
- RunOpt::run clones parser options once from ctx.state().config_options().sql_parser.
- Each file is then parsed by sql_statements(sql, &sql_parser_options).
- Session statements that mutate parser configuration (for example SQL dialect or parser recursion limit) do not affect parsing of subsequent files.

This can fail valid suites where parsing requirements intentionally change over time, such as setting MySQL dialect before parsing MySQL-specific SQL in a later file.

## Why it matters
- Correctness: suite behavior diverges from session semantics.
- Predictability: statement execution and pre-parsing use different configuration timelines.
- Maintenance: users get parse failures that are hard to explain because runtime session state appears correctly updated.

## Invariant / desired behavior
Parsing of each query file must use parser settings derived from the current SessionContext state at the time that file is parsed.

If earlier statements modify parser settings, those modifications must be visible when parsing later files.

## Proposed direction
Apply parser-option lookup at file parse time instead of startup:
- Remove one-time snapshot of parser options in RunOpt::run.
- Before parsing each file, read current options from ctx.state().config_options().sql_parser.
- Parse that file with the fresh options.

Keep scope intentionally narrow:
- Preserve current flow of parsing a file into statements before executing statements from that same file.
- Do not introduce full statement-by-statement re-parsing within a file unless a concrete use case requires it.

Also add explicit docs/comments for the still-supported limitation:
- Intra-file parser-setting changes do not retroactively reparse later statements already parsed from the same file.

## Scope
### In
- File-level parser option refresh for statistics runner.
- Tests proving cross-file parser updates are honored.
- Brief documentation note on remaining intra-file limitation.

### Out
- Global parser architecture refactor.
- Incremental or streaming SQL parser for mid-file setting changes.
- Changes outside statistics benchmark runner.

## Acceptance criteria
- [ ] Parsing options are fetched from current SessionContext state before parsing each SQL file.
- [ ] A regression test demonstrates: file A sets parser dialect, file B contains dialect-specific SQL, and file B parses and executes under updated settings.
- [ ] A regression test demonstrates parser recursion-limit update in one file affecting parse behavior of a later file.
- [ ] Code or docs clearly state remaining limitation for intra-file parser-setting changes.

## Tests / verification
- Add focused tests near benchmarks/src/statistics.rs tests:
  - Cross-file dialect propagation test using temporary query files.
  - Cross-file recursion-limit propagation test.
- Run:
  - cargo test -p datafusion-benchmarks --lib statistics::tests -- --nocapture
  - cargo check -p datafusion-benchmarks --lib

## Notes / open questions
- Confirm whether any current benchmark suites intentionally depend on parser settings changing across files.
- Decide whether limitation documentation belongs in code comments, benchmark README, or both.
