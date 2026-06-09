source: pr-22805_a
# Refactor: Avoid duplicate NLJ benchmark definitions

## Summary

DataFusion currently has two sources for the Nested Loop Join (NLJ) benchmark queries:

- SQL benchmark files under `benchmarks/sql_benchmarks/nlj/benchmarks/q*.benchmark`
- Inline Rust query definitions in `benchmarks/src/nlj.rs`

These definitions are intended to describe the same benchmark workload, but they now live in separate formats and are maintained independently. This creates drift risk for query text, labels, expected plan checks, and query numbering.

## Motivation

The PR adds SQL-based NLJ benchmark files so NLJ benchmarks can run through the generic SQL benchmark harness. This is useful because benchmark cases can be added without writing Rust code.

However, the existing Rust NLJ benchmark harness still contains an inline `NLJ_QUERIES` list with matching SQL. Keeping both copies means future changes must update two locations. If one copy changes and the other does not, benchmark results may no longer be comparable, and users may unknowingly run different workloads depending on which harness they choose.

## Current state

Relevant files:

- `benchmarks/sql_benchmarks/nlj/benchmarks/q01.benchmark` through `q17.benchmark`
- `benchmarks/src/nlj.rs`

Observed duplication:

- Query numbering `Q01`-`Q17`
- Query SQL text
- Query descriptions/comments
- Expected physical operator check (`NestedLoopJoinExec` in SQL benchmark files, equivalent plan string check in Rust harness)

## Problem

Two independent benchmark definitions make maintenance error-prone:

1. A benchmark query can be changed in one harness but not the other.
2. Comments and labels can drift from actual SQL.
3. Expected-plan validation can differ across harnesses.
4. Adding/removing/reordering NLJ cases requires duplicate edits.
5. Comparisons across old Rust output and new SQL benchmark output may become misleading.

## Proposed direction

Choose one source of truth for NLJ benchmark cases.

Preferred options:

### Option A: Use SQL benchmark files as source of truth

Retire or simplify the Rust inline `NLJ_QUERIES` list. The existing SQL benchmark harness already supports:

- benchmark grouping
- query naming
- expected plan checks
- streaming execution without buffering results
- Criterion integration

The Rust `nlj` command could either be removed if no longer needed, or changed to delegate to the SQL benchmark files.

### Option B: Generate SQL benchmark files from Rust metadata

Keep a structured Rust definition of NLJ cases and generate `.benchmark` files from it. This preserves compile-time structure but adds generation complexity. Generated files should be clearly marked and checked in only if required by workflow.

### Option C: Parse SQL benchmark files from the Rust NLJ harness

Keep the old command shape but load `benchmarks/sql_benchmarks/nlj/benchmarks/*.benchmark` instead of hard-coding SQL. This keeps the CLI-compatible path while removing query duplication.

## Recommended approach

Use Option A or C. Prefer SQL benchmark files as the canonical benchmark definition because the motivation is to make SQL-based benchmarks easy to add without Rust code changes.

If the standalone Rust `nlj` command is still useful for output format or compatibility, implement Option C and make it read the SQL benchmark files.

## Acceptance criteria

- NLJ benchmark SQL exists in exactly one canonical location.
- Running the generic SQL benchmark harness can still run all NLJ cases.
- If the standalone Rust NLJ benchmark command remains, it uses the same canonical query definitions.
- Expected plan validation for `NestedLoopJoinExec` remains enforced.
- Query names/order remain stable or migration notes are documented.
- Documentation/help text is updated if any command behavior changes.

## Suggested implementation steps

1. Decide whether `benchmarks/src/nlj.rs` should be removed, deprecated, or changed to load SQL benchmark files.
2. If keeping `benchmarks/src/nlj.rs`, replace `NLJ_QUERIES` with a loader that reads the canonical SQL benchmark files.
3. Preserve existing query selection behavior if possible (`--query 1`, etc.).
4. Ensure expected-plan validation still fails if a query stops using `NestedLoopJoinExec`.
5. Run targeted benchmark crate checks:
   - `cargo check -p datafusion-benchmarks`
   - relevant SQL benchmark parser/unit tests if touched
6. Update benchmark docs if user-facing commands change.

## Risks and considerations

- Existing users may rely on the Rust `nlj` command output format. Avoid removal without a clear migration path.
- SQL benchmark file parsing may include comments/directives that need careful handling if reused by Rust harness.
- Query numbering must remain deterministic for historical comparison.
- Keep benchmark validation lightweight; do not add result materialization for large NLJ outputs.

## Related context

This came from PR review for apache/datafusion#22805, which added SQL-based NLJ benchmark files while the existing Rust NLJ benchmark definitions remained in place.
