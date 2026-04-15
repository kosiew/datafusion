source: duplication-complexity-02-22669a
# Issue: Evaluate a Shared Row-Source Abstraction for Regex Scalar Functions

## Summary

Several regex scalar functions in `datafusion/functions/src/regex/` contain similar scalar-vs-array argument handling, row iteration, length validation, regex compilation, and cache lookup logic. The recent `regexp_count` refactor introduced private row-source helpers (`StringValueSource` and `StartValueSource`) that make this logic easier to follow for `regexp_count`, but those helpers should remain local until another regex function has matching semantics.

This issue proposes a follow-up evaluation: determine whether a shared regex row-source abstraction can safely serve multiple regex functions without changing SQL-visible behavior, especially around `NULL` handling and error ordering.

## Background

Regex functions commonly need to normalize `Datum` inputs into scalar-or-array sources and then process rows with combinations of:

- string input value;
- regex pattern;
- optional start position;
- optional flags;
- optional replacement or occurrence arguments, depending on function;
- regex compilation and cache reuse.

`regexp_count` now has a local abstraction for this shape:

- `StringValueSource<'a, S>` for scalar-or-array string arguments;
- `StartValueSource<'a>` for scalar-or-array start positions;
- centralized array length validation helpers;
- one row-processing loop.

This is useful for `regexp_count`, but promoting it immediately into shared regex infrastructure would be risky because each regex function has subtle, function-specific semantics.

## Problem

There is duplicated scalar/array branching logic across regex functions, but a premature shared abstraction could encode the wrong contract.

Known risks include:

- Scalar `NULL` regex handling differs by function or may have historically subtle behavior.
- Error ordering can be SQL-visible and is easy to change accidentally.
- Some functions compile scalar regex/flags before validating later array arguments; others may validate lengths earlier.
- Start-position validation, optional occurrence arguments, and replacement arguments may have different nullability rules.
- Regex cache behavior must avoid compile-per-row regressions for scalar pattern/flags cases.
- A shared abstraction might hide these differences instead of making each function's contract explicit.

## Goals

- Identify which regex functions have compatible row-source semantics.
- Preserve all SQL-visible behavior for each function.
- Preserve exact error messages and likely error ordering.
- Preserve regex cache reuse and scalar compile-once behavior.
- Reduce duplicated scalar/array branch code only where the same abstraction truly fits.
- Keep shared helpers small, private/module-local if possible, and contract-focused.

## Non-Goals

- No behavior changes to regex functions.
- No new regex flags or SQL features.
- No broad regex subsystem redesign.
- No public API changes.
- No dependency changes.

## Suggested Approach

1. Survey regex functions in `datafusion/functions/src/regex/`.

   Likely candidates:

   - `regexpcount.rs`
   - `regexpinstr.rs`
   - `regexplike.rs`
   - `regexpmatch.rs`
   - `regexpreplace.rs`

2. For each function, document its row-source contract:

   - Which arguments may be scalar or array?
   - Which arguments can be `NULL`?
   - What does scalar `NULL` regex do?
   - What does per-row `NULL` regex do?
   - Which array lengths are validated, and in what order?
   - When are scalar regex/flags compiled relative to length validation?
   - Which errors are SQL-visible and covered by tests?
   - When is the regex cache used?

3. Add characterization tests before refactoring any function.

   Tests should cover at least:

   - scalar invalid regex combined with mismatched array args;
   - mismatched regex/start/flags/replacement arrays where applicable;
   - scalar `NULL` regex with invalid later arguments;
   - per-row `NULL` regex with invalid later row values;
   - scalar flags vs array flags cache paths;
   - supported string types: `Utf8`, `LargeUtf8`, `Utf8View` where applicable.

4. Decide whether a shared abstraction is justified.

   Possible outcomes:

   - Keep `regexp_count` helpers private because contracts differ.
   - Extract a tiny shared primitive such as `validate_array_len` only.
   - Extract a shared `StringRowSource` but keep function-specific validation-order orchestration in each function.
   - Extract a more complete row-processing helper only for functions proven to have identical semantics.

5. If extracting, prefer an abstraction that exposes contract decisions instead of hiding them.

   For example, avoid a helper that always validates all array lengths up front if some functions must compile scalar regex first to preserve behavior.

## Acceptance Criteria

- A survey documents which regex functions can and cannot share row-source helpers.
- Any extracted helper has tests covering edge-case behavior and error ordering for all call sites.
- Existing unit tests pass.
- Existing regex SQLLogicTests pass.
- No SQL-visible behavior changes.
- No regex compile-per-row regression for scalar regex + scalar flags cases.
- Shared helper names reflect their exact contract, not a broader contract they do not guarantee.

## Files Likely Touched

Potential survey/refactor targets:

- `datafusion/functions/src/regex/mod.rs`
- `datafusion/functions/src/regex/regexpcount.rs`
- `datafusion/functions/src/regex/regexpinstr.rs`
- `datafusion/functions/src/regex/regexplike.rs`
- `datafusion/functions/src/regex/regexpmatch.rs`
- `datafusion/functions/src/regex/regexpreplace.rs`

Potential tests:

- unit tests in the same regex function files;
- `datafusion/sqllogictest/test_files/regexp/*.slt` if SQL-visible coverage is missing.

## Risks

Medium.

Why:

- Regex function error ordering is subtle and can be observable.
- `NULL` behavior differs between scalar and row-level arguments.
- A helper with the wrong validation timing can regress behavior while making code look cleaner.
- Regex compilation/cache behavior affects performance.

## Mitigation

- Do characterization tests first.
- Extract only the smallest helper that is proven compatible.
- Keep orchestration of validation and compilation order in the function unless all call sites match.
- Prefer private/module-local helpers over broadly reusable APIs.
- Review each function end-to-end, not just helper compatibility.

## Validation

Minimum targeted checks after any extraction:

```bash
cargo test -p datafusion-functions regex
```

Run relevant SQLLogicTests:

```bash
cargo test -p datafusion-sqllogictest --test sqllogictests -- regexp
```

If only one function is touched, a narrower SLT target is acceptable during iteration, followed by broader regex tests before merge.

## Rollback

Revert the helper extraction and restore each function's local scalar/array handling. No migration or external coordination needed.
