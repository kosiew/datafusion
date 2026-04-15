created #22669
source: pr-22311_a
# Issue: Consolidate regexp_count_inner Scalar/Array Branch Handling

## Summary
`regexp_count_inner` contains 8 branch combinations for `(is_regex_scalar, is_start_scalar, is_flags_scalar)` and repeats similar logic across branches for null handling, length checks, regex compilation/cache usage, and match counting. This duplication increases maintenance cost and raises regression risk when semantics change.

## Why This Matters
- Similar bug fixes must be applied in many places.
- Behavioral updates can be implemented in only some branches by accident.
- Duplicated branch logic is harder to review and reason about than a normalized data flow.

## Current Pain Points
In [datafusion/functions/src/regex/regexpcount.rs](datafusion/functions/src/regex/regexpcount.rs), the `match (is_regex_scalar, is_start_scalar, is_flags_scalar)` block duplicates:
- regex null-to-zero behavior
- validation of aligned array lengths
- regex compilation/caching call patterns
- `count_matches` invocation wiring

Even when behavior is conceptually identical, implementation is split across multiple arms.

## Desired Outcome
Refactor `regexp_count_inner` to centralize shared behavior while preserving existing SQL and unit-test semantics.

Potential directions:
1. Add helper(s) to resolve per-row `(regex, start, flags)` from scalar/array sources.
2. Normalize inputs into iterators of row views, then apply one counting pipeline.
3. Extract repeated null/length validation into dedicated helpers with consistent error messages.

## Constraints
- No user-visible behavior changes.
- Keep current error contract unless explicitly updated with test changes.
- Preserve performance characteristics (including regex cache reuse behavior).

## Proposed Implementation Plan
1. Introduce a small internal abstraction for obtaining per-row values:
   - regex value (nullable)
   - start value (defaulting to 1 when absent)
   - flags value (nullable)
2. Move all length compatibility checks into one place.
3. Implement one loop that:
   - handles null regex as zero
   - compiles/caches regex with flags
   - delegates counting to `count_matches`
4. Keep data-type specific dispatch (`Utf8`, `LargeUtf8`, `Utf8View`) unchanged.

## Acceptance Criteria
- Existing tests in `regexpcount.rs` pass without semantic updates.
- Existing SLT results for `regexp_count` remain unchanged.
- Internal duplication in branch handling is reduced materially (fewer large match arms).
- Error messages for invalid lengths/start/flags remain stable unless intentionally changed and covered by tests.

## Suggested Verification
- `cargo test -p datafusion-functions regexp_count`
- `cargo test --test sqllogictests` from `datafusion/sqllogictest` (or targeted regexp SLT scope if available)

## Out of Scope
- New SQL features for regex flags.
- Broader regex subsystem redesign outside `regexp_count`.

## References
- PR review note: [PR_REVIEW_01.md](PR_REVIEW_01.md)
- Target file: [datafusion/functions/src/regex/regexpcount.rs](datafusion/functions/src/regex/regexpcount.rs)
