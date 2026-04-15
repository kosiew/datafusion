create #22665
source: pr-22068_a
# Centralize Aggregate FILTER Some(true) Semantics for Consistency and Correctness Hardening

## Summary
Aggregate FILTER semantics in DataFusion depend on a strict invariant: a row passes the aggregate filter only when the predicate evaluates to Some(true). The current codebase applies this logic in multiple places with partially duplicated bitmap handling, and some grouped aggregate paths still maintain local filter predicates instead of reusing a single shared implementation.

This issue proposes centralizing row-validity and filter-pass evaluation into a shared helper or iterator in functions-aggregate-common, then migrating grouped accumulators incrementally. This should be treated as semantic-consistency and correctness hardening work, not only technical debt cleanup.

## Problem
Filter-row handling is currently fragmented:
- #22068 introduced filter_to_validity and updated grouped-accumulator helper paths (including accumulate_multiple and filter_to_nulls plumbing).
- first_last.rs still uses local grouped FILTER checks instead of a shared Some(true) utility.
- accumulate_indices still repeats 64-bit validity iteration logic across multiple branches.

This creates three practical risks:
1. Semantic drift: Different code paths may interpret nullable boolean filters inconsistently.
2. Regression risk: Future accumulator changes can accidentally reintroduce NULL handling bugs.
3. Maintenance overhead: Bitmap logic is repeated and harder to audit for correctness.

## Why This Matters
SQL aggregate FILTER behavior is subtle but strict. A nullable predicate must include only rows where the predicate result is Some(true), and exclude both Some(false) and None. If this rule is encoded differently across code paths, grouped aggregates may diverge in observable query results under nullable predicates.

A single shared predicate/validity utility would encode this once and reduce the chance of behavior mismatch between grouped and non-grouped accumulator implementations. In practice, this is correctness hardening for nullable FILTER semantics with a refactor as the delivery mechanism.

## Proposed Direction
Introduce a shared utility in functions-aggregate-common that represents the canonical rule:
- Row passes aggregate filter if and only if filter result is Some(true).

Implementation options:
- A helper that returns effective validity for a boolean filter array.
- A row iterator/predicate adaptor that yields pass/fail decisions with NULL-aware semantics.

Then migrate consumers incrementally:
1. Keep existing correctness fixes as-is.
2. Switch first_last grouped paths to the shared utility.
3. Refactor accumulate_indices to reuse shared validity iteration and remove duplicated 64-bit loops.
4. Migrate other grouped accumulators over time where relevant.

## Scope
In scope:
- Shared helper or iterator in functions-aggregate-common.
- Targeted migration for first_last grouped filter checks.
- Targeted cleanup of duplicated validity loops in accumulate_indices.
- Focused tests that protect nullable FILTER semantics.

Out of scope:
- Broad allocator/performance rewrites unrelated to FILTER semantics.
- Large cross-workspace refactors beyond aggregate filter validity behavior.

## Acceptance Criteria
1. A single reusable utility encodes aggregate FILTER pass criteria as Some(true).
2. first_last grouped aggregate paths no longer use bespoke nullable filter checks.
3. accumulate_indices no longer has duplicated validity loop logic that can diverge semantically.
4. Existing aggregate FILTER tests continue to pass.
5. New tests cover nullable FILTER predicates for first_value and last_value grouped cases.
6. No behavior regression for non-null filter predicates.

## Testing Plan
- Add focused SQL logic tests for grouped first_value and last_value with nullable FILTER predicates.
- Include cases where predicate value-bit may be true but validity is null.
- Validate grouped result is null when no rows satisfy Some(true).
- Run relevant crate tests and sqllogictest aggregate coverage.

## Risks and Mitigations
Risk: Hidden behavior changes in paths that previously relied on local predicate handling.
Mitigation: Incremental migration with targeted tests before and after each migrated path.

Risk: Performance regression from abstraction changes.
Mitigation: Keep helper low-overhead and reuse bitmap-friendly iteration patterns; benchmark hot paths if needed.

## Estimated Effort
Medium.
- Initial shared utility: low to medium.
- Incremental migration and validation across grouped accumulators: medium.

## References
- Source review context: PR_REVIEW_01.md lines 21-26.
- Related blocking correctness discussion in the same review: first_last grouped FILTER handling under nullable predicates.

## GitHub-Ready Version

### Suggested Title
Centralize aggregate FILTER Some(true) semantics for consistency and correctness hardening

### Suggested Labels
- enhancement
- technical-debt
- aggregates

### Suggested Body
## Summary
Aggregate FILTER row-validity handling is currently fragmented across helper and grouped accumulator paths. We should centralize the invariant that a row passes aggregate FILTER if and only if the predicate is Some(true).

This is not only a technical-debt refactor. It is semantic-consistency and correctness hardening for nullable FILTER behavior.

## Current State
- `filter_to_validity` was introduced and grouped-accumulator helper paths were updated (including `accumulate_multiple` and `filter_to_nulls` plumbing).
- `first_last.rs` still uses local filter checks in grouped paths.
- `accumulate_indices` still repeats 64-bit validity iteration logic across multiple branches.

## Problem
Duplicated nullable-filter logic increases the risk of semantic drift and regressions (especially around NULL predicate handling), and makes validity bitmap code harder to maintain.

Because grouped and non-grouped paths can evolve independently, this drift can become user-visible as incorrect aggregate results under nullable predicates.

## Proposal
1. Add one shared helper/iterator in `functions-aggregate-common` that encodes:
	- row passes FILTER iff predicate is `Some(true)`
2. Migrate grouped `first_value` / `last_value` filter paths to this shared utility.
3. Refactor `accumulate_indices` to reuse shared validity iteration and remove duplicated loops.
4. Migrate remaining grouped accumulators incrementally.

## Scope
In scope:
- Shared FILTER validity/predicate utility.
- Targeted migration of `first_last` grouped paths.
- Targeted cleanup in `accumulate_indices`.
- Focused test coverage for nullable FILTER predicates.

Out of scope:
- Broad performance rewrites not tied to FILTER semantics.
- Large cross-workspace refactors.

## Acceptance Criteria
1. A single reusable utility encodes FILTER pass criteria as `Some(true)`.
2. Grouped `first_value` / `last_value` no longer use bespoke nullable filter checks.
3. `accumulate_indices` no longer carries duplicated validity loop logic that can drift semantically.
4. Existing aggregate FILTER tests pass.
5. New tests cover nullable FILTER predicates for grouped first/last behavior.
6. No regression for non-null filter predicates.

## Testing
- Add SQLLogicTests for grouped `first_value` and `last_value` with nullable FILTER predicates.
- Include cases where value bits may be true while validity is NULL.
- Validate result is NULL when no rows satisfy `Some(true)`.
- Run relevant aggregate crate tests and sqllogictest aggregate coverage.

## Notes
This is a high-impact consistency and correctness-hardening refactor that reduces long-term regression risk by encoding FILTER semantics once.
