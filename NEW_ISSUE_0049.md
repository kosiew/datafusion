source: pr-22444_a
# Refactor eliminate_outer_join null-rejection tracking to side-level state

## Summary
The eliminate_outer_join rule currently computes null-rejection evidence as a vector of Column values, then later reduces that information to side-level booleans (left_non_nullable, right_non_nullable). This introduces avoidable indirection and repeated schema membership checks, and it obscures the actual invariant the optimizer needs: whether the predicate null-rejects the left side, the right side, or both.

This issue proposes refactoring the helper return contract from "collected columns" to "null-rejecting join sides".

Current-codebase check: this remains a valid refactor opportunity. `datafusion/optimizer/src/eliminate_outer_join.rs` still builds `null_rejecting_cols: Vec<Column>` in `try_simplify_join`, calls `extract_null_rejecting_columns(...)`, and then reduces the collected columns to `left_non_nullable` / `right_non_nullable` with repeated `schema.has_column(col)` checks. The OR / nested-AND path also still pushes representative columns per side, even though the downstream decision only needs side-level booleans.

## Current Behavior
In datafusion/optimizer/src/eliminate_outer_join.rs:

- extract_null_rejecting_columns appends Column values into Vec<Column>.
- Callers do not use specific column identity for join elimination decisions.
- Callers iterate the collected vector and repeatedly test schema membership (has_column) to derive side booleans.
- OR and nested-context logic pushes one representative column per side, which works but is indirect and can be ambiguous about which exact representative was chosen.

## Problem
The current representation is more general than required and leaks implementation details into decision logic:

1. Mismatch between data model and invariant
The optimizer decision only depends on side-level null rejection, not specific columns.

2. Repeated work
Converting Vec<Column> to side booleans requires repeated has_column scans over join schemas.

3. Ambiguous intent in OR handling
When one representative column per side is pushed, behavior is side-driven, but encoded as column-driven data.

4. Harder to reason about contracts
Tests and maintenance require mentally mapping from "which columns were collected" to "which join sides are null-rejected".

## Proposed Change
Introduce a compact side-level structure and use it end-to-end in eliminate_outer_join analysis.

The refactor should be treated as behavior-preserving cleanup, not a semantic change. In particular, preserve current handling for ambiguous or qualified columns by keeping side membership checks equivalent to the existing `DFSchema::has_column`-based behavior.

Suggested shape:

- struct NullRejectingSides { left: bool, right: bool }

Suggested API direction:

- Replace extract_null_rejecting_columns(...) with a helper that computes/merges NullRejectingSides.
- Keep existing semantic behavior unchanged.
- Preserve top-level AND, OR-side intersection behavior, and top_level guards for IS NOT NULL / IS TRUE / IS FALSE / IS NOT UNKNOWN handling.

Possible merge helpers:

- and_merge: union of side rejection signals where appropriate.
- or_merge: side is null-rejecting only if both branches reject null on that side.

## Non-Goals
- No behavior change for join elimination outcomes.
- No cross-optimizer redesign outside eliminate_outer_join.
- No SQL-visible semantic changes.

## Acceptance Criteria
1. Behavior parity
- Existing eliminate_outer_join tests continue to pass without changed expected plans unless mechanically required by refactor.

2. Representation cleanup
- Join elimination decision path no longer depends on Vec<Column> as intermediate state.
- Side-level null rejection is explicit in code.

3. OR semantics preserved
- Branch-combination logic preserves current side-level behavior for OR and nested AND contexts.
- Top-level `AND` should continue to union null-rejection evidence across conjuncts.
- `OR` and nested `AND` should continue to intersect null-rejection evidence per side: a side is credited only when both branches independently reject NULLs from that side.

4. Readability improvement
- Code makes it obvious that join conversion depends on side-level null rejection, not specific column identity.

## Suggested Test Strategy
1. Keep and run current unit suite:
- cargo test -p datafusion-optimizer eliminate_outer_join --lib

2. Ensure coverage for side-sensitive cases already present in tests:
- LEFT/RIGHT/FULL conversions under null-rejecting predicates
- No-conversion cases under null-accepting predicates (for example NOT(IS TRUE)-family)
- OR predicates that reject on one or both sides
- Top-level guards for `IS NOT NULL`, `IS TRUE`, `IS FALSE`, and `IS NOT UNKNOWN`

3. Optional follow-up hardening:
- Add focused unit tests that assert side-level merge behavior directly if helper extraction makes this practical.

## Impact
- Maintains current optimizer behavior.
- Reduces conceptual and implementation complexity in null-rejection tracking.
- Makes future changes less error-prone by encoding the true invariant directly.

## Origin
Pre-existing design opportunity identified during PR review for apache/datafusion PR #22444.
