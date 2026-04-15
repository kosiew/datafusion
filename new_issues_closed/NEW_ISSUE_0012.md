create #22772
source: pr-21666_a
# [Follow-up Refactor] Centralize dynamic-filter expression policy outside shared accumulator finalization

## Summary
This is a follow-up refinement issue.

Recent changes already extracted key composition helpers in hash-join dynamic filtering (membership, bounds, and membership+bounds combination). However, `SharedBuildAccumulator::build_filter` still owns substantial expression-shape branching, fallback policy, and mode-specific assembly behavior.

This follow-up focuses on moving finalize-time expression policy into a dedicated composer/helper so accumulator finalization stays lifecycle-focused. The goal is not to force identical expression shapes for all modes, but to make the mode-specific rules explicit and centralized.

## Current State
What is already in place:
1. Helper-level extraction exists for:
   - membership predicate construction
   - bounds predicate construction
   - membership+bounds combination
2. Dynamic-filter update coordination remains in `SharedBuildAccumulator`.

What still needs refinement:
1. `CollectLeft` and `Partitioned` finalize paths still repeat the membership/bounds composition flow.
2. `build_filter` still mixes finalization/lifecycle concerns with expression-shape and fallback policy.
3. Test coverage in this area is currently stronger for lifecycle/reporting behavior than for expression-policy parity and edge cases.

## Problem Statement
The remaining coupling makes expression-shape changes riskier than necessary:
- mode-specific branching and fallback expression decisions are embedded in accumulator finalization
- composition behavior is partially duplicated between finalize paths
- there is no focused guard ensuring equivalent policy where inputs are equivalent, while preserving legitimate mode-specific differences

## Goals
- Extract finalize-time expression policy into a dedicated composer/builder layer.
- Keep `SharedBuildAccumulator` focused on synchronization, completion, and update orchestration.
- Remove repeated membership/bounds composition flow between `CollectLeft` and `Partitioned` paths.
- Keep partitioned routing, empty-partition handling, and canceled/unknown fallbacks explicit.
- Add explicit tests for policy parity and key edge cases.

## Non-Goals
- Changing dynamic-filter query semantics.
- Reworking hash-join state-machine behavior beyond this separation.
- Planner-level changes unrelated to dynamic-filter expression composition.

## Proposed Follow-up Design
Introduce a focused composition API for finalize-time expression assembly (name can vary, for example `DynamicFilterExprComposer` or `BuildSideFilterComposer`).

Responsibilities:
1. Accept normalized inputs per partition/finalize mode (membership, bounds, partition status metadata).
2. Produce final expression outcomes from one explicit policy surface for both `CollectLeft` and `Partitioned`.
3. Encapsulate fallback behavior (for example no-op, true/false branch defaults, case branch handling) in one place.
4. Preserve legitimate mode-specific output shapes, especially partitioned `CASE` routing.
5. Keep return values explicit for "no expression", "direct expression", and "composed/case expression" outcomes.

`SharedBuildAccumulator` should:
- collect and normalize finalize inputs
- call the composer
- apply `dynamic_filter.update(...)` when composition yields an expression
- avoid embedding expression-shape policy directly in finalization control flow

## Implementation Plan
1. Extract remaining mode-specific composition branches from `build_filter` into a helper/composer.
2. Refactor both `CollectLeft` and `Partitioned` finalize paths to delegate composition.
3. Keep state/lifecycle transitions unchanged.
4. Add tests that assert expression-shape parity and edge-case behavior.
5. Run targeted hash-join and sqllogictest coverage to confirm no semantic drift.

## Testing Strategy
Add focused tests for composition behavior:
- Policy parity test: equivalent logical inputs across `CollectLeft` and `Partitioned` follow equivalent membership/bounds/fallback rules, allowing expected mode-specific wrappers such as partitioned `CASE` routing.
- Edge cases:
  - missing membership predicate
  - missing/partial bounds
  - empty partition handling
  - canceled/unknown partition handling where relevant
  - no-branch fallbacks
- Keep existing lifecycle/reporting tests intact and passing.
- Ensure existing hash-join dynamic-filter and SQL behavior tests continue to pass unchanged.

Suggested targeted commands (adjust names as needed):
- `cargo test -p datafusion-physical-plan hash_join -- --nocapture`
- `cargo test -p datafusion --test sqllogictests -- hash_join`

## Risks and Mitigations
- Risk: expression-shape drift during extraction.
  - Mitigation: parity tests plus representative behavior checks.
- Risk: over-abstracting and reintroducing hidden coupling.
  - Mitigation: keep composer API explicit and near-pure with typed inputs.

## Acceptance Criteria
- `build_filter` no longer contains substantial expression-shape/fallback policy beyond delegating to the composer/helper and applying updates.
- Finalize-time expression composition policy is centralized behind a dedicated helper/composer API.
- New tests cover policy parity and edge-case composition behavior.
- Existing dynamic-filter behavior/tests pass unchanged.
- Readability improves by clearly separating synchronization logic from expression construction logic.

## References
- Related area:
  - `datafusion/physical-plan/src/joins/hash_join/shared_bounds.rs`
  - `datafusion/physical-plan/src/joins/hash_join/stream.rs`
