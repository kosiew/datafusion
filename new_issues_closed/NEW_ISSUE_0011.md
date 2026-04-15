created https://github.com/apache/datafusion/issues/22622
source: pr-21666_a
# [Refactor] Introduce a build-report lifecycle handle for hash-join partitions

## Summary
Current behavior is correct and regression-covered, but build-report lifecycle ownership is still split across `HashJoinStream`, `OnceFut`, `Drop`, and shared accumulator state.

This issue tracks a maintainability refactor: introduce one lifecycle handle to centralize transitions and improve readability.

## Background and Motivation
#21666 addressed correctness around scheduled-vs-delivered reporting and drop-time cancellation. Lifecycle logic is still distributed across multiple components, which makes code review and future edits harder.

Coordination remains spread across:
- `HashJoinStream` transition logic
- lazy `OnceFut` execution/polling
- `Drop` cancellation behavior
- shared bounds/accumulator terminal state

This distribution increases cognitive overhead and regression risk during maintenance.

## Problem Statement
The stream-level state machine still carries too much lifecycle responsibility. Transition ownership is documented, but spread across several types.

We want one component to own lifecycle transitions and invariants for architectural clarity.

## Goals
- Centralize build-report lifecycle decisions in one abstraction.
- Preserve exactly one terminal outcome per partition report:
  - Delivered
  - Canceled
  - Finalized/no-op
- Make drop-time behavior deterministic and self-documenting.
- Reduce lifecycle-related branching in `HashJoinStream`.
- Improve local reasoning for future contributors.

## Non-Goals
- Re-opening already-fixed correctness bugs as part of this issue.
- Redesigning hash-join dynamic filtering behavior.
- Broad changes to join planner/optimizer behavior.
- Performance tuning outside lifecycle maintainability.

## Proposed Design
Introduce a dedicated lifecycle type, for example:
- `BuildReportHandle`
- `BuildReportLifecycle`
- `PartitionReportToken`

Responsibilities:
1. Track explicit state (`NotReported`, `Scheduled`, `Delivered`, `Canceled`, `Finalized`).
2. Own transitions (`schedule`, `mark_delivered`, `cancel_if_pending`, `finalize`).
3. Ensure terminal transition happens exactly once.
4. Provide drop-safe default behavior (`cancel_if_pending`) when dropped without delivery.
5. Expose minimal API to stream state machine so callers cannot bypass invariants.

Potential integration pattern:
- `transition_after_build_collected()` obtains or updates handle state.
- `wait_for_partition_bounds_report()` marks `Delivered` only after successful completion.
- `Drop` path delegates to handle `cancel_if_pending`.
- Shared bounds/accumulator interactions occur through handle entry points rather than direct bool checks.

This refactor must preserve current behavior and test outcomes.

## API and Invariants
Enforce with assertions/tests:
- A report cannot be both `Delivered` and `Canceled`.
- `Delivered` may only occur after successful waiter completion.
- `Drop` on `Scheduled` must attempt cancellation.
- Repeated terminal operations are idempotent and side-effect free.
- Transition intent is encoded in API shape, not scattered call-site conventions.

## Implementation Plan
1. Add lifecycle type in hash-join module with explicit state enum.
2. Replace direct stream-level lifecycle bookkeeping in `HashJoinStream` with lifecycle handle entry points.
3. Route report scheduling/completion/drop cancellation through lifecycle methods.
4. Keep external behavior unchanged.
5. Add focused unit/regression tests for transition edges.

## Testing Strategy
Required coverage:
- Existing regression behavior remains covered (drop after schedule, cancellation/idempotency).
- Successful waiter completion marks delivered exactly once.
- Pending scheduled report cancels exactly once on drop.
- Duplicate terminal operations are idempotent.
- Affected tests run under `force_hash_collisions`.

Suggested targeted commands (adjust test names as implemented):
- `cargo test -p datafusion-physical-plan hash_join -- --nocapture`
- `cargo test -p datafusion-physical-plan force_hash_collisions -- --nocapture`

## Risks and Mitigations
- Risk: introducing new lifecycle abstraction could accidentally change call ordering.
  - Mitigation: preserve behavior with transition-focused tests and no semantic expansion.
- Risk: hidden coupling with shared bounds accumulator.
  - Mitigation: migrate incrementally and verify dynamic-filter tests under both normal and collision configurations.

## Acceptance Criteria
- Issue is explicitly tracked as a refactor/maintainability task, not an open correctness bug.
- Lifecycle transitions are centralized behind a dedicated abstraction.
- Stream-level lifecycle branching is reduced and easier to follow.
- Existing regression scenarios (including drop-before-delivery behavior) remain covered and passing.
- Dynamic-filter hash-join tests pass in normal and `force_hash_collisions` modes.
- Code review confirms lifecycle semantics are self-documenting and localized.

## References
- Review notes in `PR_REVIEW_01.md`
- Related area:
  - `datafusion/physical-plan/src/joins/hash_join/stream.rs`
  - `datafusion/physical-plan/src/joins/hash_join/shared_bounds.rs`

