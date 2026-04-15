source: lifecycle-management-22622a
# Refactor Issue 01: Keep accumulator finalization as the single coordinator truth

## Source
- Review reference: PR_REVIEW_01.md, High-impact refactor opportunities, item 1

## Title
Keep accumulator finalization as the single coordinator truth

## Problem statement
The current hash join lifecycle intentionally splits responsibilities:
- BuildReportHandle in stream.rs manages per-stream, per-partition delivery and cancellation lifecycle.
- SharedBuildAccumulator in shared_bounds.rs owns terminal coordination across partitions, including Pending, Reported, CanceledUnknown, and finalization gating.

This split is correct for current scope, but the boundary is subtle and can drift over time. Future changes may accidentally migrate coordinator decisions into BuildReportHandle, creating dual sources of truth for completion and cancellation semantics.

## Why this matters
Finalization correctness is concurrency-sensitive and impacts dynamic filter readiness:
- A duplicate or misplaced terminal decision can hide coordinator bugs.
- Premature or duplicated completion may produce incorrect pushdown behavior.
- Fragmenting terminal logic across stream and accumulator increases maintenance risk and test blind spots.

## Current behavior snapshot
- BuildReportHandle:
  - Schedules one report future.
  - Marks delivery only after waiter completion.
  - Cancels pending partition report on drop (partitioned mode).
- SharedBuildAccumulator:
  - Stores partition states and completed counts.
  - Performs readiness checks for finalize input.
  - Executes final filter build and publishes completion.

This issue is about preserving and hardening that contract, not changing external behavior.

## Refactor goal
Strengthen and document the boundary so that SharedBuildAccumulator remains the only coordinator authority for terminal partition aggregation and finalization.

## Non-goals
- No behavior changes to hash join output.
- No redesign of dynamic filter expression building.
- No migration of partition terminal state machine into BuildReportHandle.
- No broad cross-crate API changes unless strictly required.

## Proposed refactor
1. Make ownership boundaries explicit in types and docs
- Add concise internal docs describing authority split:
  - BuildReportHandle: local stream lifecycle only.
  - SharedBuildAccumulator: global terminal coordination only.
- Add invariants near state transitions in both modules.

2. Prevent coordinator state from leaking into stream-side handle
- Keep BuildReportHandle states limited to local lifecycle markers.
- Avoid adding terminal aggregate counters, readiness checks, or finalization election logic to BuildReportHandle.

3. Consolidate terminal coordination contracts in accumulator
- Ensure all transitions affecting aggregate completion remain in SharedBuildAccumulator.
- Keep finalization eligibility checks centralized in accumulator readiness logic.

4. Add contract-focused tests
- Add or tighten tests proving that:
  - Stream-side cancellation only reports local canceled partition intent.
  - Accumulator remains sole decider for when finalization is triggered.
  - Duplicate local notifications do not create duplicate global terminal transitions.

5. Add lightweight guardrails for future contributors
- Add targeted comments in transition points describing why logic must stay centralized.
- Optionally add debug assertions where feasible to catch accidental authority drift during development.

## Design alternatives considered
1. Move terminal coordination into BuildReportHandle
- Rejected for now.
- Increases coupling between stream lifecycle and global coordinator logic.
- Raises regression risk by duplicating or relocating completion authority.

2. Introduce a new shared coordinator trait immediately
- Deferred.
- Could improve structure long-term but adds abstraction cost and migration risk now.

## Risks and mitigations
- Risk: Refactor accidentally alters ordering in async paths.
  - Mitigation: Keep behavior-preserving scope and use focused lifecycle tests.
- Risk: Over-documentation without enforcement.
  - Mitigation: Pair docs with tests that encode authority boundaries.
- Risk: Hidden dependency on existing state names.
  - Mitigation: Preserve semantics first, rename only when justified.

## Testing plan
- Run targeted hash join lifecycle tests in datafusion-physical-plan.
- Add tests that assert accumulator-side finalization remains single-path.
- Validate idempotency of repeated local cancellation or delivery notifications.
- Execute crate-scoped tests first, then expand only if boundaries are crossed.

## Acceptance criteria
- Coordinator authority remains centralized in SharedBuildAccumulator.
- BuildReportHandle does not own aggregate terminal coordination.
- Contract tests cover delivery, cancellation, and finalization boundary behavior.
- No SQL-visible behavior change for hash join.
- Existing targeted lifecycle tests pass.

## Rollout notes
- Land as behavior-preserving refactor.
- Keep diff scoped to hash join lifecycle files and tests.
- Defer larger architectural consolidation to a separate design proposal if still needed.

## Estimated effort
Medium to large, depending on test additions and documentation depth.
