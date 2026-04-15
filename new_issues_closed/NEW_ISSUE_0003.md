created #21914
# Make spill codec availability an explicit contract of the spill stack

## Summary
Spill support currently depends on workspace-level Cargo feature unification (`arrow-ipc/zstd`) rather than an explicit crate-local contract near the spill implementation. This works today, but it is easy to regress during dependency cleanup or feature refactors.

This issue tracks making spill codec availability explicit and discoverable in the spill-owning crate(s) so future changes cannot silently break compressed spill behavior.

## Background
A recent fix (#21540) moved `arrow-ipc/zstd` into workspace dependencies to stabilize spill compression behavior across test configurations. That addressed the immediate failure mode, but the dependency remains implicit:

- Spill behavior relies on global feature wiring.
- The contract is not declared close to `SpillManager` and spill execution paths.
- Future feature edits may accidentally remove or bypass required codec support.

Related context:
- PR review for apache/datafusion#21504 identified this as a high-impact, medium-effort refactor opportunity.

## Problem Statement
The spill subsystem has a hidden dependency on codec availability. Because this dependency is not modeled as an explicit API/build contract in spill-related crates, the system is vulnerable to:

- Regressions in reduced-feature or crate-scoped builds.
- Non-obvious coupling across unrelated Cargo feature changes.
- Increased maintenance burden when diagnosing spill failures.

## Goals
- Define codec availability for spill as an explicit, local contract in the spill-owning crate(s).
- Reduce cross-crate implicit coupling for spill compression correctness.
- Improve maintainability and reviewer confidence for future dependency changes.

## Non-Goals
- Redesigning DataFusion-wide compression feature policy.
- Reworking non-spill Arrow IPC usage that is unrelated to spill files.
- Moving unrelated Arrow IPC feature declarations solely for consistency.

## Proposed Approach
1. Identify the spill-owned crate boundary and document where compressed spill files are produced/consumed.
2. Introduce an explicit spill codec contract at that boundary, for example:
  - A crate-local dependency declaration in the spill-owning crate that guarantees required codec availability for spill paths, or
  - A narrowly scoped spill-related feature on the spill-owning crate if a direct dependency contract is not sufficient.
3. Ensure the contract is visible and discoverable:
  - Add a concise dependency comment where the contract is declared.
  - Add a short implementation comment near the spill IPC writer if needed to explain why the dependency is local and intentional.
4. Add targeted tests that fail if spill codec availability is removed accidentally.
5. Validate at least one reduced-feature configuration that still exercises spill compression through the spill-owning crate.

## Acceptance Criteria
- Spill codec requirements are declared explicitly at the spill ownership boundary.
- Removing workspace-level feature unification alone cannot silently break spill compression without a clear compile/test signal.
- Targeted spill compression test coverage exists and passes in the spill-owning crate.
- At least one reduced-feature test path exercising spill compression through the spill-owning crate is documented and passing.
- Developer-facing comments make the local contract clear enough to prevent accidental cleanup regressions.

## Suggested Test Plan
- Run targeted spill test:
  - `cargo test -p datafusion-physical-plan test_spill_compression --lib`
- Run at least one reduced-feature configuration that still exercises spill compression behavior.
- Confirm no regressions in default workspace test behavior for spill-related paths.

## Risks and Mitigations
- Risk: Over-constraining dependencies across crates.
  - Mitigation: Keep the contract scoped to the spill ownership boundary only.
- Risk: Introducing feature complexity.
  - Mitigation: Prefer the smallest explicit contract that makes the spill dependency local and testable.

## Implementation Notes
- Keep the change crate-scoped and reviewable.
- Prefer an explicit crate-local contract over implicit fallback logic or broader workspace policy changes.
- Add tests with clear failure modes that pinpoint codec-contract regressions.

## References
- PR review: `PR_REVIEW_01.md`
- Source PR: apache/datafusion#21504