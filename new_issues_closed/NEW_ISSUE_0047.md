Overall: No, not a high-impact refactor issue as written.
 Split into 2 issues. Reserve helper = small cleanup. Disk helper = useful test
 hardening.

source: pr-22386_a
# Issue: Centralize fallible Vec reservation error mapping

## Summary
Multiple call sites in the codebase now use the same fallible reservation pattern:
try_reserve followed by map_err to wrap allocation failures in exec_datafusion_err.
This is the correct behavior for the new disallowed reserve lint, but the mapping logic and message construction are duplicated across modules.

This issue proposes introducing a small shared helper to reduce duplication and make the reserve-failure invariant easier to apply consistently as more reserve call sites are migrated.

## Context
The workspace lint policy disallows Vec::reserve in favor of fallible reservation:
- [clippy.toml](clippy.toml#L4)

Recent call sites demonstrate consistent but repeated patterns:
- [datafusion/functions-aggregate/src/median.rs](datafusion/functions-aggregate/src/median.rs#L286)
- [datafusion/functions-aggregate/src/percentile_cont.rs](datafusion/functions-aggregate/src/percentile_cont.rs#L425)
- [datafusion/physical-plan/src/recursive_query.rs](datafusion/physical-plan/src/recursive_query.rs#L496)
- [datafusion/spark/src/function/math/hex.rs](datafusion/spark/src/function/math/hex.rs#L185)
- [datafusion/spark/src/function/math/unhex.rs](datafusion/spark/src/function/math/unhex.rs#L131)

## Problem Statement
The current pattern is semantically correct but repeated in multiple crates:
1. It increases boilerplate at each new conversion site.
2. It risks subtle drift in error message quality and format.
3. It increases reviewer effort because each call site must independently validate equivalent error mapping behavior.

## Why This Matters
As disallowed reserve migrations continue, repeated map_err blocks will spread across more modules.
Without a shared helper, consistency and maintainability degrade over time, and future updates to error wording or reserve-failure handling become costly.

## Proposed Solution
Add a small helper in a shared crate (likely datafusion_common) to wrap try_reserve failures into DataFusion errors with a standard shape.

Example direction (illustrative, not final API):
1. A helper that accepts:
- additional reservation size
- operation context string
- reserve error
2. Returns a DataFusion error with standardized wording.

Potential API shape options:
1. Function-style helper:
- reserve_error(context, additional, err) -> DataFusionError
2. Convenience wrapper:
- try_reserve_with_context(vec, additional, context) -> Result<()> 

The final API should prioritize:
1. Clear call-site readability
2. Minimal generic complexity
3. Compatibility across crates using datafusion_common

## Scope
In scope:
1. Introduce one reusable helper for try_reserve error mapping.
2. Update the newly added repetitive call sites to use the helper.
3. Keep behavior unchanged except for any intentional error-message normalization.

Out of scope:
1. Converting every existing reserve-related path in the workspace.
2. Broad error taxonomy redesign.
3. Functional behavior changes unrelated to allocation failure reporting.

## Acceptance Criteria
1. A shared helper exists in an appropriate common crate.
2. The five identified call sites are migrated to the helper.
3. All updated paths continue to return actionable allocation-failure errors.
4. No new lint violations are introduced.
5. Existing tests pass, and any affected tests are updated only if message text is intentionally standardized.

## Risks and Mitigations
1. Risk: Over-generalized helper harms readability.
   Mitigation: Keep helper API narrow and focused on reserve-failure mapping.
2. Risk: Message standardization breaks tests expecting exact strings.
   Mitigation: Use stable, intentional message format and update assertions where appropriate.
3. Risk: Wrong crate placement creates dependency friction.
   Mitigation: Place helper in the lowest shared crate already used by target call sites.

## Suggested Implementation Steps
1. Identify common crate location and finalize helper signature.
2. Add helper and unit tests for formatting behavior.
3. Migrate the five known call sites.
4. Run targeted crate tests first.
5. Run workspace lint checks relevant to disallowed methods.

## Verification Plan
1. Build and test touched crates.
2. Confirm all migrated call sites compile and preserve error propagation behavior.
3. Run lint checks to ensure reserve usage policy remains enforced.

## References
- Source review note: [PR_REVIEW_01.md](PR_REVIEW_01.md#L26)
- Reserve policy: [clippy.toml](clippy.toml#L4)
# Issue: Add Real Spill Accounting Test Helpers for Disk Usage

## Summary
The new dynamic-limit tests in [datafusion/execution/src/disk_manager.rs](datafusion/execution/src/disk_manager.rs) validate behavior by directly mutating internal atomics, rather than exercising the real spill accounting path through temp file writes and update_disk_usage. This leaves a coverage gap for regressions in file lifecycle accounting.

## Source
- Review file: [PR_REVIEW_01.md](PR_REVIEW_01.md)
- Review section: High-impact refactor opportunities (out of scope)

## Problem Statement
Current dynamic-limit tests simulate usage with direct atomic mutations:
- used_disk_space.store(...)
- used_disk_space.fetch_add(...)
- used_disk_space.fetch_sub(...)

These operations bypass the production accounting flow that runs when spill files are written and resized.

The real accounting contract depends on:
1. Writing data to RefCountedTempFile
2. Calling update_disk_usage to reconcile file-local and global counters
3. Dropping the last file reference to release global usage

Because current tests skip that flow, they can miss defects where global counters become inconsistent with file-local usage, especially around failed updates and cleanup paths.

## Why This Matters
- Higher confidence in spill accounting invariants under dynamic limit changes
- Better protection against leaks where used_disk_space remains inflated after failures
- Tests align with real system behavior instead of implementation shortcuts
- Reduces risk of production regressions when update_disk_usage or Drop logic changes

## Scope
In scope:
- Add small reusable test helper(s) in the disk_manager tests module
- Use real temporary file writes and update_disk_usage calls
- Validate both per-file and global accounting
- Add at least one regression-style test that would fail for accounting drift

Out of scope:
- Refactoring spill writer internals
- Changing public APIs unrelated to testability
- Performance optimization of disk spill paths

## Proposed Test Helper Design
Create a helper that performs the full accounting path:
1. Create temp file through DiskManager
2. Write provided bytes to file
3. Call update_disk_usage
4. Return the file handle and observed usage values for assertions

Optional companion helper:
- Assert global usage matches expected sum across active files

## Acceptance Criteria
1. New helper(s) are added under tests in [datafusion/execution/src/disk_manager.rs](datafusion/execution/src/disk_manager.rs).
2. At least one dynamic-limit test uses helper(s) instead of direct atomic mutations.
3. A regression test covers this sequence:
   - create and account spill file
   - lower limit below current usage
   - attempt update_disk_usage that fails
   - drop file
   - assert used_disk_space returns to zero
4. Existing disk manager tests continue to pass.

## Suggested Test Cases
1. Real write and accounting roundtrip
- Write bytes, call update_disk_usage, assert file usage equals global usage, drop file, assert zero.

2. Failed update cleanup regression
- Write initial bytes and account.
- Lower limit below current usage.
- Grow file and call update_disk_usage expecting error.
- Drop file and assert global usage returns to zero.

3. Multi-file accounting consistency
- Account two files through helper.
- Assert global usage equals sum of file usages.
- Drop one file and verify remaining usage.

## Risks and Mitigations
- Risk: Test flakiness from filesystem timing.
  Mitigation: Keep writes small, call update_disk_usage after each write, avoid timing assumptions.

- Risk: Over-coupling tests to internal implementation.
  Mitigation: Assert external behavior via public methods and file lifecycle outcomes.

## Implementation Notes
- Prefer helper reuse to keep tests concise and readable.
- Keep helper local to test module until reused elsewhere.
- Use deterministic assertions on accounting values and cleanup behavior.

## Definition of Done
- Helper(s) merged with test updates
- Regression path is covered by a real temp-file-based test
- No direct atomic mutation in dynamic-limit behavior tests unless explicitly justified in comments
