Verdict: Low–Medium cleanup, not high-impact.

 Better issue shape:
 - First centralize only nullable count access (NEW_ISSUE_0059 narrow form).
 - Avoid shared “row-state branching” until repeated bugs prove need.
 - If attempted, migrate only remove + replace; stop if readability worsens.

source: pr-22508_a
# Issue 02: Centralize Row-State Branching for Nested Array Row Loops

## Summary
Create a shared row-state branching utility for nested array functions that repeatedly implement the same row-loop structure: check list validity, check nullable count availability, and then choose between null-row output, pass-through/copy, or mutation logic.

## Background
Current row loops in remove.rs, replace.rs, repeat.rs, and resize.rs perform similar control flow:
- Validate row-level list/value nullability.
- Validate nullable count/size availability.
- Update offsets and null buffers.
- Invoke function-specific mutation behavior.

While each function has distinct SQL semantics, the structural branching logic is largely repeated.

## Problem Statement
Repeated row-state branching logic increases maintenance burden and the chance of inconsistent offset/null buffer handling, especially when new behavior is added or bugs are fixed in one place only.

## Goals
- Introduce a reusable row-state utility to reduce duplicated branching and bookkeeping.
- Keep function-specific operations pluggable and explicit.
- Improve readability and reviewability of row loops.

## Non-Goals
- No behavioral rewrite of array algorithms.
- No change to SQL-visible output semantics.
- No large cross-crate abstraction.

## Proposed Design
1. Add a small internal enum to represent row state, for example:
   - NullOutputRow,
   - ValidRowWithCount(i64),
   - ValidRowNoWork.
2. Add a helper that decides row state from:
   - list row validity,
   - optional count for the row,
   - any function-local preconditions.
3. Keep mutation/copy decisions in closures or caller-side match arms.
4. Reuse helper for offset/null bookkeeping patterns where safe.

## Acceptance Criteria
- At least two functions share the row-state utility.
- No SQL result changes in existing targeted tests.
- Offset and null-buffer handling remains correct for list and large list variants.
- Code complexity and duplication are visibly reduced in migrated functions.

## Test Plan
- Add focused unit tests for row-state helper transitions.
- Add regression tests for cases with:
  - null input rows,
  - null count rows,
  - zero and negative count rows,
  - mixed validity in the same batch.
- Run function-specific tests and targeted sqllogictest coverage.

## Risks and Mitigations
- Risk: Over-abstracting row behavior and obscuring intent.
  - Mitigation: Keep API minimal and local to functions-nested.
- Risk: Incorrect generalized offset handling.
  - Mitigation: Add explicit tests for empty rows, large counts, and mixed null patterns.

## Suggested Execution Order
1. Implement helper with tests.
2. Migrate two most similar functions first (remove and replace).
3. Evaluate readability and correctness before broader rollout.
