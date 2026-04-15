 Verdict: Not high-impact refactor. Medium-risk consistency cleanup.

 Maybe worth only if:
 - helper from 0059 is tiny and clearly improves code,
 - migrate one function at a time,
 - no row-state abstraction from 0060,
 - strong SLT + unit parity tests.

 As written: too broad for payoff.

source: pr-22508_a
# Issue 03: Complete Migration Across remove, replace, repeat, resize with Contract Regression Coverage

## Summary
After introducing shared helper primitives for nullable counts and row-state branching, migrate all four nested array functions and add comprehensive regression coverage to ensure function-specific SQL null contracts remain unchanged.

## Background
A staged migration is safer than a one-shot rewrite. The end state should preserve existing behavior while removing duplicated control-flow and count extraction logic.

## Problem Statement
Without a dedicated migration and validation issue, partial adoption can leave inconsistent patterns in place and make future maintenance harder.

## Goals
- Migrate remove.rs, replace.rs, repeat.rs, and resize.rs to the shared helper(s).
- Preserve each function’s documented and observed SQL behavior.
- Add explicit regression coverage for null-contract edge cases.

## Non-Goals
- No expansion of feature scope.
- No unrelated performance tuning unless required for parity.

## Migration Scope
- remove.rs:
  - migrate nullable max/removal count handling.
  - preserve current null-row behavior for list and element/count interactions.
- replace.rs:
  - migrate nullable max replacement count handling.
  - preserve semantics for null from/to values.
- repeat.rs:
  - align repeat_count style handling with shared helper where practical.
  - preserve output null mask behavior based on nullable count input.
- resize.rs:
  - migrate nullable size row handling and row-state branching.
  - preserve default fill semantics and null propagation.

## Acceptance Criteria
- All four target files use shared helper(s) where applicable.
- No behavior changes in targeted sqllogictest files:
  - array_remove.slt
  - array_replace.slt
  - array_repeat.slt
  - array_resize.slt
- Function-level unit tests include mixed null and count edge cases.
- Reviewers can trace function-specific semantics clearly in each implementation.

## Validation Checklist
- Build and test touched crate(s).
- Run targeted sqllogictest files listed above.
- Add or update unit tests for each function’s null-contract edge cases.
- Confirm behavior parity for:
  - nullable count values,
  - zero and negative counts,
  - null input list rows,
  - mixed row validity.

## Risks and Mitigations
- Risk: Contract drift hidden behind abstraction.
  - Mitigation: Add explicit contract tests and keep helper boundaries strict.
- Risk: Multi-file migration becomes hard to review.
  - Mitigation: Land in small commits or PR slices by function.

## Suggested Rollout
1. Merge shared helper foundations.
2. Migrate remove and replace.
3. Migrate repeat and resize.
4. Run full targeted SQL and unit validation before final merge.
