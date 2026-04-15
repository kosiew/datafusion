created #22687
source: pr-22309_a
# Issue: Make Scalar Distance Overflow-Aware and Domain-Sized

## Summary
Scalar distance currently returns Option<usize>, but interval cardinality semantics are defined in terms of u64 domain counts. This introduces architecture-coupled behavior for large numeric ranges and requires downstream casting when a u64 semantic result is expected.

## Source
Derived from [PR_REVIEW_01.md](PR_REVIEW_01.md#L23).

## Problem Statement
The distance API in [datafusion/common/src/scalar/mod.rs](datafusion/common/src/scalar/mod.rs#L2566) uses usize as its numeric result type. usize is machine-width dependent and represents memory/index sizing, not value-domain cardinality.

A key consumer, [datafusion/expr-common/src/interval_arithmetic.rs](datafusion/expr-common/src/interval_arithmetic.rs#L916), computes interval cardinality by converting distance to u64 and then applying inclusive arithmetic. This cross-type flow can obscure boundary behavior and makes reasoning about full-domain ranges harder than necessary.

## Why This Matters
- Correctness clarity: Cardinality is a value-domain concept and should be expressed in a stable domain-sized type.
- Platform consistency: usize-based overflow/None behavior can differ across targets.
- Maintainability: Callers that need cardinality semantics should not need ad hoc conversions from usize.
- Boundary safety: Full-range integer and temporal cases are easier to reason about when distance and cardinality share an explicit overflow-aware contract.

## Current Behavior
- Scalar distance returns Option<usize> for integer, temporal, and float-derived distances at [datafusion/common/src/scalar/mod.rs](datafusion/common/src/scalar/mod.rs#L2566).
- Interval cardinality maps distance to u64 and then uses checked_add(1) in [datafusion/expr-common/src/interval_arithmetic.rs](datafusion/expr-common/src/interval_arithmetic.rs#L908).
- Stats overlap estimation also uses distance directly as usize at [datafusion/common/src/stats.rs](datafusion/common/src/stats.rs#L826).

## Desired Behavior
Distance calculations used for value-domain semantics should expose an overflow-aware u64-oriented contract (or a dedicated domain type), so large-range behavior is explicit and target-independent.

## Scope
Out of scope for the original PR.

In scope for this issue:
- Introduce either:
  - A revised scalar distance API that returns Option<u64>, or
  - A dedicated helper for domain cardinality distance (for example, for interval/statistics semantics) that returns Option<u64>.
- Update key consumers (interval cardinality and stats overlap paths) to use the new contract.
- Preserve existing user-visible behavior where representable, and make overflow-to-None rules explicit.

## Proposed Approach
1. Add a domain-oriented distance function in [datafusion/common/src/scalar/mod.rs](datafusion/common/src/scalar/mod.rs).
2. Keep existing usize API temporarily (if needed) as a compatibility shim and migrate internal callers.
3. Update interval cardinality implementation in [datafusion/expr-common/src/interval_arithmetic.rs](datafusion/expr-common/src/interval_arithmetic.rs).
4. Evaluate and update stats overlap estimation in [datafusion/common/src/stats.rs](datafusion/common/src/stats.rs).
5. Add boundary tests for full-domain integer/temporal ranges and overflow edges.
6. Remove or deprecate the usize-based path after call-site migration (as appropriate for semver policy).

## Acceptance Criteria
- Distance semantics for cardinality-sensitive paths are represented with an overflow-aware u64 contract.
- No behavior regression for currently representable ranges.
- Explicit tests cover:
  - Full signed range edge.
  - Full unsigned range edge.
  - Large temporal range edge.
  - Overflow-to-None behavior.
- Interval cardinality path no longer depends on usize-to-u64 conversion for domain semantics.
- Stats overlap path documents and tests its intended behavior under large ranges.

## Test Plan
- Unit tests in scalar distance module for boundary values.
- Unit tests in interval arithmetic for cardinality boundaries.
- Unit tests in stats overlap estimation for large range handling.
- Run targeted suites:
  - cargo test -p datafusion-common
  - cargo test -p datafusion-expr-common

## Risks and Mitigations
- API churn risk: Introduce migration helper/shim first, then migrate callers incrementally.
- Behavioral drift risk: Add explicit boundary regression tests before and after migration.
- Performance risk: Keep arithmetic simple and branch structure comparable; benchmark hot paths if needed.

## Notes
This issue tracks a high-impact refactor opportunity to improve semantic correctness and long-term robustness at numeric boundaries.