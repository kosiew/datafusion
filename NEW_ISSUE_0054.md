source: pr-22295_a
# Issue: Centralize repeat size planning and capacity validation in array_repeat

## Summary
The current array_repeat implementation performs repeat-size planning, overflow checking, offset-bound validation, and allocation preparation across two separate paths:
- general_repeat for scalar/flat inputs
- general_list_repeat for List and LargeList inputs

Both paths enforce similar safety rules but do so with partially separate logic and error pathways. This creates maintenance risk: future changes may update one path but miss equivalent protections in the other.

## Background
In datafusion/functions-nested/src/repeat.rs, array_repeat has:
- a scalar path that computes running offsets and validates offset conversion
- a nested-list path that computes outer and inner totals, validates offset type bounds, and validates allocation capacities

Recent fixes strengthened overflow handling, but the core planning logic remains duplicated at a structural level.

## Problem Statement
Repeat planning and validation are distributed across multiple code regions rather than represented as one reusable planning contract.

As a result:
1. Safety invariants are encoded more than once.
2. Error behavior can drift between scalar and list variants.
3. Future refactors have a higher chance of reintroducing unchecked arithmetic or allocation attempts before returning a DataFusion error.

## Why This Matters
The safety-critical invariant for array_repeat should be explicit and centralized:
- No unchecked arithmetic.
- No offset conversion beyond Arrow offset type limits.
- No large allocation attempt before validation failure is surfaced as a DataFusion execution error.

When this invariant is fragmented, correctness depends on multiple implementations staying in lockstep.

## Proposed Direction
Introduce a small internal planning helper layer that computes and validates repeat sizes before materialization.

Potential structure:
1. Repeat planning input:
- count values (already normalized to non-negative usize)
- optional per-row element length for nested inputs
- offset type marker (i32/i64 via OffsetSizeTrait)

2. Repeat planning output:
- validated total repeated elements
- validated outer and inner offset totals where applicable
- pre-allocation capacities for offsets and indices

3. Shared validation guarantees:
- checked_add and checked_mul for all arithmetic
- early offset bound check through OffsetSizeTrait conversion guards
- guarded capacity preparation before Vec/bitmap allocation paths

This helper should be internal to functions-nested and only shape planning logic; output materialization can stay path-specific.

## Scope
In scope:
- Local refactor in datafusion/functions-nested/src/repeat.rs (or a small nearby private module)
- Replace duplicated planning/validation logic in general_repeat and general_list_repeat with shared helper usage
- Preserve existing user-visible behavior and error category

Out of scope:
- Functional changes to array_repeat semantics
- Broad cross-crate abstraction
- Unrelated nested function rewrites

## Acceptance Criteria
1. Both scalar and list paths consume shared planning logic for totals/capacities/offset-bound checks.
2. No unchecked arithmetic remains in repeat planning and capacity computation.
3. Validation occurs before large allocations in both paths.
4. Existing tests continue to pass.
5. New regression tests cover:
- scalar overflow-bound behavior
- List and LargeList offset-bound behavior
- allocation-bound behavior
- null and empty-row edge cases

## Test Plan
Add targeted tests near existing repeat coverage:
1. Unit tests in datafusion/functions-nested/src/repeat.rs for planning edge cases and error text expectations.
2. SQLLogicTests in datafusion/sqllogictest/test_files/array/array_repeat.slt for SQL-visible error behavior where appropriate.

Suggested minimum matrix:
- Scalar input with very large repeat count crossing i32 offset bound.
- List input with empty row and count above i32 max to trigger offset-bound guard without inner-value multiplication overflow.
- List input where inner length times count exceeds usize capacity path.
- LargeList input that should pass i32-bound scenarios but still respect usize/allocation limits.

## Risks and Mitigations
Risk: Refactor may accidentally alter error strings used in assertions.
Mitigation: Keep existing messages where feasible; update tests intentionally if any wording must change.

Risk: Over-generalized helper can reduce readability.
Mitigation: Keep helper small and focused on planning/validation only; leave materialization in current path-specific functions.

Risk: Performance regressions from extra abstraction.
Mitigation: Use simple structs and inlined helpers; avoid extra allocations in planning stage.

## Definition of Done
- Refactor merged with all relevant tests green.
- Repeat safety invariant documented in code comments at helper boundary.
- Coverage includes scalar, List, LargeList, offset-bound, and allocation-bound scenarios.
