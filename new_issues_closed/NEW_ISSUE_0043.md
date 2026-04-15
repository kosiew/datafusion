created #22684
source: pr-22169_a
# Issue: Share post-aggregate clause rebasing and validation logic in SQL planner

## Summary
Refactor post-aggregate planning in `aggregate()` to remove duplicated rebasing and column-satisfaction validation logic across SELECT, HAVING, QUALIFY, ORDER BY, and DISTINCT ON. Introduce a shared helper abstraction that centralizes these checks while preserving clause-specific behavior and error messaging.

## Problem Statement
The aggregate planning path currently performs near-parallel steps for multiple SQL clauses:
- Rebase clause expressions against post-aggregate projection context
- Validate that expressions are satisfied by grouped/aggregated outputs
- Apply clause-specific error context

This logic is spread across multiple local code paths. As new clauses and edge cases were added (especially around DISTINCT ON), duplication increased and behavioral drift risk rose.

### Why this is a problem
- Makes behavior consistency harder to maintain across clauses with the same aggregate-boundary rules
- Increases likelihood of subtle regressions when one clause path is updated but others are not
- Raises maintenance cost for future SQL feature work in aggregate planning
- Complicates review and test coverage because equivalent rules are implemented repeatedly

## Goal
Create a shared internal utility for post-aggregate clause processing that:
- Reuses a single rebasing and validation pipeline
- Supports clause-specific context and diagnostics
- Keeps existing SQL semantics unchanged
- Makes future clause additions less error-prone

## Non-Goals
- Changing SQL semantics for SELECT, HAVING, QUALIFY, ORDER BY, or DISTINCT ON
- Large architectural rewrite of planner stages beyond this refactor boundary
- Reworking unrelated expression normalization paths

## Proposed Design
Introduce a small helper struct or function family in `datafusion/sql/src/select.rs` (or a nearby planner module) with responsibilities:

1. Accept inputs for a clause:
- Raw clause expressions
- Aggregate/grouping context
- Purpose tag (for clause-specific error text)
- Alias/substitution context where applicable

2. Execute common steps:
- Rebase clause expressions into post-aggregate context
- Validate aggregate-boundary satisfaction rules
- Return rebased expressions with normalized metadata needed by clause consumers

3. Keep clause-specific behavior pluggable:
- ORDER BY and DISTINCT ON alias handling remains clause-aware
- Error messages retain clause-specific wording
- Existing planner output shape remains unchanged

## Expected Benefits
- Reduces divergence between clause paths that should follow identical aggregate-boundary rules
- Improves correctness confidence by concentrating critical logic in one place
- Lowers implementation effort for future post-aggregate clause additions
- Simplifies code review by reducing repeated planning patterns

## Implementation Tasks
1. Identify duplicated post-aggregate rebasing/validation code paths in `aggregate()`.
2. Extract shared helper abstraction with explicit input/output contract.
3. Migrate SELECT, HAVING, QUALIFY, ORDER BY, and DISTINCT ON call sites to helper.
4. Preserve clause-specific diagnostics and alias-substitution semantics.
5. Add/adjust tests to verify behavior parity and no regressions.
6. Run focused SQL logic tests for clause interactions around aggregation.

## Acceptance Criteria
- One shared utility is used by all targeted post-aggregate clause paths.
- Existing behavior for valid and invalid queries is unchanged.
- Clause-specific error purpose/context remains clear and correct.
- No new regressions in aggregate + ORDER BY / DISTINCT ON / HAVING / QUALIFY interactions.
- Relevant tests pass in CI-equivalent local runs.

## Testing Plan
- Add or update unit tests covering helper behavior and clause-specific adapters.
- Add SQL logic tests for representative combinations:
  - GROUP BY with SELECT + HAVING
  - GROUP BY with QUALIFY
  - GROUP BY with ORDER BY alias and non-alias expressions
  - GROUP BY with DISTINCT ON and ORDER BY interactions
- Re-run targeted sqllogictest subsets for DISTINCT ON and aggregation scenarios.

## Risks and Mitigations
- Risk: Behavior drift during extraction
  - Mitigation: Migrate incrementally and maintain parity tests per clause
- Risk: Loss of clause-specific diagnostics
  - Mitigation: Carry explicit clause-purpose metadata through helper API
- Risk: Over-generalized helper becomes hard to follow
  - Mitigation: Keep helper narrow, strongly typed, and planner-local

## Suggested Labels
- `refactor`
- `sql-planner`
- `good-first-followup` (optional, if scope remains contained)

## Source
Derived from the refactor opportunity documented in [PR_REVIEW_01.md](PR_REVIEW_01.md#L24-L28).
