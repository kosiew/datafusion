-> copied to refactor
# Issue: Centralize recursive CTE schema contract documentation

## Summary
Recursive CTE schema behavior is currently described in multiple places: `RecursiveQuery::try_new`, SQL two-pass recursive planning comments, physical recursive child alignment, and SLT/physical tests. This scattered contract increases the chance of inconsistent assumptions, makes the approved recursive CTE semantics harder to verify, and raises the risk of regressions as the recursive planner evolves.

## Motivation
The PR review for issue-22034 shows the contract has shifted from a static-output nullability model to a union-like nullability model. Without a single authoritative description, future changes can accidentally diverge in:
- logical planner nullability widening rules
- physical child alignment behavior
- field and schema metadata expectations
- name/position stability across anchor/static/recursive terms

A centralized contract will improve maintainability and make it easier to confirm that logical planning, physical execution, and tests express the same invariant.

## Problem Statement
- Recursive CTE semantics are documented in ad hoc comments rather than one canonical contract.
- `RecursiveQuery::try_new` enforces part of the contract, but the rationale is not captured in a shared place.
- `align_recursive_plan_to_schema` has a generic name and does not clearly document that the logical recursive CTE schema is authoritative.
- SLT and physical tests encode expected behavior, but they are not tied to a central schema contract comment.

Because the current approved contract is:
- anchor/static names are authoritative
- nullability is widened to a UNION-like output schema
- the logical recursive CTE schema is authoritative for physical alignment

this contract should be documented once and referenced everywhere.

## Proposed Refactor
1. Add a canonical recursive CTE schema contract comment in a shared location, ideally near the recursive planning helper or in a common module used by both logical and physical planning.
2. Rename the recursive physical alignment helper to make its intent explicit, e.g. `align_recursive_child_to_logical_schema`.
3. Update `RecursiveQuery::try_new`, SQL planner comments, physical helper comments, and SLT/physical tests so they all refer to the same documented contract.
4. Add or clarify section-level documentation describing:
   - how anchor/static names are preserved
   - how nullability is widened across recursive terms
   - that logical schema is authoritative for physical query execution
   - that physical children must be aligned to this logical recursive CTE schema
5. Optionally, link the contract comment to validation helpers such as `validate_recursive_schema_rebind` so the code and documentation remain in sync.

## Acceptance Criteria
- A single, explicit recursive CTE schema contract is documented in the repository.
- The contract is referenced by logical planner code, physical recursive query helpers, and regression tests.
- The physical helper name clearly communicates that it aligns recursive children to the logical recursive CTE schema.
- Existing semantics are preserved for valid recursive CTEs.
- Regression coverage is added or updated to protect the documented contract.

## Benefits
- Reduces the chance of future recursive CTE schema contract drift.
- Makes the intended semantics easier for maintainers and reviewers to understand.
- Improves test traceability by tying behavior to a single canonical contract.
- Supports safer refactoring of recursive query planning and execution.

## Risks / Considerations
- The refactor should avoid changing runtime behavior; it should focus on documentation and naming consistency.
- If the shared contract comment is placed in a lower-level crate boundary, dependency impact must be checked.
- The wording must be precise enough to capture the current approved semantics, especially around nullability widening and schema authority.
