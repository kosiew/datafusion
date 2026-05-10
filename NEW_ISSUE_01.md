-> copied to refactor
# Issue: Unify recursive CTE schema derivation contract

## Summary
Recursive CTE schema derivation is currently implemented separately in `datafusion/expr/src/logical_plan/plan.rs` and `datafusion/physical-plan/src/recursive_query.rs`. This split contract increases the risk of divergence in field metadata, field count handling, and type compatibility between logical and physical planning, which can cause subtle schema mismatches and incorrect query execution.

## Motivation
The existing PR review identified a schema-contract risk where logical and physical recursive CTE schema derivation disagree on metadata handling. Logical derivation intersects field and schema metadata across the static and recursive terms, while physical derivation retains static metadata only. This asymmetry can expose different schemas at the boundary between logical planning and physical execution, violating DataFusion's schema contract and potentially causing wrong results or planner/executor inconsistency.

## Problem Statement
- `recursive_query_schema` in `datafusion/expr/src/logical_plan/plan.rs` and `recursive_query_output_schema` in `datafusion/physical-plan/src/recursive_query.rs` compute output schema independently.
- The current logical and physical derivations differ in how they handle field metadata when static and recursive terms conflict.
- Divergent behavior can occur even if the field count and names align, because metadata intersections may differ.
- As the review notes, `RecursiveQuery::try_new` is used in other code paths such as proto deserialization and tree rewrites, so validations and contract enforcement must be centralized.

## Proposed Refactor
Introduce a shared recursive CTE schema contract helper or pass the expected logical schema through physical planning so that both phases derive the same schema from the same rules.

Possible approaches:
1. Extract a shared helper into a lower-level crate boundary used by both logical and physical recursion planning.
2. Modify physical planning to accept the already-derived logical recursive CTE schema and use it as the authoritative target for output schema alignment.

The shared contract should cover:
- Field count validation
- Type compatibility checks
- Nullability widening behavior
- Metadata intersection rules matching logical plan semantics
- Alignment of both recursive and static term children to the same widened, expected schema

## Acceptance Criteria
- A single recursive CTE schema derivation contract is defined and used by both the logical planner and physical recursive query planning.
- The contract preserves the current logical semantics for metadata intersection and output schema derivation.
- Physical planning no longer computes output schema independently in a way that can diverge from the logical plan.
- Direct `RecursiveQuery::try_new` and proto deserialization paths are covered by validation or clear contract enforcement.
- Regression tests cover:
  - conflicting metadata between static and recursive terms
  - mismatched recursive CTE column counts
  - mismatched recursive CTE types
  - physical vs logical schema equality for recursive queries

## Benefits
- Reduces schema-contract risk across the logical/physical boundary.
- Prevents silent schema mismatches in recursive CTEs.
- Makes recursive query planning more maintainable by centralizing contract logic.
- Provides stronger regression coverage for a known failure mode.

## Risks / Considerations
- Introducing a new shared helper may require careful crate dependency design to avoid undesirable new workspace dependencies.
- The chosen approach should preserve existing behavior for non-conflicting recursive CTEs while fixing divergence in metadata and schema alignment.
- Any change to schema derivation must be validated with appropriate recursive CTE SQL tests and schema-level planner/executor tests.
