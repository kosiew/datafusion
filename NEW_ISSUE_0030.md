source: nullability-mismatch-22034a
# Issue: Centralize recursive CTE schema derivation helpers

## Summary
Recursive CTE schema derivation is currently implemented in multiple layers:
- SQL work-table setup via `nullable_schema`
- Logical schema construction via `recursive_query_schema`
- Physical fallback schema construction via `recursive_query_output_schema`

This split increases the chance that recursive CTE behavior diverges across planning stages for names, types, nullability, or metadata.

## Motivation
The recursive CTE nullability direction introduced in issue-22034 tightened behavior toward a conservative contract. That contract is now represented by separate helper paths in SQL, logical planning, and physical fallback planning.

Even when current end-to-end tests pass, duplicated derivation logic creates a long-term maintenance risk:
- future changes may update one layer but not the others
- subtle schema drift can appear at logical/physical boundaries
- review burden increases because schema rules must be compared across files

A single shared contract would make recursive CTE semantics easier to reason about and safer to evolve.

## Problem Statement
Current recursive CTE schema derivation is decentralized:
- SQL planning creates a nullable work-table schema for self-reference binding.
- Logical planning computes recursive query output schema.
- Physical planning still has fallback output schema derivation.

Because these paths are separate, drift can occur in:
- field name authority (anchor/static vs recursive)
- type compatibility handling
- nullability widening behavior
- field and schema metadata preservation/intersection
- mismatch diagnostics and error messaging

The refactor opportunity is to ensure these layers use one consistent schema contract.

## Goals
- Define one authoritative recursive CTE schema derivation contract.
- Reuse that contract from SQL, logical, and physical planning paths.
- Preserve current approved behavior for recursive CTE output semantics.
- Reduce future schema divergence risk.

## Non-Goals
- No change to recursive CTE SQL syntax or user-facing feature scope.
- No broad planner architecture rewrite beyond scoped helper extraction.
- No performance optimization work beyond keeping current behavior neutral.

## Proposed Approach
1. Introduce a shared helper (or focused schema-contract module) for recursive CTE output schema construction.
2. Move common rules into that helper:
   - field count and type compatibility checks
   - output field naming authority
   - conservative nullability widening
   - metadata handling policy
3. Update call sites:
   - SQL recursive work-table setup
   - logical recursive query schema construction
   - physical fallback schema construction
4. Ensure physical alignment targets the same logical schema contract, rather than re-deriving independently.

## Design Constraints
- Keep crate boundaries clean and avoid introducing undesirable dependency edges.
- Retain existing error style and DataFusion schema-contract expectations.
- Preserve behavior for currently valid recursive CTE queries.

## Acceptance Criteria
- A single reusable recursive CTE schema derivation helper/module exists.
- SQL work-table setup, logical schema derivation, and physical fallback derivation consume the same contract.
- Names/types/nullability/metadata behavior is consistent across these layers.
- Existing recursive CTE tests continue to pass.
- New or updated tests explicitly guard against cross-layer drift.

## Testing Plan
Add focused coverage for:
- field name preservation across anchor/static and recursive terms
- conservative output nullability behavior
- metadata behavior for conflicting/mismatched metadata
- mismatched field counts and type errors
- logical vs physical schema parity for representative recursive CTE plans

Run at least:
- `cargo test -p datafusion-physical-plan recursive_query_exec`
- `cargo test -p datafusion-expr recursive_query --lib`
- `cargo test -p datafusion-sqllogictest --test sqllogictests -- cte`

## Risks and Mitigations
- Risk: helper extraction introduces crate coupling.
  - Mitigation: place helper at minimal shared layer and keep API narrow.
- Risk: behavior changes unintentionally during deduplication.
  - Mitigation: preserve existing tests and add parity-focused regression tests.
- Risk: metadata policy remains ambiguous.
  - Mitigation: encode policy explicitly in helper docs and tests.

## Expected Impact
- Lower probability of recursive CTE schema-contract regressions.
- Easier review and maintenance due to one source of truth.
- Better confidence that logical and physical planning remain aligned as recursive features evolve.

## Origin
Derived from PR review refactor opportunity in issue-22034:
- “Centralize recursive CTE schema derivation helpers”
- Effort estimated as small to medium, with high long-term consistency value.
