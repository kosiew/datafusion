source: nullability-mismatch-22034a
# New Issue 03: Consolidate Recursive Output Schema Derivation Across Logical and Physical Plans

## Title
Consolidate recursive output schema derivation for DFSchema and Arrow SchemaRef into a shared contract implementation

## Summary
Recursive CTE output schema derivation is now implemented in two places with the same intended semantics:
- Logical layer: `datafusion/common/src/recursive_schema.rs` (`recursive_query_output_schema` on `DFSchema`)
- Physical layer: `datafusion/physical-plan/src/recursive_query.rs` (`recursive_output_schema` on Arrow `SchemaRef`)

This creates a cross-layer contract with duplicated logic. If these paths diverge on validation details (count, names, data types, field metadata, schema metadata, nullability widening), recursive behavior can become inconsistent and harder to audit.

## Problem Statement
Two separate implementations currently encode the same recursive schema invariant but operate over different schema types (`DFSchema` vs Arrow `SchemaRef`). This duplication increases risk of drift in:
1. Which dimensions are validated strictly.
2. How mismatches are surfaced.
3. How nullability widening is applied.
4. How metadata and qualifiers/functional dependencies are preserved.

Because recursive CTE schema behavior is a correctness contract spanning logical planning and physical execution, implementation drift can introduce subtle bugs that only appear at one layer.

## Why This Matters
- Correctness: recursive output schema must be identical in intent across planning and execution.
- Maintainability: a single canonical implementation reduces bug surface and review burden.
- Testability: shared logic makes it easier to add targeted contract tests once.

## Scope
In scope:
- Introduce a shared Arrow-schema helper that computes recursive output schema with explicit contract checks.
- Refactor physical recursive query code to call the shared helper.
- Wrap the Arrow helper for DFSchema usage in logical code, preserving qualifiers and functional dependencies.
- Add/adjust tests at both layers to prove contract alignment.

Out of scope:
- Broad redesign of recursive query planning.
- Changes to unrelated schema coercion behavior.

## Contract Requirements
The consolidated helper path should enforce:
1. Field count must match exactly between static and recursive branches.
2. Field order must match exactly.
3. Field names must match exactly.
4. Data types must match exactly.
5. Field metadata must match exactly.
6. Schema metadata must match exactly.
7. Output nullability is conservatively widened per field:
   - `output_nullable = static_nullable || recursive_nullable`
8. For DFSchema call sites, qualifiers and functional dependencies must be preserved.

## Proposed Design
1. Add a common Arrow helper in `datafusion/common` (or another shared crate agreed by maintainers), with a narrow API focused on recursive output schema derivation.
2. Make physical `recursive_output_schema` delegate to this helper.
3. Make DFSchema `recursive_query_output_schema` a wrapper:
   - convert/align to Arrow fields for contract computation
   - apply resulting nullability/field contract back to DFSchema
   - preserve qualifiers and functional dependencies
4. Keep error messages explicit so mismatches are diagnosable at review time.

## Affected Areas
- `datafusion/common/src/recursive_schema.rs`
- `datafusion/physical-plan/src/recursive_query.rs`
- Potential supporting modules where shared Arrow helper is introduced
- Tests for logical recursive schema derivation
- Tests for physical recursive query schema derivation

## Acceptance Criteria
1. Logical and physical recursive schema derivation use one shared Arrow-level contract implementation.
2. DFSchema wrapper preserves qualifiers and functional dependencies.
3. Validation behavior is identical across logical and physical call sites.
4. Nullability widening semantics are identical and tested.
5. Contract mismatch errors are deterministic and clear.

## Test Plan
- Add unit tests for shared helper covering:
  - valid nullability widening
  - field count mismatch
  - field name mismatch
  - type mismatch
  - field metadata mismatch
  - schema metadata mismatch
- Add logical-layer tests proving qualifiers and functional dependencies are preserved.
- Add physical-layer tests proving behavior is unchanged except for consolidation.
- Run targeted recursive CTE tests to ensure end-to-end stability.

## Risks and Mitigations
- Risk: moving helper to a shared crate introduces dependency layering concerns.
- Mitigation: keep API minimal and place helper in an already shared layer with no cyclic deps.

- Risk: DFSchema wrapper could accidentally drop logical-only attributes.
- Mitigation: add explicit tests for qualifiers and functional dependencies.

- Risk: stricter validation could expose previously hidden mismatches.
- Mitigation: treat this as positive hardening and provide actionable error text.

## Dependencies
- Naturally related to ongoing recursive CTE nullability/schema work.
- Can be implemented after immediate bugfixes land to reduce PR scope risk.

## Definition of Done
- Shared Arrow helper exists and is used by both logical and physical recursive schema derivation paths.
- DFSchema wrapper preserves logical schema-specific attributes.
- Tests demonstrate invariant parity across layers and prevent future drift.
