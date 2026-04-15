source: nullability-mismatch-22034a
# New Issue 01: Centralize Recursive CTE Schema Reconciliation

## Title
Centralize recursive CTE schema reconciliation to enforce a single schema contract across SQL, logical planning, and physical planning

## Summary
Recursive CTE schema handling is currently split across multiple crates and phases:
- `datafusion/sql/src/cte.rs` (`nullable_schema`)
- `datafusion/expr/src/logical_plan/builder.rs` (`recursive_query_output_schema`, `plan_with_schema`)
- `datafusion/core/src/physical_planner.rs` (`reconcile_logical_schema_nullability`)

The core contract is single and strict:
- Recursive work-table references and recursive query children must expose an output schema that is conservative on nullability.
- All non-nullability schema dimensions must remain exact (field count, field order, names, types, field metadata, schema metadata).

Because this logic is distributed, future edits can preserve only part of the contract and accidentally mask mismatches late in planning.

## Problem Statement
The current design allows nullability reconciliation to happen in different locations with different visibility into schema correctness. This increases the risk that:
1. Nullability widening is applied without proving all other schema dimensions match.
2. Mismatches are discovered too late (or are partially hidden) during physical planning.
3. Contract behavior diverges across planning layers as code evolves.

A concrete risk is that helpers which zip logical and physical fields can truncate on length mismatch and obscure earlier contract violations.

## Why This Matters
- Correctness: recursive schema contracts are foundational for planner and executor safety.
- Maintainability: one contract implemented three different ways invites regressions.
- Reviewability: centralized logic makes future changes easier to reason about and test.

## Scope
In scope:
- Introduce a small internal contract helper (or helper set) that computes recursive output schema and enforces schema invariants.
- Ensure nullability widening is the only permitted divergence.
- Rewire existing call sites to use the centralized contract path.
- Add targeted tests for exact schema dimensions.

Out of scope:
- Broad planner architecture redesign.
- Unrelated projection display/API improvements.

## Desired Contract (Authoritative)
For recursive CTE output schema reconciliation:
1. Field count must match exactly.
2. Field order must match exactly.
3. Field names must match exactly.
4. Data types must match exactly.
5. Field metadata must match exactly.
6. Schema metadata must match exactly.
7. Nullability may only widen conservatively (`false -> true`), never narrow (`true -> false`).

## Proposed Design Direction
- Create one internal reconciliation routine with explicit precondition checks for all non-nullability dimensions.
- If any non-nullability dimension differs, return a structured planning error (do not auto-reconcile).
- If all non-nullability dimensions match, apply nullability widening deterministically.
- Keep this contract helper close to recursive CTE planning code, then expose thin wrappers from other layers if needed.

## Affected Areas
- `datafusion/sql/src/cte.rs`
- `datafusion/expr/src/logical_plan/builder.rs`
- `datafusion/core/src/physical_planner.rs`
- Tests under recursive CTE planning and schema reconciliation paths.

## Acceptance Criteria
1. Recursive schema contract logic is centralized (single source of truth).
2. All non-nullability mismatches fail fast with clear errors.
3. Only nullability widening is reconciled automatically.
4. Unit coverage exists for positive and negative cases across:
   - field count mismatch
   - field name mismatch
   - data type mismatch
   - field metadata mismatch
   - schema metadata mismatch
   - valid nullability widening
5. Existing recursive CTE behavior remains correct for supported queries.

## Validation Plan
- Run targeted tests for crates touched by the implementation.
- Add/extend unit tests around recursive schema helper behavior.
- Run recursive CTE SQLLogicTests to verify end-to-end behavior.
- Confirm no regressions in schema-sensitive planner/executor tests.

## Risks and Mitigations
- Risk: over-centralization introduces cross-crate coupling.
- Mitigation: keep helper minimal, internal, and contract-focused.

- Risk: stricter checks surface latent mismatches and fail previously passing plans.
- Mitigation: add explicit error messages and targeted migration fixes where needed.

## Dependencies
- None required to open this issue.
- May overlap with existing recursive nullability fixes; coordinate sequencing to avoid duplicate work.

## Definition of Done
- A single reconciliation contract path is used for recursive schema handling.
- Non-nullability mismatches are no longer silently tolerated.
- Test coverage proves contract enforcement and nullability-widening behavior.
