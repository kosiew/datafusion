created #24098
source: pr-23861_a
# Align UNION child schemas during physical planning

## Problem
`UnionExec` derives one output schema from its children: field nullability and metadata are merged, while output field names follow the left input. It currently retains children with their original schemas, then `SchemaConformingStream` re-stamps every mismatched batch at execution time. `InterleaveExec` applies the same runtime repair after optimizer rewrites.

The output mismatch is knowable when the union-like plan is constructed. Keeping it as a stream wrapper leaves child plans whose declared schemas differ from the parent contract and requires every stream-combining execution path to preserve the repair.

## Why it matters
Plan-time alignment makes the contract explicit: every child handed to a union-like operator declares the canonical output schema that it emits. This:

- establishes `plan.schema() == child.schema() == emitted_batch.schema()` before execution;
- prevents `UnionExec` -> `InterleaveExec`, repartition, spill, or future stream combiners from bypassing local stream repair;
- removes normal-path per-batch `RecordBatch` re-stamping; and
- preserves exact Arrow schemas for Arrow C Stream / PyArrow consumers.

## Invariant / desired behavior
For every union-like physical plan, each child schema and every emitted batch schema exactly equal the operator's canonical output schema.

The canonical schema retains current `union_schema` semantics:

- field names come from the left input;
- nullability is widened when any input field is nullable;
- field metadata and schema metadata are merged as they are today; and
- compatible values, row counts, ordering, partitioning, cancellation, errors, and metrics are preserved.

A non-nullable child may be widened to the nullable canonical field. A nullable child must never be narrowed. Count or data-type mismatches remain construction errors.

## Proposed direction
Align children when constructing `UnionExec` and `InterleaveExec`, before exposing output properties. Use one shared helper so direct construction and optimizer rewrites follow the same rule.

`physical_plan::common::project_plan_to_schema` is the starting mechanism: it is already a no-op for exact schemas and uses a same-type `CastExpr::new_with_target_field(...)` to widen nullability. Extend its projection construction, or add a narrowly named union-schema alignment helper, so successful projections expose the canonical field names, nullability, field metadata, and schema metadata exactly. Do not reject metadata merely because inputs differ: that would regress existing `union_schema` metadata-merge behavior.

If a schema difference cannot be represented exactly while preserving current union semantics, identify and fix that representation boundary before replacing runtime conformance; do not silently drop metadata or introduce a new construction-time rejection.

Once constructors align every child, keep `CombinedRecordBatchStream` passive and remove `SchemaConformingStream` as the normal correctness mechanism.

## Scope
### In
- Align each `UnionExec` and `InterleaveExec` child to the computed canonical schema during construction.
- Reuse or extend a shared alignment path for exact safe schema alignment.
- Support nullability widening (`NOT NULL` -> nullable) with a same-type target-field cast.
- Preserve current field and schema metadata merge semantics in aligned child schemas.
- Preserve values, row counts, ordering, partitioning, cancellation, error propagation, and existing operator-metric semantics.
- Add direct and optimizer-rewrite regression coverage.

### Out
- Changing logical `UNION` type/nullability rules.
- Changing current union metadata merge semantics.
- Allowing nullable -> non-nullable coercion.
- Broad schema normalization for unrelated physical operators.
- Changing Arrow schema-equality requirements.

## Acceptance criteria
- [ ] `UnionExec::try_new` and `InterleaveExec::try_new` use one plan-time alignment path.
- [ ] Every constructed union-like child schema exactly equals the operator's canonical output schema.
- [ ] A non-nullable child is widened to a nullable canonical field without changing values or row count.
- [ ] Field names, field metadata, and schema metadata of successful aligned children exactly match the current canonical `union_schema` output.
- [ ] Count and data-type mismatches fail during construction as they do today.
- [ ] `CombinedRecordBatchStream` does not own schema-repair semantics; normal union-like execution needs no per-batch schema re-stamping wrapper.
- [ ] Direct `UnionExec`, direct `InterleaveExec`, and optimizer-created `InterleaveExec` emit only batches whose schemas exactly equal the plan schema.

## Tests / verification
- Unit tests for the shared alignment path: exact no-op, renaming, nullability widening, count/type mismatch, and merged field/schema metadata preservation.
- Physical-plan tests with non-empty batches for direct `UnionExec` and `InterleaveExec`; assert child schemas, plan schema, and every collected batch schema match exactly.
- Metadata regression with distinct input field/schema metadata; assert the aligned child and emitted schemas equal the existing `union_schema` merged output.
- Optimizer regression: force the `UnionExec` -> `InterleaveExec` rewrite and assert the same schema invariant.
- Existing SQL and spill/repartition mixed-nullability UNION regressions.

## Notes / open questions
- Confirm the smallest mechanism that gives `ProjectionExec` an exact target schema, including merged metadata, without changing array values. If the existing target-field cast cannot carry all schema metadata, extend that schema-construction boundary rather than rejecting compatible unions.
