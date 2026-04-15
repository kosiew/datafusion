source: nullability-mismatch-22034a
# Issue: Store derived recursive CTE schema directly on `RecursiveQuery`

## Summary
`RecursiveQuery` currently derives the recursive CTE output schema and then encodes that schema indirectly by rewriting `static_term` (via schema alignment / projection wrapping). `LogicalPlan::schema()` for `LogicalPlan::RecursiveQuery` then returns `static_term.schema()`.

This works today, but it makes schema ownership implicit in a child plan rather than explicit on the owning logical node. The refactor is to store the derived output schema directly on `RecursiveQuery` itself and make that schema the canonical contract for planning, rewrites, and serde reconstruction.

## Motivation
The current shape has a hidden invariant:
- The recursive query output schema is not a dedicated field on `RecursiveQuery`.
- The schema is represented by ensuring `static_term` has been aligned to the derived recursive schema.
- Consumers rely on this child rewrite (`static_term.schema()`) as if it were node-owned schema.

That implicit ownership has practical costs:
- It introduces schema-only projection churn when alignment is needed solely to carry output schema.
- It couples schema correctness to child rewrite behavior in tree transformations.
- It makes proto/tree reconstruction depend on replaying child alignment correctly.
- It makes the structural contract harder to reason about and validate in isolation.

Making schema ownership explicit on `RecursiveQuery` aligns with the stronger direction captured in DIRECTION_02 and reduces long-term drift risk between logical shape and schema contract.

## Current State (Verified)
1. `RecursiveQuery` has fields: `name`, `static_term`, `recursive_term`, and `is_distinct`.
2. `RecursiveQuery::try_new` derives schema, then aligns/wraps `static_term` to that schema.
3. `LogicalPlan::schema()` for `LogicalPlan::RecursiveQuery` returns `static_term.schema()`.
4. Reconstruction paths (optimizer rewrites, tree transforms, proto deserialization) call `RecursiveQuery::try_new`, which re-applies this alignment logic.

## Problem Statement
Schema ownership for recursive CTEs is currently encoded as a side effect on a child node. This creates avoidable fragility across plan construction and reconstruction paths:
- Schema contract is implicit rather than explicit on the node that semantically owns it.
- Rewriters must preserve child alignment details even when they only intend structural changes.
- Serde / roundtrip paths can accidentally diverge if child rewrite behavior changes.
- Debuggability suffers because node-level schema is not directly inspectable as a first-class field.

## Proposal
Add an explicit schema field to `RecursiveQuery` and use it as the authoritative output schema.

### API / Shape changes
1. Add `schema: DFSchemaRef` to `RecursiveQuery`.
2. Update `RecursiveQuery::try_new` to:
- derive schema from `static_term` and `recursive_term`
- validate terms against derived schema
- set `self.schema` explicitly
- only align children when required for semantic correctness, not just to encode ownership
3. Update `LogicalPlan::schema()` arm for `RecursiveQuery` to return `&recursive.schema`.

### Contract changes
1. The recursive query node owns its output schema directly.
2. Child plans are inputs to schema derivation and validation, not storage for parent schema identity.
3. Reconstruction logic must preserve or recompute `schema` deterministically and validate consistency.

### Integration surface to update
1. Logical plan creation paths (`LogicalPlanBuilder`, parser/planner sites).
2. Tree rewrite / `TreeNode` reconstruction paths using `RecursiveQuery::try_new`.
3. Proto serialization and deserialization for `RecursiveQuery` shape changes.
4. Any optimizer rules relying on `static_term.schema()` as recursive node output schema.
5. Display/debug and equality/hash behavior impacted by the additional field.

## Design Options
1. Eager schema field (recommended): always compute/store `schema` during `try_new`.
- Pros: simple contract, deterministic, no lazy cache behavior.
- Cons: constructor changes across call sites.

2. Lazy derived schema accessor without field.
- Pros: avoids shape change.
- Cons: keeps ownership implicit and does not solve reconstruction/proto coupling.

3. Hybrid (store + validate against children on construction).
- Pros: strongest invariant enforcement.
- Cons: slightly more code, but best long-term correctness profile.

## Migration Plan
1. Introduce `schema` field and adapt `RecursiveQuery::try_new` + tests.
2. Switch `LogicalPlan::schema()` to return node-owned schema.
3. Update tree/proto constructors to preserve roundtrip compatibility.
4. Adjust projection/alignment logic to avoid schema-only wrappers where unnecessary.
5. Add regression tests covering construction, rewrite, and serde invariants.

## Backward Compatibility and Risks
1. Logical plan node shape changes are semi-public within DataFusion internals; this may impact downstream code touching `RecursiveQuery` fields directly.
2. Proto changes may require compatibility handling if older serialized plans are expected.
3. Equality/hash/ordering semantics may change if schema participates directly; keep behavior intentional and documented.
4. Planner/optimizer code paths that assumed recursive output equals `static_term.schema()` must be audited.

## Acceptance Criteria
1. `RecursiveQuery` stores a dedicated derived schema field.
2. `LogicalPlan::schema()` for recursive queries returns node-owned schema, not child schema.
3. Construction/reconstruction paths enforce consistency between stored schema and terms.
4. Proto and tree rewrite roundtrips preserve recursive query schema invariants.
5. Redundant schema-only projection wrapping is reduced where safe.
6. Existing recursive CTE behavior remains unchanged for valid queries.

## Testing Strategy
### Unit tests (logical)
1. `RecursiveQuery::try_new` stores expected derived schema on node.
2. `LogicalPlan::schema()` equals stored schema even if child schema differs structurally.
3. Validation rejects mismatched child schemas (field count/type incompatibility).
4. Rebuild/rewrite preserves stored schema invariants.

### Serialization / reconstruction tests
1. Proto roundtrip preserves recursive schema.
2. TreeNode transform roundtrip preserves schema without requiring schema-only child rewrites.

### Physical / integration tests
1. Recursive planning and execution still succeed for representative recursive CTE plans.
2. No behavioral regression in recursive CTE sqllogictests.

### Suggested validation commands
1. `cargo test -p datafusion-expr recursive_query -- --nocapture`
2. `cargo test -p datafusion-physical-plan recursive_query_exec -- --nocapture`
3. `cargo test -p datafusion-physical-plan project_plan_to_schema -- --nocapture`
4. `cargo test -p datafusion-sqllogictest --test sqllogictests -- cte`

## Out of Scope
1. Broader recursive CTE feature expansion.
2. Non-recursive schema ownership refactors in unrelated plan nodes.
3. Major planner architecture rewrites beyond recursive query schema ownership.

## Rationale for Priority
This is medium-to-large effort, but high leverage. It improves core schema-contract clarity, reduces hidden coupling, and lowers future regression risk in recursive CTE planning, especially as nullability/metadata semantics continue to evolve.

## GitHub Issue Draft (Ready to Paste)

### Title
Store derived recursive CTE schema directly on `RecursiveQuery`

### Body
## Problem
`RecursiveQuery` currently owns its output schema implicitly by aligning/wrapping `static_term`, and `LogicalPlan::schema()` for recursive queries returns `static_term.schema()`.

This creates a hidden invariant: schema ownership is encoded in child rewrite shape rather than as explicit node state. As a result, schema-only projection churn, rewrite fragility, and serde/reconstruction coupling are higher than necessary.

## Why this matters
- Core recursive CTE schema invariants are harder to inspect and enforce.
- Tree/proto reconstruction depends on replaying child alignment logic.
- Planner and optimizer code can accidentally couple to `static_term` shape instead of node-owned schema.
- This diverges from the stronger structural direction to make invariants explicit on the owning node.

## Proposed change
Add a dedicated `schema: DFSchemaRef` field to `RecursiveQuery` and treat it as authoritative.

### Scope
1. Add `schema` field to `RecursiveQuery`.
2. Update `RecursiveQuery::try_new` to derive/validate schema and store it directly.
3. Update `LogicalPlan::schema()` recursive arm to return node-owned schema.
4. Update tree rewrite and proto serde paths to preserve/reconstruct this invariant.
5. Reduce schema-only child wrapping where it is only carrying parent schema identity.

## Acceptance criteria
- `RecursiveQuery` stores explicit derived schema.
- `LogicalPlan::schema()` for recursive query returns node-owned schema.
- Construction and reconstruction validate schema consistency with static/recursive terms.
- Proto and tree roundtrips preserve recursive schema contract.
- Existing valid recursive CTE behavior is unchanged.

## Test plan
### Logical tests
- `RecursiveQuery::try_new` stores expected derived schema.
- Recursive `LogicalPlan::schema()` returns stored node schema.
- Invalid column count/type mismatches are rejected.

### Roundtrip tests
- Proto roundtrip preserves recursive schema invariants.
- Tree transform/rebuild preserves recursive schema without requiring schema-only child rewrites.

### Execution/integration checks
- `cargo test -p datafusion-expr recursive_query -- --nocapture`
- `cargo test -p datafusion-physical-plan recursive_query_exec -- --nocapture`
- `cargo test -p datafusion-physical-plan project_plan_to_schema -- --nocapture`
- `cargo test -p datafusion-sqllogictest --test sqllogictests -- cte`

## Risks / compatibility notes
- Logical node shape changes may affect internal/downstream code that pattern matches `RecursiveQuery` fields.
- Proto compatibility may need explicit handling for older serialized plans.
- Equality/hash/ordering semantics should remain deliberate if schema is included directly.

### Checklist
- [ ] Add `schema: DFSchemaRef` to `RecursiveQuery`
- [ ] Switch recursive `LogicalPlan::schema()` to node-owned schema
- [ ] Update `RecursiveQuery::try_new` invariant checks
- [ ] Update tree reconstruction paths
- [ ] Update proto serde paths
- [ ] Add logical + roundtrip regression tests
- [ ] Run recursive CTE validation test suite