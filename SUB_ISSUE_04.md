# Sub-Issue 04: Add Proto Roundtrip Regression for Recursive CTE Nullability Widening

## Title
Proto deserialize/roundtrip should preserve widened recursive CTE output nullability

## Problem Statement
The protobuf deserializer now reconstructs recursive queries through `LogicalPlanBuilder::to_recursive_query`, which is the correct layer to re-apply recursive output-schema nullability widening. However, this change crosses a serialization boundary (`LogicalPlan` -> proto -> `LogicalPlan`) and currently lacks targeted regression coverage.

Without an explicit proto roundtrip test, we can pass SQL behavior tests while still regressing plan schema fidelity after serialization/deserialization.

## Why This Matters
- Correctness at boundaries: schema invariants must survive proto roundtrip, not only in direct SQL planning.
- Regression prevention: recursive CTE nullability bugs can reappear in serialized plans used by distributed or persisted workflows.
- Contract clarity: confirms `to_recursive_query` remains the canonical revalidation layer during deserialization.

## Scope
In scope:
- Add a proto roundtrip regression test for a recursive CTE where:
  - anchor column is non-null
  - recursive term can produce NULL
- Assert deserialized plan/schema exposes the recursive output column as nullable.
- Keep test focused on serialization boundary behavior.

Out of scope:
- Broad recursive schema refactors.
- Explain output formatting changes.
- Physical-plan schema helper consolidation.

## Reproducer Shape (Conceptual)
Use a recursive query pattern equivalent to:
- Anchor emits non-null value (for example, literal integer).
- Recursive branch can emit NULL for the same column (for example, conditional/null-cast expression).

The exact SQL or logical-plan construction used in test should remain deterministic and minimal.

## Required Assertions
1. Build/obtain recursive logical plan with widened output nullability.
2. Serialize plan to proto.
3. Deserialize proto back to logical plan.
4. Inspect deserialized recursive output schema.
5. Assert target column nullability is `true` after deserialization.
6. Optionally assert key non-nullability dimensions (name/type/order/metadata) remain stable.

## Suggested Test Placement
- Primary: proto logical plan roundtrip tests in:
  - `datafusion/proto/src/logical_plan/mod.rs`
  - or existing proto roundtrip test module adjacent to recursive-query coverage.

Prefer colocating with existing roundtrip tests to keep maintenance straightforward.

## Acceptance Criteria
1. New regression test fails if deserializer skips or breaks recursive nullability widening.
2. Test passes on current intended behavior using `LogicalPlanBuilder::to_recursive_query`.
3. Test explicitly verifies nullability on deserialized schema, not only query output rows.
4. No unrelated behavior changes are required to satisfy the test.

## Validation Plan
- Run targeted proto tests for logical plan roundtrip.
- Run targeted recursive CTE SQLLogicTests as a sanity check that SQL behavior remains unchanged.
- Ensure deterministic assertions (no brittle full-plan string matching unless necessary).

## Risks and Mitigations
- Risk: test accidentally validates SQL planning path instead of deserialize path.
- Mitigation: explicitly assert pre- and post-roundtrip schema nullability on the logical plan.

- Risk: brittle assertions tied to complete plan formatting.
- Mitigation: assert schema properties directly (nullability/type/name/order), avoid broad textual snapshots.

## Dependencies
- Depends on existing recursive CTE logical plan/proto helpers used by current tests.
- Independent of larger recursive schema consolidation work.

## Definition of Done
- A proto roundtrip regression test exists for recursive CTE nullability widening.
- The test proves deserialized recursive output schema remains nullable where required.
- The test guards the serialization boundary against future regressions.
