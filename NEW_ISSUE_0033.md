source: nullability-mismatch-22034a
# Issue: Preserve safe physical properties through SchemaAdapterExec

## Summary
`SchemaAdapterExec::try_new` currently remaps schema but rebuilds physical properties conservatively, including `UnknownPartitioning(partition_count)`. The operator preserves row count and positional value mapping, so some physical properties can be preserved safely in remapped form.

This issue proposes adding explicit property-preservation/remapping rules for `SchemaAdapterExec` to reduce avoidable optimizer regressions while keeping schema correctness guarantees intact.

## Motivation
`SchemaAdapterExec` is currently used for recursive CTE child schema alignment, but it is a generally useful adapter. If reused more broadly, conservative property reset can reduce optimization opportunities:

1. Lost ordering information may disable sort-avoidance optimizations.
2. Lost partitioning guarantees may weaken distribution-aware planning.
3. Lost equivalence relationships may reduce predicate/ordering reasoning power.

Because the adapter does not reorder rows and only changes schema presentation/casting for aligned fields, a subset of properties is safe to preserve after remapping.

## Problem Statement
Current contract is too conservative relative to actual operator semantics:
- Input rows are preserved one-to-one.
- Column positions remain aligned to target schema mapping.
- Output values are semantically equivalent to aligned input fields (subject to validated casts).

Yet output properties are rebuilt without preserving safe inherited guarantees. This can produce downstream plan quality regressions even when schema alignment is semantically lossless.

## Proposed Refactor
Define a dedicated property-remapping layer in `SchemaAdapterExec`.

### 1. Ordering preservation
Preserve output ordering when each ordering expression can be remapped from input schema/index to output schema/index without semantic change.

Rules:
- Preserve ordering for direct field pass-through.
- Preserve ordering through no-op/safe casts where ordering semantics are unchanged.
- Drop only the specific ordering components that cannot be safely remapped.

### 2. Partitioning preservation
Preserve partitioning when mapping is positional and does not alter partition key semantics.

Rules:
- Preserve hash/range partitioning when all partition expressions map safely.
- Fall back to `UnknownPartitioning` only when remapping cannot be proven safe.

### 3. Equivalence preservation
Remap equivalence classes from input to output columns for fields that are retained/aligned.

Rules:
- Carry over class members with direct column mapping.
- Drop members requiring unsafe or non-invertible transformations.
- Keep existing safety checks for cast/expr compatibility.

### 4. Documentation
Add explicit docs to `SchemaAdapterExec` clarifying:
- which properties are preserved,
- when they are downgraded,
- and why each downgrade is required for soundness.

## Non-Goals
- No unsound property propagation.
- No claim that all properties can always be preserved.
- No widening of cast support beyond existing schema-alignment safety constraints.

## Acceptance Criteria
1. `SchemaAdapterExec` preserves ordering/partitioning/equivalence when safe remapping is provable.
2. Property downgrades occur only for non-remappable/unsafe cases.
3. Existing correctness and schema-validation behavior remain unchanged.
4. Added tests cover both preserved and downgraded scenarios.
5. Optimizer/planner behavior shows no regressions in correctness.

## Suggested Test Plan
1. Unit tests for ordering:
   - pass-through aligned projection preserves sort expressions
   - partial-remap case drops only unremappable ordering components
2. Unit tests for partitioning:
   - hash/range partitioning preserved when partition keys remap directly
   - fallback to `UnknownPartitioning` when remap is unsafe
3. Unit tests for equivalence properties:
   - direct column-equivalence classes are preserved through remap
   - unsafe members are dropped while safe members remain
4. Integration/planner tests:
   - plans that depend on preserved ordering avoid unnecessary extra sort
   - recursive CTE schema-alignment path remains correct

## Risks and Mitigations
- Risk: unsound property retention introduces wrong-plan assumptions.
  - Mitigation: implement strictly proof-based remapping with explicit fallback to conservative behavior.
- Risk: added complexity in property mapping logic.
  - Mitigation: isolate remapping helpers and keep each property family independently testable.
- Risk: behavior differences across edge casts/types.
  - Mitigation: gate preservation on existing cast safety checks and add targeted edge-case tests.

## Expected Impact
- Better optimizer outcomes where schema adaptation is semantically transparent.
- Reduced unnecessary property loss in future `SchemaAdapterExec` reuse scenarios.
- Stronger long-term contract for schema adaptation operators.

## Origin
Derived from PR review refactor opportunity:
"Preserve safe physical properties through `SchemaAdapterExec`".
