source: pr-19420_a
# Refactor: Introduce BinaryExprOptions for execution-policy metadata

## Summary
`BinaryExpr` currently stores execution-policy metadata (`fail_on_overflow`, `preselection_threshold`) as separate ad-hoc fields. This has already caused maintenance bugs where expression rewrites (for example `with_new_children`) preserved one flag but accidentally dropped another.

This issue proposes introducing a dedicated options container (for example `BinaryExprOptions`) and migrating `BinaryExpr` to use it consistently.

## Problem Statement
`BinaryExpr` mixes semantic identity (left, operator, right) with execution policy fields that influence runtime behavior but are not part of SQL semantics. As the number of policy fields grows, the implementation requires repetitive, error-prone forwarding across:

- expression rewrites (`with_new_children`)
- serialization/deserialization (proto)
- builders and constructors
- equality/hash implementation details

A recent regression demonstrates this risk: `preselection_threshold` is not forwarded in `with_new_children`, so optimizer rewrites can silently reset it.

## Why This Matters
- Correctness risk: policy values can be silently dropped.
- Maintenance risk: each new policy requires touching many call sites.
- Review burden: hard to verify all forwarding/serde paths by inspection.
- API churn risk: adding fields directly to `BinaryExpr` repeatedly expands surface area.

## Goals
1. Centralize binary-expression execution policy in one place.
2. Make rewrite and serde forwarding mechanically safer.
3. Keep behavior and public semantics unchanged.
4. Make future policy additions low-friction and less error-prone.

## Non-Goals
- No change to SQL semantics.
- No immediate addition of new user-facing configuration knobs.
- No broad refactor of non-binary physical expressions in this issue.

## Proposed Design

### 1. Add options struct
Introduce a dedicated options container in `datafusion/physical-expr/src/expressions/binary.rs`:

- `fail_on_overflow: bool`
- `preselection_threshold: f32`

Potential shape:

- `#[derive(Debug, Clone, PartialEq, Eq, Hash)]` if representation allows
- or manual impls if float handling requires bitwise comparison (`to_bits()`).

### 2. Update BinaryExpr storage
Replace separate fields on `BinaryExpr` with a single field:

- `options: BinaryExprOptions`

Keep existing methods for compatibility:

- `with_fail_on_overflow(...)`
- `with_preselection_threshold(...)`
- accessors like `preselection_threshold()`

These should delegate to `options` internally.

### 3. Make rewrites preserve options by construction
Update `with_new_children` to reuse/copy `self.options` rather than manually forwarding each policy field.

This should eliminate the class of bugs where one field is forgotten during reconstruction.

### 4. Localize proto mapping
Map the options struct as a single conceptual unit in proto conversion code:

- `datafusion/proto/src/physical_plan/to_proto.rs`
- `datafusion/proto/src/physical_plan/from_proto.rs`
- `datafusion/proto/proto/datafusion.proto`

Even if fields remain flattened in proto, conversion logic should operate through `BinaryExprOptions` to reduce drift.

### 5. Equality/Hash consistency for floating fields
Use one consistent policy for `preselection_threshold` equality/hash (prefer bitwise equality via `to_bits()`), so `Eq` and `Hash` contracts are upheld.

## Implementation Plan
1. Introduce `BinaryExprOptions` with defaults matching current behavior.
2. Migrate `BinaryExpr` internals to `options`.
3. Update builders/accessors to preserve external API behavior.
4. Update `with_new_children` to preserve options.
5. Update proto schema + serialization/deserialization paths.
6. Add/update tests (unit + roundtrip).
7. Run crate-scoped checks and relevant proto tests.

## Testing Plan

### Unit tests (`binary.rs`)
- Verify default options values remain unchanged.
- Verify each builder mutates only expected option field.
- Verify `with_new_children` preserves all options.
- Verify equality/hash behavior for finite values and NaN edge case handling.

### Proto tests
- Add roundtrip coverage for non-default options in:
  - `datafusion/proto/tests/cases/roundtrip_physical_plan.rs`
- Ensure non-default `preselection_threshold` survives encode/decode.

### Regression tests
- Explicit test for prior bug class: rebuild expression tree and assert options unchanged.

## Backward Compatibility
- Keep public builder/accessor methods unchanged to avoid downstream breakage.
- Maintain existing default behavior (`preselection_threshold = 0.2`, overflow default unchanged).
- Proto changes must be additive and backward-compatible with existing decoders where possible.

## Risks and Mitigations
- Risk: accidental API break during field migration.
  - Mitigation: preserve existing method signatures and behavior.
- Risk: proto compatibility regression.
  - Mitigation: additive proto fields + roundtrip and compatibility tests.
- Risk: float equality surprises.
  - Mitigation: explicit equality/hash policy and dedicated tests.

## Acceptance Criteria
- `BinaryExpr` stores execution policy via dedicated options struct.
- `with_new_children` preserves all policy options.
- Proto roundtrip preserves non-default binary-expression options.
- Equality/hash contracts are consistent for threshold representation.
- Existing behavior and defaults remain unchanged for users.

## Suggested Scope Labeling
- `refactor`
- `physical-expr`
- `proto`
- `good first follow-up` (optional; only if maintainers agree complexity is suitable)
