Verdict: No, not high-impact refactor. Likely obsolete/stale.
source: pr-22451_a
# Issue: Centralize HashTableLookupExpr -> lit(true) proto encoding

## Type
Refactor / consistency hardening

## Summary
The current PR introduces a second implementation of HashTableLookupExpr protobuf encoding to a literal true node while retaining the existing fallback implementation in the proto serializer. The duplicated logic increases maintenance risk and can diverge over time.

## Why this is a problem
Two separate code paths encode the same conceptual behavior:
- Expression-local path in `datafusion/physical-plan/src/joins/hash_join/partitioned_hash_eval.rs`
- Centralized fallback path in `datafusion/proto/src/physical_plan/to_proto.rs`

These paths already differ in behavior around `expr_id` handling:
- Expression-local construction currently sets `expr_id: None`
- Fallback construction uses serializer-provided `expr_id`

If both paths remain active during migration, future edits may unintentionally change one path but not the other, causing inconsistent serialization behavior depending on dispatch route.

## Evidence
The expression-local implementation constructs:
- Literal bool true value
- PhysicalExprNode with `expr_id: None`

The fallback implementation constructs:
- Literal bool true value
- PhysicalExprNode with `expr_id` preserved from the current expression

The duplicate comments describing correctness rationale are also now split between crates.

## Scope
This issue is about reducing duplicate serialization logic while both fallback and expression-local encoding paths may coexist.

## Proposed resolution
Choose one of the following:

1. Preferred transitional approach
- Extract a shared helper that builds the literal-true PhysicalExprNode.
- Use that helper in both locations while migration is incomplete.
- Parameterize helper input so `expr_id` behavior is explicit and consistent.

2. Preferred end state
- Once `PhysicalExpr::try_to_proto` override is correctly implemented and wired for HashTableLookupExpr, remove the fallback special-case from `to_proto.rs`.
- Keep a single source of truth in expression-local serialization.

## Acceptance criteria
- Exactly one canonical construction path exists for HashTableLookupExpr -> literal true encoding, or both call a single shared helper.
- `expr_id` behavior is explicitly specified and consistent with serializer expectations.
- Comments explaining why literal true is correctness-safe exist in one maintained location (or are minimal references to that location).
- No behavior regression in existing physical expression serialization.

## Test plan
- Add/update a targeted serialization test that covers HashTableLookupExpr encoding and asserts:
  - Encoded node is a literal bool true expression.
  - `expr_id` semantics match intended contract.
- Run:
  - `cargo test -p datafusion-proto`
  - Any targeted tests touching HashTableLookupExpr serialization paths.

## Risk if not addressed
- Serialization drift between fallback and expression-local paths.
- Future regressions that are hard to diagnose because behavior depends on dispatch path.
- Inconsistent `expr_id` propagation that can break expression identity expectations.
