source: nullability-mismatch-22034a
# Issue: Make recursive CTE schema ownership explicit without a public struct field

## Summary
`RecursiveQuery::try_new` currently makes recursive CTE output schema ownership implicit by wrapping `static_term` in an idempotent schema-only `Projection`. The physical planner then depends on an identity-projection fast path to replace that projection with `project_plan_to_schema` / `SchemaAdapterExec` alignment.

This works, but the invariant is distributed across multiple layers and is easy to miss. The proposal is to introduce a private helper/constructor abstraction that makes this schema-ownership contract explicit while keeping `RecursiveQuery` public shape source-compatible.

## Motivation
The current implementation avoids a semver-breaking public field change on `RecursiveQuery`, which is good. However, the contract now relies on two separate pieces that must remain synchronized:

1. Logical planning encodes derived schema ownership using a schema-only projection.
2. Physical planning recognizes that projection pattern and applies a special-case fast path.

This coupling is subtle. Future changes in either layer can regress recursive CTE schema alignment without obvious compile-time failures.

## Problem Statement
Current behavior depends on an implicit cross-layer contract:
- `RecursiveQuery` has no explicit internal marker for "schema ownership is encoded in `static_term` shape".
- Callers can construct or rewrite plans without noticing this requirement.
- The physical planner special case is generic (`is_identity_projection`) and not strongly named around recursive CTE schema alignment semantics.

Risks:
- Harder reasoning about invariants during plan rewrites and proto/tree reconstruction.
- Higher chance of accidental drift between logical construction and physical planning behavior.
- Reduced discoverability for maintainers unfamiliar with this specific implementation detail.

## Proposed Refactor
Introduce a private, well-named abstraction in logical planning to centralize this contract.

### Option A (preferred)
Create a private helper in logical-plan code, for example:
- `RecursiveQuery::align_static_term_to_recursive_output_schema(...)`
- or `build_schema_owned_static_term(...)`

Responsibilities:
1. Derive/validate recursive CTE output schema.
2. Construct the schema-only static-term alignment projection when needed.
3. Guarantee idempotence (no-op if already aligned).
4. Document the physical-planner counterpart expectation.

### Option B
Create a private constructor wrapper around `RecursiveQuery::try_new` dedicated to this invariant, and keep `try_new` delegating to it.

Responsibilities:
1. Same as above.
2. Keep all call sites routed through one invariant gate.

## Non-Goals
- No semver-breaking addition of a new public field on `RecursiveQuery`.
- No behavior change to recursive CTE SQL semantics.
- No broad planner architecture rewrite.

## Acceptance Criteria
1. A private logical-plan abstraction exists for schema-only recursive static-term alignment.
2. `RecursiveQuery::try_new` (and reconstruction paths) use this abstraction.
3. The abstraction documents why identity projection exists and how physical planning handles it.
4. Existing valid recursive CTE behavior is preserved, including `0 AS level` coverage.
5. Regression tests verify invariants for:
   - idempotent schema-only alignment
   - recursive output schema consistency through logical -> physical planning
   - behavior when no alignment is needed

## Suggested Test Plan
1. Unit tests (logical):
   - helper is idempotent for already-aligned static terms
   - helper preserves expression identity and only changes advertised schema when intended
2. Planner tests:
   - recursive CTE plan with schema-only static projection triggers physical schema alignment path
   - recursive CTE plan without schema mismatch does not add unnecessary adapter/projection
3. SQL logic tests:
   - keep `0 AS level` recursive CTE regression guard
   - preserve NULL-termination recursive cases

## Risks and Mitigations
- Risk: helper naming/API still too generic.
  - Mitigation: use recursive CTE-specific naming and doc comments.
- Risk: hidden behavior changes in optimizer rewrite paths.
  - Mitigation: add reconstruction/roundtrip tests that validate schema contract before and after rewrites.
- Risk: accidental performance regressions from redundant wrappers.
  - Mitigation: preserve idempotence and existing fast-path behavior.

## Expected Impact
- Better maintainability of recursive CTE schema contract.
- Reduced coupling risk between logical construction and physical special casing.
- Clearer onboarding for future contributors touching recursive planning.

## Origin
Derived from PR review refactor opportunity:
"Make recursive CTE schema ownership explicit without a public struct field".
