closed - not high impact anymore
source: pr-*21402_a*
# [Refactor] Centralize InList haystack normalization and logical type validation

## Summary
`InListExpr` currently performs related normalization/validation work in multiple places with overlapping logic:
- `try_new_from_array(...)` validates logical type equality between the needle expression and the haystack array.
- `try_new(...)` validates logical type equality between the needle expression and each list expression.
- `instantiate_static_filter(...)` performs dictionary flattening / haystack normalization used by static filter construction.

Because these responsibilities are split, future updates to dictionary behavior or logical-type compatibility can drift across constructors.

## Is your feature request related to a problem or challenge?
The current design duplicates compatibility checks and keeps normalization behavior separate from constructor-level validation.

Observed pain points:
- Error behavior can diverge between constructor entry points.
- Type-support changes require touching several locations and keeping semantics in sync manually.
- Review burden is higher because correctness depends on parallel edits.

Relevant code paths today:
- `datafusion/physical-expr/src/expressions/in_list.rs` (`InListExpr::try_new_from_array`)
- `datafusion/physical-expr/src/expressions/in_list.rs` (`InListExpr::try_new`)
- `datafusion/physical-expr/src/expressions/in_list.rs` (`instantiate_static_filter`)

## Describe the solution you'd like
Introduce a single helper-based normalization + validation path that both constructors use.

### Proposed direction
1. Add a private helper that validates logical compatibility with one authoritative error shape.
2. Add a private helper that normalizes haystack array representation before static filter instantiation (including dictionary flattening behavior).
3. Route both `try_new(...)` and `try_new_from_array(...)` through these helpers so constructor behavior is aligned.
4. Keep external APIs unchanged.

### Scope boundaries
In scope:
- Internal refactor for consistency and maintainability.
- No semantic changes intended for successful queries.
- Preserve existing support for dictionary and non-dictionary haystacks.

Out of scope:
- New SQL features.
- New public API surface.
- Broad redesign of `StaticFilter` internals.

## Acceptance criteria
1. `try_new(...)` and `try_new_from_array(...)` share one validation path for logical type checks.
2. Validation failure text is consistent regardless of constructor path.
3. Dictionary flattening/normalization rules used for static filter construction are centralized and documented in code comments.
4. Existing behavior remains unchanged for already-supported valid/invalid combinations.
5. Existing tests continue to pass; any changed expectations are justified and minimal.

## Testing strategy
- Keep and run existing `in_list.rs` tests that cover dictionary and mismatch behavior.
- Add focused constructor-parity tests proving equivalent outcomes from `try_new(...)` vs `try_new_from_array(...)` for representative cases:
  - Primitive
  - String
  - Dictionary value equivalence
  - Logical type mismatch
- Add one assertion-focused test to confirm error message parity for mismatch failures across both constructors.

## Alternatives considered
1. Keep the current split and rely on comments/tests only.
- Pros: no refactor risk.
- Cons: higher long-term drift risk and maintenance overhead.

2. Move all logic into only one constructor and treat the other as thin conversion glue.
- Pros: very explicit single entry point.
- Cons: may introduce avoidable conversion overhead or awkward control flow.

## Risks and mitigations
- Risk: behavior change in edge-case dictionary/literal compatibility.
  - Mitigation: parity tests for current behavior and targeted mismatch cases.
- Risk: accidental performance regression if normalization adds extra work.
  - Mitigation: preserve current fast paths and avoid unnecessary allocations/clones.

## Additional context
This issue comes from PR review follow-up for apache/datafusion#21402 and is intended as a medium-effort refactor to reduce semantic drift risk in `IN` expression construction and static filter setup.
