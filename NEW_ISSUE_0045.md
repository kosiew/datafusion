source: nullability-mismatch-22034a
# New Issue 02: Make Nullability-Only Projection Widening Explicit

## Title
Make nullability-only schema widening explicit in plan display (or API) to avoid misleading cast output in EXPLAIN

## Summary
Current schema alignment can represent nullability widening via same-type casts (for example, `CAST(1 AS Int64)`). This is semantically correct but confusing in `EXPLAIN`, because it looks like a value/type cast instead of a nullability adaptation.

This behavior is pre-existing, but recent recursive CTE changes surfaced it in SQLLogicTest diffs and plan review workflows.

## Problem Statement
Plan alignment currently uses generic cast-like expression output for a case where only nullability changes. As a result:
1. `EXPLAIN` output suggests type conversion where no type conversion is intended.
2. Plan diffs become noisy and harder to review.
3. Reviewers may misinterpret harmless schema adaptation as semantic expression change.

## Why This Matters
- Developer ergonomics: clearer explain plans reduce review friction.
- Debugging speed: users can distinguish value-level casts from schema adaptation.
- Test stability/readability: SQLLogicTest explain output is easier to reason about.

## Scope
In scope:
- Evaluate and implement one of the following:
  - Display-only improvement (preferred low-risk path).
  - Dedicated nullability-widening adapter expression/API path.
- Preserve existing runtime semantics.
- Add tests to pin intended explain/debug output.

Out of scope:
- Broad expression formatting redesign.
- Changing type coercion semantics.

## Candidate Approaches
### Option A: Display-only Annotation (Lower Risk)
- Keep existing execution expression behavior.
- Detect no-op same-type cast used solely for nullability adaptation.
- Render as explicit nullability adaptation in explain output (example: `NULLABILITY_WIDEN(expr)` or similar naming agreed by maintainers).

Pros:
- Minimal behavior risk.
- Focused change for readability.

Cons:
- Internal representation still uses generic cast semantics.

### Option B: Dedicated Nullability Adaptation Node (Higher Impact)
- Introduce an explicit expression or adapter for nullability-only widening.
- Use it in schema-alignment paths instead of cast.
- Update display/formatting and tests accordingly.

Pros:
- Strong semantic clarity in internals and display.

Cons:
- Larger API/runtime surface change.
- Potential ripple effects in optimizers and explain formatting.

## Suggested Initial Direction
Start with Option A (display-only) unless maintainers want stronger semantic modeling immediately. This addresses confusion quickly while preserving runtime behavior.

## Affected Areas
Likely touchpoints:
- Schema alignment path where nullability widening is introduced.
- Explain/display formatting logic for logical/physical expressions.
- SQLLogicTest expected explain output for impacted queries.

## Acceptance Criteria
1. Explain output no longer presents nullability-only widening as a misleading type/value cast.
2. Runtime behavior and query results remain unchanged.
3. Tests cover at least one case where previous output showed same-type cast for nullability adaptation.
4. Plan diffs are meaningfully clearer in affected recursive/schema-alignment scenarios.

## Validation Plan
- Add or update focused explain tests for nullability-only widening cases.
- Run impacted SQLLogicTests.
- Run targeted crate tests for expression display and planner alignment logic.

## Risks and Mitigations
- Risk: display change could break many plan text assertions.
- Mitigation: scope formatting change narrowly and update only affected expected outputs.

- Risk: users/tools rely on existing cast text in downstream parsing.
- Mitigation: if needed, gate new rendering behind explicit explain mode or keep textual compatibility where required.

## Dependencies
- Independent issue, but naturally related to recursive CTE/schema reconciliation work where this output appears frequently.

## Definition of Done
- Nullability-only schema adaptation is explicit and understandable in explain output (or API).
- Existing semantics are preserved.
- Tests lock in the improved behavior and prevent regressions.
