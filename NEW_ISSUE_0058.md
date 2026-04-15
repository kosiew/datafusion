source: unnested-pruned-02-20118a
# Issue: Encapsulate the UNNEST pruning safety proof

## Summary
The optimizer rule that removes unused `UNNEST` under duplicate-insensitive aggregates is correct, but its safety proof is spread across multiple helpers and naming that does not explicitly encode all required invariants.

This issue proposes a small/medium refactor to make the proof contract explicit and harder to violate in future extensions.

## Context
Current rewrite entry point:
- `remove_unused_unnest_from_duplicate_insensitive_input` in `datafusion/optimizer/src/optimize_projections/mod.rs`

Current proof helpers:
- `can_remove_unused_unnest_for_exprs`
- `is_unnested_input_index`
- `unnest_preserves_at_least_one_row_per_input`

The rewrite is only attempted when aggregate expressions are pruned away (`new_aggr_expr.is_empty()`), and then uses `can_remove_unused_unnest_for_exprs` for safety.

## Problem
The rule depends on two independent safety properties:

1. Expression-independence:
   Required expressions above the `UNNEST` must not depend on unnested outputs.

2. Row-preservation:
   The `UNNEST` must preserve at least one output row per input row (under current narrow literal list conditions).

Today, these properties are enforced, but the parent helper name emphasizes duplicate-insensitive context and does not clearly advertise the full safety contract. This increases risk that future changes broaden the rule without preserving both checks.

## Why this matters
This is a contract-clarity and maintainability issue, not a known correctness bug.

If future edits weaken either side of the proof, behavior can silently regress for cardinality-sensitive plans or grouped outputs.

## Proposed change
Refactor for explicit contract without behavior change:

1. Introduce or rename to a clearly named predicate that encodes both requirements, for example:
   - `can_remove_unused_row_preserving_unnest_for_exprs`
   - or similarly explicit wording

2. Add a short doc comment above the predicate listing both invariants.

3. Keep existing helper decomposition (`is_unnested_input_index`, `unnest_preserves_at_least_one_row_per_input`) but group and order helpers so the proof reads top-down.

4. Update call sites to use the explicit predicate name.

## Acceptance criteria
- No behavior change in optimizer output.
- Safety predicate naming or documentation explicitly states both:
  - required expressions do not depend on unnested outputs
  - unnest preserves at least one row per input row
- Call sites reference the explicit predicate directly.
- Existing tests continue to pass.

## Suggested validation
- Run optimizer unit tests around `optimize_projections`.
- Run relevant UNNEST SQLLogicTests to ensure no regressions in cardinality-sensitive cases.

## Non-goals
- Expanding pruning eligibility beyond current narrow literal list conditions.
- Changing SQL semantics for `UNNEST`.
- Broad optimizer architecture changes outside this local proof surface.

## Labels (suggested)
- `optimizer`
- `refactor`
- `good first issue` (optional; only if maintainers agree scope is suitable)

## Notes for implementers
The current logic is already mostly encapsulated in `can_remove_unused_unnest_for_exprs`. This issue is primarily about making the safety contract explicit in naming/docs so extensions remain safe by construction.
