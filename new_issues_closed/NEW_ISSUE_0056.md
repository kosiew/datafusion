Verdict: Not high-impact refactor.
 Better framed as: “targeted physical optimization; moderate complexity; niche
 payoff.”

source: pr-22239_a
# Issue Proposal: Physical Sort-Key Simplification for get_field(named_struct(...), f)

## Summary
Physical plans can retain sort keys of the form `get_field(s, field)` even when `s` is produced by `named_struct(...)` in the same plan. This creates avoidable per-row sort-key expression work and prevents direct-column sort keys from being used where they are derivable.

## Background
In SQLLogicTest coverage under `datafusion/sqllogictest/test_files/order.slt`, the following pattern appears:

- Query projects `s = named_struct('a', a, 'b', b)`
- `ORDER BY s['a']`
- Physical plan shows `SortExec: expr=[get_field(s@0, a) ...]`

This is semantically correct, but it misses an optimization opportunity: the sort key can often be rewritten to the underlying source expression (`a@0` in this case) rather than computed via `get_field`.

## Problem Statement
The physical optimizer does not currently normalize sort expressions through inline struct constructors, even though an equivalent simplification exists at the expression level for logical forms.

As a result:
- Sort key evaluation incurs extra expression work per row.
- Sorts cannot always leverage the simplest key representation.
- Plan readability and downstream optimizer opportunities are reduced.

## Why This Matters
The core sort complexity remains `O(N log N)`, but key extraction overhead can still affect runtime constants, CPU usage, and memory/cache behavior on large inputs. A direct-column key is generally cheaper and easier for the optimizer to reason about.

## Scope
In scope:
- Add a physical sort-key normalization step that can fold `get_field(named_struct(...), literal_field)` to the corresponding child expression when safe.
- Restrict the rewrite to provably safe cases (literal field name, deterministic constructor semantics, no ambiguity).
- Add targeted tests for plan shape and correctness.

Out of scope:
- Broad physical-expression canonicalization unrelated to sort keys.
- Non-literal or dynamic field-name rewrites.
- Behavioral changes to null ordering or Arrow field resolution semantics.

## Proposed Approach
1. Identify where physical sort expressions are normalized/rebuilt in the physical optimizer pipeline.
2. Introduce a focused rewrite pass for sort keys:
   - Match `get_field(base, field_name)` with optional nested path handling if supported.
   - If `base` is structurally equivalent to `named_struct(...)` and `field_name` is a known literal, resolve to the mapped child expression.
   - Keep conservative bail-outs for ambiguous or dynamic cases.
3. Ensure first-match semantics remain consistent with existing field lookup behavior.
4. Run the new rewrite only where it is safe and beneficial for sort execution.

## Acceptance Criteria
- For representative queries in `order.slt`, plans no longer require `get_field(s@0, a)` when direct key extraction is derivable.
- Query results are unchanged for all existing tests.
- New tests cover:
  - Positive simplification case.
  - Duplicate-name or ambiguous-name conservative behavior.
  - Dynamic/non-literal field name no-op behavior.
- No regressions in sort correctness (ordering, nulls, stability assumptions).

## Validation Plan
- Run targeted sqllogictests involving struct projections and `ORDER BY` field access.
- Add/adjust unit tests in the relevant physical optimizer module for expression rewrite behavior.
- Run crate-scoped tests for touched crates, then wider checks if cross-crate behavior is affected.

## Risks and Mitigations
Risk: Unsound rewrite in ambiguous field-name scenarios.
Mitigation: Keep strict guards and conservative early exits.

Risk: Rule interaction with existing optimizer passes.
Mitigation: Add tests that exercise pass ordering and final plan shape.

## Notes
This issue is intentionally focused on sort-key normalization and can be implemented incrementally without changing SQL semantics.