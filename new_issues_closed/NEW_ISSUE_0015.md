closed - issue is stale
source: pr-21402_a
# [Refactor] Extract shared primitive/float StaticFilter contains flow

## Summary
`primitive_static_filter!` and `float_static_filter!` in `in_list.rs` currently duplicate most of their `contains()` implementation flow. The duplicated sections include:
- Dictionary-needle recursion.
- SQL three-valued null-mask handling.
- Boolean result array assembly.

The primary difference is membership lookup representation:
- Primitive branch uses native values.
- Float branch uses ordered-float wrappers for stable hashing/equality semantics.

## Is your feature request related to a problem or challenge?
The duplication increases maintenance cost and semantic drift risk:
- Bug fixes in one macro can be missed in the other.
- Subtle null-semantics differences can creep in over time.
- Code review of behavior changes is harder because logic is repeated.

Relevant location:
- `datafusion/physical-expr/src/expressions/in_list.rs` (`primitive_static_filter!`, `float_static_filter!`)

## Describe the solution you'd like
Refactor shared `contains()` workflow into a reusable helper while preserving specialized set-storage and lookup differences for primitive and float implementations.

### Proposed direction
1. Extract common steps into a helper that accepts a membership predicate/adapter:
- Handle dictionary-needle recursion.
- Build null mask using SQL three-valued logic.
- Construct output `BooleanArray` from nullable boolean results.
2. Keep primitive and float membership implementations specialized by passing type-specific lookup closures/adapters.
3. Preserve macro-generated type coverage and existing public behavior.

### Scope boundaries
In scope:
- Internal refactor only.
- No behavior change intended.
- Keep current specialization for primitive and float storage/lookup.

Out of scope:
- Rewriting macro strategy for all filter types.
- Changing SQL `IN` null semantics.
- Broad refactor of non-numeric `StaticFilter` paths.

## Acceptance criteria
1. Shared workflow code is centralized (null-mask + boolean assembly + dictionary-needle recursion).
2. Primitive and float paths retain current type-specific membership semantics.
3. Existing tests for dictionary and null behavior continue to pass without behavior regressions.
4. New regression tests verify parity between primitive and float branches for equivalent scenarios.

## Testing strategy
- Re-run existing `in_list.rs` test suite, especially dictionary-needle and null-semantic tests.
- Add targeted parity tests that exercise both primitive and float filters under the same logical scenarios:
  - value present / absent
  - haystack includes null
  - needle includes null
  - dictionary-encoded needle path
- Add one test focused on boolean output/null-mask parity across primitive vs float implementations.

## Alternatives considered
1. Keep duplicated macros and add comments.
- Pros: lowest refactor effort.
- Cons: drift risk remains.

2. Fully unify primitive and float storage model.
- Pros: less specialized code.
- Cons: may reduce clarity/performance and increase migration risk.

## Risks and mitigations
- Risk: helper abstraction obscures performance-critical paths.
  - Mitigation: keep hot-path membership lookup inlined/specialized and only share orchestration logic.
- Risk: accidental semantic shift in null handling.
  - Mitigation: add explicit null-semantics regression tests and compare outputs against existing behavior.

## Additional context
This issue comes from PR review follow-up for apache/datafusion#21402. The opportunity predates that PR but remains medium-effort/high-value because it reduces future divergence between numeric `IN` implementations.
