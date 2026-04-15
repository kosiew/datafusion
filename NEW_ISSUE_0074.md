source: pr-22628_a
# Centralize SQL float equality normalization

## Summary
DataFusion currently applies SQL `-0.0 == +0.0` semantics in several local equality/comparison paths rather than through one shared abstraction. This makes it easy for one execution path to be fixed while another path keeps Arrow totalOrder or raw-bit float behavior.

## Problem
SQL-visible equality should treat `-0.0` and `+0.0` as equal. After PR #22628, comparison kernels and some `IN` paths normalize negative zero before equality checks. However, equality semantics are also implemented in other places, including hash-based paths and join equality helpers.

Examples of places that may need consistent semantics:

- scalar/array comparison expressions
- `IN` list evaluation, including static-filter fast paths and fallback comparator paths
- hash joins / equality joins
- grouping and `DISTINCT`
- hash utilities used before equality verification
- nested, dictionary, and run-end-encoded float-containing arrays
- `IS [NOT] DISTINCT FROM` paths

When each area handles float equality independently, regressions can appear as inconsistent SQL behavior: `WHERE x = 0.0` may match `-0.0`, while an equijoin or grouping operation may still distinguish it.

## Proposed improvement
Introduce one shared SQL float equality normalization contract and route all SQL equality/hash paths through it.

Possible design:

1. Add shared helpers for SQL equality normalization:
   - normalize `-0.0` to `+0.0` for `Float16`, `Float32`, and `Float64`
   - support arrays, scalars, dictionary arrays, and run-end-encoded arrays
   - avoid allocation when no negative zero is present

2. Add shared hash behavior for SQL equality keys:
   - hash `-0.0` and `+0.0` identically
   - preserve existing NaN semantics intentionally, with tests documenting the chosen behavior
   - handle nested/dictionary/REE values consistently with equality comparison

3. Replace local ad hoc normalization in callers with the shared helpers.

4. Add regression coverage across representative SQL-visible paths.

## Suggested test coverage
Add SQLLogicTests and/or unit tests for:

- `-0.0 = +0.0`
- `-0.0 < +0.0` is false
- `-0.0 <= +0.0` is true
- `-0.0 IS NOT DISTINCT FROM +0.0`
- `-0.0 IN (+0.0)` and `+0.0 IN (-0.0)`
- equijoin on float keys containing `-0.0` and `+0.0`
- `GROUP BY` / `DISTINCT` behavior for float zeros
- dictionary-encoded float values if supported
- run-end-encoded float values if supported
- nullable cases to preserve SQL three-valued logic
- NaN cases to prevent accidental semantic changes

## Acceptance criteria
- All SQL-visible equality paths treat `-0.0` and `+0.0` consistently.
- Hash-based equality paths hash `-0.0` and `+0.0` identically when SQL equality semantics apply.
- Existing NaN behavior is documented and covered by tests.
- Local duplicated normalization logic is removed or reduced to wrappers around the shared implementation.
- Regression tests cover comparison, `IN`, join, and grouping/DISTINCT behavior.

## Notes
This is out of scope for PR #22628 unless that PR claims to fix all SQL equality semantics globally. It is still worth tracking separately because partial fixes can leave hard-to-debug semantic inconsistencies across execution operators.
