source: pr-22564_a
# Refactor Issue 01: Centralize log domain validation across runtime and simplification

## Title
Centralize logarithm domain validation so simplification and runtime enforce the same invariants

## Problem Statement
The log function currently enforces value-domain constraints during runtime evaluation, but expression simplification uses independent algebraic rewrites that do not consistently apply those constraints. This can produce optimizer-only results that differ from runtime behavior for the same logical input.

A concrete example is the simplification of log(a, a) to 1, which can incorrectly simplify log(0, 0) to 1.0 instead of raising the same domain error returned by runtime execution paths.

## Why This Matters
- Correctness: SQL users should get consistent semantics regardless of whether an expression was simplified.
- Safety: Invalid-domain inputs should fail predictably in all execution paths.
- Maintainability: Domain rules duplicated in multiple places are easy to drift apart.

## Current Symptoms
- Runtime path rejects zero value inputs with a compute error.
- Simplification path can fold some invalid expressions into constants.
- Coverage does not currently guarantee parity for binary log simplification when literals are involved.

## Root Cause
Domain validation is not modeled as a shared contract. Runtime and simplifier each apply partially overlapping logic:
- Runtime validates domain during scalar/array evaluation.
- Simplifier applies algebraic identities without always gating them on domain-valid literals.

## Goals
- Ensure simplification never produces a result for inputs that runtime would reject.
- Encode log domain checks in reusable helper logic used by simplification decisions.
- Add regression tests that lock in parity between optimized and non-optimized behavior for zero-value and related literal cases.

## Non-Goals
- Redesigning all numeric domain validation for every function in this change.
- Broad optimizer architecture changes beyond what is needed for log correctness.
- Performance tuning beyond avoiding obvious regressions.

## Proposed Refactor
1. Introduce shared domain predicate helpers for log literal analysis.
   - Add focused helper functions that can answer whether a literal value/base pair is definitely valid, definitely invalid, or unknown at plan time.
   - Keep helpers small and deterministic, operating on expression or scalar literal forms used by simplification.

2. Gate simplification rules behind domain-aware checks.
   - For identities such as log(a, a) => 1 and log(a, power(a, b)) => b, require domain validity when literals make that decidable.
   - If validity is unknown (non-literal expressions), preserve current behavior where safe, but do not fold known-invalid literal cases.

3. Preserve runtime validation as the final enforcement layer.
   - Runtime checks remain authoritative for dynamic values.
   - Simplifier should avoid introducing semantic divergence, not replace runtime validation.

4. Add targeted tests for optimized-path parity.
   - Unit tests in log simplification for literal edge cases.
   - SQL-level regression tests covering both unary and binary forms where optimization is expected.

## Candidate Test Matrix
At minimum, verify both planning/simplification correctness and runtime outcomes for:
- log(0)
- log(2, 0)
- log(0, 0)
- log(2, 2)
- log(10, 10)
- log(base, power(base, x)) for valid literal base/value

Expected direction:
- Known invalid literal domains should not simplify to constants that bypass errors.
- Valid identities should still simplify.

## Implementation Sketch
1. Add helper API in log function module for literal-domain checks used by simplification.
2. Update simplify match arms to consult helper before folding.
3. Add/extend unit tests around simplify behavior for invalid literals.
4. Add SQL logic tests to lock query-level behavior.
5. Run crate-scoped tests and targeted sqllogictest files.

## Risks and Mitigations
- Risk: Over-constraining simplification and missing valid folds.
  - Mitigation: Tri-state helper outcome (valid/invalid/unknown) and conservative folding.

- Risk: Behavior drift for decimal or mixed numeric literal types.
  - Mitigation: Explicit tests for representative float and decimal literal cases.

- Risk: Hidden dependency on existing simplifier assumptions.
  - Mitigation: Keep refactor scoped to log and preserve current behavior when domain validity is not decidable.

## Acceptance Criteria
- log(0, 0) no longer simplifies to 1.0.
- Optimized and non-optimized execution produce equivalent domain-error semantics for covered log cases.
- Existing valid simplifications for log remain intact.
- New regression tests added and passing.

## Suggested Labels
- bug
- refactor
- optimizer
- sql-semantics
- tests

## Suggested Owner Area
- datafusion/functions
- expression simplification and optimizer behavior for scalar UDFs
