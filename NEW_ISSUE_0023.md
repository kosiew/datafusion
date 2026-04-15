source: pr-21787_a
# [Refactor] Centralize Binary Expression Formatting Rules Across `Display` and `human_display`

## Summary

DataFusion currently formats binary expressions through duplicated precedence-aware logic in two separate paths:
- `BinaryExpr::fmt` (`Display` for detailed expression output)
- `SqlDisplay::fmt` (`Expr::human_display()` path used by explain tree / human-readable rendering)

The duplicated implementation has already drifted behaviorally, and recent changes exposed missing associativity/equal-precedence handling in one path. This issue proposes a focused refactor to centralize parenthesizing rules for binary expressions in one shared formatter utility.

## Background / Context

A recent PR copied precedence-aware child-parenthesizing rules from `BinaryExpr::fmt` into `SqlDisplay::fmt` to improve readability correctness, but this created two independently maintained implementations.

Current state:
- Both paths apply precedence-based parenthesis insertion.
- Neither path robustly distinguishes left vs right child semantics for equal-precedence operators that are not safely associative.
- `human_display()` is expected to remain semantically faithful to the expression tree when removing parentheses.

Examples of tree/print ambiguity if equal-precedence right child is not parenthesized appropriately:
- `1 - (2 - 3)` vs printed `1 - 2 - 3`
- `1 - (2 + 3)` vs printed `1 - 2 + 3`
- `1 / (2 * 3)` vs printed `1 / 2 * 3`

## Problem Statement

Binary-expression formatting rules are duplicated and incomplete, causing two classes of risk:

1. **Semantic drift risk**
   Future fixes to one display path can be missed in the other.

2. **Incorrect omission of parentheses**
   Equal-precedence right-child expressions may print in a form that conventionally parses to a different expression tree.

## Goals

1. Create a single shared binary-expression formatting utility used by both `BinaryExpr::fmt` and `SqlDisplay::fmt`.
2. Preserve current output where semantically safe.
3. Add associativity-aware equal-precedence handling for right children.
4. Keep the utility small, explicit, and easy to extend for future display modes.
5. Add regression tests that lock behavior for ambiguous operator combinations.

## Non-Goals

- Reworking all expression display formatting beyond binary-expression parenthesis rules.
- SQL canonicalization/unparsing changes.
- Altering operator precedence tables.

## Proposed Design

### 1) Introduce a shared helper API

Add a shared formatting helper in `datafusion/expr/src/expr.rs` (or nearby internal module) that:
- Accepts parent operator, child expression, and child side (`Left`/`Right`).
- Decides whether to parenthesize based on:
  - Precedence comparison (`child < parent` must parenthesize)
  - Equal-precedence side-aware rule (right child may require parenthesizing)
  - Unknown/zero precedence fallback safety
- Delegates child rendering via callback/strategy to support both:
  - detailed `Display` formatting
  - SQL-like `SqlDisplay` formatting

### 2) Add explicit equal-precedence right-child rule

When parent and child precedence are equal:
- Parenthesize right child unless `(parent_op, child_op)` is in a small allowlist of semantically safe combinations.
- Keep left child unparenthesized for left-associative conventions unless required by lower precedence.

A conservative approach is acceptable initially (more parentheses over wrong semantics).

### 3) Route both formatters through shared helper

- Replace duplicated local `write_child` logic in `BinaryExpr::fmt`.
- Replace duplicated local `write_child` logic in `SqlDisplay::fmt`.

### 4) Extend tests

Add/expand regression coverage for:
- Equal-precedence non-associative right-child cases:
  - `1 - (2 - 3)`
  - `1 - (2 + 3)`
  - `1 / (2 * 3)`
- Existing precedence-only cases remain unchanged:
  - `(1 + 2) * 3`
  - `1 + 2 * 3`
  - `a AND b OR c`
- Cross-path consistency checks where appropriate (`Display` and `human_display`).

## Acceptance Criteria

1. `BinaryExpr::fmt` and `SqlDisplay::fmt` no longer duplicate binary-child parenthesizing logic.
2. Parenthesis insertion is side-aware and semantically safe for equal-precedence right children.
3. New tests covering ambiguous equal-precedence cases pass.
4. Existing display/human-display tests continue to pass (except expected updates where previous output was semantically ambiguous).
5. Code remains crate-scoped and does not introduce unrelated refactors.

## Implementation Checklist

- [ ] Add shared binary formatting helper and supporting side enum/flags.
- [ ] Migrate `BinaryExpr::fmt` to shared helper.
- [ ] Migrate `SqlDisplay::fmt` to shared helper.
- [ ] Add equal-precedence right-child regression tests.
- [ ] Validate unchanged output for non-ambiguous precedence examples.
- [ ] Run targeted tests in `datafusion-expr`.

## Test Plan

Suggested targeted commands:

```bash
cargo test -p datafusion-expr test_human_display_binary_expr_parens
cargo test -p datafusion-expr test_display_binary_expr
cargo test -p datafusion-expr
```

If test names differ after implementation, run crate-scoped tests and ensure any changed snapshots/expectations are intentional and documented.

## Risks and Mitigations

1. **Risk:** Output changes may affect explain-plan expectations.
   **Mitigation:** Prefer conservative parenthesizing and update tests explicitly with rationale.

2. **Risk:** Over-generalized helper becomes hard to read.
   **Mitigation:** Keep helper focused on binary parenthesizing only; avoid introducing broad formatting framework.

3. **Risk:** Incomplete operator pair safety matrix.
   **Mitigation:** Start conservative; add allowlist only for verified safe combos with tests.

## References

- Review context: `PR_REVIEW_01.md`
- Primary code locations:
  - `datafusion/expr/src/expr.rs` (`BinaryExpr::fmt` and `SqlDisplay::fmt`)
  - `datafusion/expr-common/src/operator.rs` (`Operator::precedence`)

## Labels (suggested)

- `refactor`
- `expr`
- `good first issue` (optional, if maintainers agree scope is approachable)
- `tech debt`
