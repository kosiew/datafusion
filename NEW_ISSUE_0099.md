source: pr-22999_a
# Refactor: Factor distinct-from unparsing into a shared helper

## Summary

`datafusion/sql/src/unparser/expr.rs` currently handles `Operator::IsDistinctFrom` and `Operator::IsNotDistinctFrom` in separate match arms with duplicated dialect-dispatch logic. Factor this logic into a small helper so the dialect-specific spelling, null-safe semantics, and parenthesization rules are encoded in one place.

## Context

PR apache/datafusion#22999 adds MySQL-specific unparsing for DataFusion's distinct-from operators:

- `a IS NOT DISTINCT FROM b` -> `a <=> b`
- `a IS DISTINCT FROM b` -> `NOT (a <=> b)`

The implementation adds similar `match self.dialect.distinct_from_style()` blocks in both operator arms. This duplication makes it easier for future changes to update one operator but miss the other, especially around SQL precedence and dialect-specific syntax.

## Problem

The invariant is: both `IS DISTINCT FROM` and `IS NOT DISTINCT FROM` must be emitted as a matched pair for each dialect, preserving null-safe equality semantics and unambiguous SQL parsing.

Duplicating the dialect mapping in two branches weakens that invariant because:

- Parenthesization fixes must be applied in both places.
- New dialect styles must update both operators consistently.
- Error messages and unsupported-dialect behavior can drift.
- The relationship between the two operators is not explicit in code.

## Proposed refactor

Add a private helper in `datafusion/sql/src/unparser/expr.rs`, for example:

```rust
fn distinct_from_to_sql(
    &self,
    left: ast::Expr,
    right: ast::Expr,
    negated: bool,
    original_expr: &Expr,
) -> Result<ast::Expr> {
    match self.dialect.distinct_from_style() {
        DistinctFromStyle::FullText => {
            let expr = if negated {
                ast::Expr::IsNotDistinctFrom(Box::new(left), Box::new(right))
            } else {
                ast::Expr::IsDistinctFrom(Box::new(left), Box::new(right))
            };
            Ok(ast::Expr::Nested(Box::new(expr)))
        }
        DistinctFromStyle::DiamondOperators => {
            let spaceship = ast::Expr::Nested(Box::new(ast::Expr::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::Spaceship,
                right: Box::new(right),
            }));

            if negated {
                Ok(spaceship)
            } else {
                Ok(ast::Expr::Nested(Box::new(ast::Expr::UnaryOp {
                    op: UnaryOperator::Not,
                    expr: Box::new(spaceship),
                })))
            }
        }
        DistinctFromStyle::Unsupported => {
            not_impl_err!("dialect does not support expression: {original_expr:?}")
        }
    }
}
```

Then both match arms become small and symmetrical:

```rust
Operator::IsDistinctFrom => {
    let left = self.expr_to_sql_inner(left.as_ref())?;
    let right = self.expr_to_sql_inner(right.as_ref())?;
    self.distinct_from_to_sql(left, right, false, expr)
}
Operator::IsNotDistinctFrom => {
    let left = self.expr_to_sql_inner(left.as_ref())?;
    let right = self.expr_to_sql_inner(right.as_ref())?;
    self.distinct_from_to_sql(left, right, true, expr)
}
```

Exact naming is flexible. A clearer enum than `negated: bool` would also be reasonable if preferred.

## Expected benefits

- Keeps null-safe equality and its negation in one abstraction.
- Reduces duplicate control flow in a large unparser function.
- Makes precedence-sensitive parenthesization harder to get wrong.
- Simplifies future dialect additions or syntax changes.
- Keeps the refactor local to `expr.rs` with no public API change.

## Suggested tests

Add or keep tests covering:

1. Default/full-text dialect:
   - `c1 IS DISTINCT FROM true`
   - `c1 IS NOT DISTINCT FROM true`

2. MySQL dialect:
   - `IS NOT DISTINCT FROM` unparses to `<=>`
   - `IS DISTINCT FROM` unparses as `NOT (<=>)` or another unambiguous equivalent, not ambiguous `NOT a <=> b`

3. If `DistinctFromStyle::Unsupported` remains possible, a small unit test or existing coverage should verify the unsupported path returns an actionable error.

## Scope

Small local cleanup. No behavior change intended beyond preserving existing correct output. If done after PR #22999 fixes the blockers, this should be a follow-up refactor only.
