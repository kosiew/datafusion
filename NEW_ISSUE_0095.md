source: pr-22943_a
# Refactor: Thread `PlannerContext` through array literal planning

## Summary

Array literal planning currently resets planner state for each array element by constructing a fresh `PlannerContext`:

```rust
self.sql_expr_to_logical_expr(element, schema, &mut PlannerContext::new())
```

This should be refactored so array elements are planned with the caller's existing `PlannerContext`.

## Current code

- `datafusion/sql/src/expr/mod.rs`
  - `SQLExpr::Array(arr) => self.sql_array_literal(arr.elem, schema)`
- `datafusion/sql/src/expr/value.rs`
  - `sql_array_literal` creates `PlannerContext::new()` for every element.

## Problem

`PlannerContext` carries scoped planner state needed to plan expressions correctly. Resetting it inside array literals creates a special path where array elements do not see the same state as sibling expressions.

This can affect current and future planner state, including:

- prepared parameter type metadata
- lambda parameter scope
- subquery / outer query context
- expression-depth accounting added for deep expression protection
- future planner context fields that rely on consistent propagation

The immediate trigger was review of PR #22943, which added expression-depth tracking to prevent stack overflows from very deep flat binary expression chains. Because array element planning creates a fresh context, depth accounting can reset inside array literals rather than continuing from the enclosing expression.

## Desired behavior

Array literal elements should be planned using the same mutable `PlannerContext` passed to the enclosing expression planner.

Planning an array literal should behave like planning function arguments, struct fields, tuple elements, and other nested expression lists: child expressions inherit the surrounding planner state unless there is an explicit semantic reason to isolate them.

## Suggested implementation

Change `sql_array_literal` to accept `&mut PlannerContext`:

```rust
pub(super) fn sql_array_literal(
    &self,
    elements: Vec<SQLExpr>,
    schema: &DFSchema,
    planner_context: &mut PlannerContext,
) -> Result<Expr> {
    let values = elements
        .into_iter()
        .map(|element| self.sql_expr_to_logical_expr(element, schema, planner_context))
        .collect::<Result<Vec<_>>>()?;

    self.try_plan_array_literal(values, schema)
}
```

Then update the call site in `datafusion/sql/src/expr/mod.rs`:

```rust
SQLExpr::Array(arr) => self.sql_array_literal(arr.elem, schema, planner_context),
```

Check borrow/lifetime impact. If the direct iterator closure causes borrow friction, a simple loop is fine:

```rust
let mut values = Vec::with_capacity(elements.len());
for element in elements {
    values.push(self.sql_expr_to_logical_expr(element, schema, planner_context)?);
}
```

## Tests

Add focused tests showing array elements preserve planner context.

Useful coverage options:

1. Prepared parameters inside arrays preserve parameter type info.
   - Example shape: prepare or planner-level expression with `[$1]`.
   - Assert the placeholder inside the planned array carries the expected field/type.

2. Deep expression-depth accounting accumulates through array elements.
   - Build an expression where a nested array element contains a binary chain that should exceed the configured depth only if the existing `PlannerContext` is preserved.
   - Assert planning returns the expression-depth planning error rather than resetting depth inside the array.

3. Existing array literal behavior stays unchanged for normal literals.
   - Add or reuse a simple array literal planning assertion if needed.

Prefer the narrowest unit test in `datafusion/sql/src/expr/` if it can inspect the planned expression directly. If behavior is SQL-visible, add a small SQLLogicTest or planner integration test as appropriate.

## Acceptance criteria

- `sql_array_literal` no longer creates `PlannerContext::new()` for child elements.
- Array element planning receives the caller's `&mut PlannerContext`.
- Tests cover at least one planner-state value being preserved inside an array literal.
- Existing array literal planning tests continue to pass.
- No public API changes unless necessary.

## Risk / notes

This is a behavior-correctness refactor, not a feature. It may expose latent bugs where array literals accidentally relied on missing context. Those should be treated as correctness fixes unless there is an explicit SQL semantic reason for array elements to be isolated.
