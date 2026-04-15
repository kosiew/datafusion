stale
# Centralize join predicate placement policy in the unparser

## Summary

The SQL unparser currently decides join predicate placement inline inside the `LogicalPlan::Join` arm in `datafusion/sql/src/unparser/plan.rs`. Recent changes added a `JoinType::Inner` special case that routes extracted `TableScan` filters to `WHERE`, while outer joins keep those filters on the `ON` side by folding them together with any existing join filter.

That behavior is correct, but the policy is now encoded directly in a large control-flow branch that also handles relation rewriting, projection plumbing, and join construction. This makes the code harder to scan and raises the cost of future changes, especially if dialect-specific placement rules need to be introduced.

## Is your feature request related to a problem or challenge?

Yes.

The current implementation mixes two separate concerns in one place:

1. Extracting `TableScan` filters from the join inputs.
2. Deciding where those predicates should be emitted in the final SQL (`WHERE` for inner joins, `ON` for outer joins).

Today, the join arm in `datafusion/sql/src/unparser/plan.rs` performs all of the following inline:

- append extracted filters to the top-level `SELECT` predicate for inner joins,
- preserve `join.filter` as the actual join predicate for inner joins,
- combine extracted filters with `join.filter` using `AND` for outer joins,
- pass the resulting predicate into `join_constraint_to_sql`.

This works, but it leaves the placement policy distributed across low-level plumbing. That makes it harder to reason about the intended contract, review future changes, and add alternative behavior for dialects that treat `JOIN ON` predicates differently.

## Describe the solution you'd like

Introduce a small policy/helper layer for extracted join filters so the `LogicalPlan::Join` branch stays focused on orchestration.

One reasonable direction would be to move the placement logic behind one or two helpers with responsibilities such as:

- deciding whether extracted filters belong in `WHERE` or `ON` for the current join type,
- pushing filters into the `SELECT` selection when the policy chooses `WHERE`,
- combining extracted filters with `join.filter` when the policy chooses `ON`.

The exact API can vary, but the result should make the join arm read more like:

- extract filters,
- apply predicate-placement policy,
- build the join constraint,
- continue with join-specific SQL generation.

That would make the policy explicit and give future work a dedicated place to express dialect-specific variations.

## Acceptance criteria

- The predicate-placement decision for extracted table-scan filters is no longer spelled out inline in the main `LogicalPlan::Join` control flow.
- The `LogicalPlan::Join` arm in `datafusion/sql/src/unparser/plan.rs` becomes shorter and easier to scan.
- Existing behavior is preserved:
  - inner joins continue placing extracted table-scan filters in `WHERE`,
  - outer joins continue keeping those filters on the `ON` side,
  - existing join filters still combine correctly with extracted filters.
- Existing regression coverage continues to pass, and any new helper-specific tests are added where they make the policy easier to verify.

## Describe alternatives you've considered

- Leave the current inline implementation as-is. This has the lowest short-term cost, but it keeps predicate-placement policy embedded inside an already large join branch.
- Push the whole problem into dialect-specific code immediately. That may be useful later, but it is a larger change than necessary if the immediate goal is to make the current behavior easier to understand and maintain.
- Refactor only the `AND` folding into a helper and keep the placement decision inline. This would help somewhat, but it would still leave the main policy split across the join branch.

## Additional context

- Current behavior lives in `datafusion/sql/src/unparser/plan.rs` in the `LogicalPlan::Join` handling.
- The motivating scenario is that some dialects reject subqueries inside `JOIN ON`, so inner joins now emit extracted table-scan filters into `WHERE` instead.
- This issue is intentionally scoped as a refactor. It should preserve current semantics rather than introduce new unparser behavior.
- Related PR review context: `PR_REVIEW_01.md`.
