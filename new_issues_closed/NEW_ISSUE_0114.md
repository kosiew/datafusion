created #23668
source: pr-23327_a
# Centralize aggregate-scope expression rendering in the SQL unparser

## Problem
The SQL unparser now has several aggregate-aware rendering paths that manually compose the same small set of operations:

- unproject logical column references through an `Aggregate` with `unproject_agg_exprs`,
- detect whether the aggregate input is rendered through a derived-projection boundary with `contains_projection_before_relation`,
- normalize aggregate input column qualifiers with `normalize_agg_input_columns`,
- then render the expression into the appropriate SQL clause.

This composition is repeated in `datafusion/sql/src/unparser/plan.rs` for projection-over-aggregate rendering, direct `LogicalPlan::Aggregate` rendering, and `Filter` rendering for `HAVING` / `QUALIFY`. Nearby aggregate-aware paths, such as window projection and `ORDER BY` unprojection, have to remember which parts of the same rule apply.

The current code is mostly correct for the recently fixed `SELECT` / `GROUP BY` derived-projection shape, but the invariant is still distributed across clause-specific call sites. That makes future fixes easy to apply partially.

## Why it matters
This is primarily a maintainability refactor with correctness-risk reduction.

Aggregate unparsing has a scope boundary that is easy to miss: if an aggregate input is emitted as a derived relation, outer aggregate clauses must reference the derived relation output columns, not aliases from inside that relation. When each clause manually wires unprojection and normalization, reviewers must re-check the same invariant for every new path.

A small helper would make the intended scope rule obvious and reduce drift between `SELECT`, `GROUP BY`, `HAVING`, `QUALIFY`, and aggregate-aware `ORDER BY` handling.

## Invariant / desired behavior
For every aggregate expression rendered in one SQL `SELECT` block, column references should be normalized against the same visible SQL scope.

If an `Aggregate` reads from a child plan that the unparser renders through a derived-projection boundary, aggregate-related expressions emitted above that boundary should not retain out-of-scope base-table qualifiers for columns that belong to the aggregate input schema.

This invariant should be applied consistently where the unparser already performs aggregate unprojection or aggregate input normalization.

## Proposed direction
Introduce one narrow internal helper in `datafusion/sql/src/unparser/plan.rs` or a nearby unparser utility module.

The helper should centralize the common aggregate expression preparation step, for example:

1. optionally unproject an expression through an `Aggregate` when the caller needs unprojection;
2. apply aggregate-input qualifier normalization when the aggregate input crosses a derived-projection boundary;
3. return an `Expr` ready for `expr_to_sql` or `select_item_to_sql`.

The helper should stay small and semantic. It should not become a general SQL rendering abstraction or rewrite unrelated plan-node behavior.

Call sites should become clear about intent, for example:

- projection-over-aggregate: prepare projected expressions and group expressions through the helper;
- direct aggregate rendering: prepare aggregate and group expressions through the helper;
- `HAVING` / `QUALIFY`: prepare unprojected filter predicates through the helper;
- aggregate-aware `ORDER BY` / window-projection paths: either use the helper where the same scope rule applies, or document why those paths are already scoped differently.

## Scope
### In
- Add a focused internal helper for aggregate expression unprojection plus derived-input qualifier normalization.
- Migrate existing aggregate-aware call sites in `datafusion/sql/src/unparser/plan.rs` where the helper directly matches current behavior.
- Audit nearby paths that unproject aggregate expressions, especially `project_window_output` and `unproject_sort_expr` callers, and either route them through the helper or leave a short comment explaining why normalization is not applicable there.
- Preserve current SQL output for cases that do not cross a derived-projection boundary.
- Add focused regression coverage for the derived-projection aggregate-input shape across at least the clauses that are migrated.

### Out
- Do not redesign the SQL unparser.
- Do not change optimizer behavior.
- Do not change SQL formatting, alias generation, or dialect behavior except where required to preserve aggregate scope.
- Do not claim new support for unrelated nested-query, window, or sort shapes.
- Do not add public APIs unless a very small internal helper needs to move modules.

## Acceptance criteria
- [ ] Aggregate expression preparation has one canonical internal helper for unprojection plus derived-input qualifier normalization.
- [ ] Existing `SELECT` and `GROUP BY` behavior for derived-projection aggregate inputs is preserved.
- [ ] `HAVING` / `QUALIFY` paths use the same helper where they unproject aggregate expressions.
- [ ] Aggregate-aware `ORDER BY` and window-projection paths are audited; they either use the helper or document why the helper does not apply.
- [ ] Existing unparser tests pass.
- [ ] New or updated tests cover at least one derived-projection aggregate-input regression shape beyond the already-fixed `SELECT` / `GROUP BY` path, where practical.

## Tests / verification
- Add or update tests in `datafusion/core/tests/sql/unparser.rs` using an issue_23317-style schema / plan shape.
- Prefer tests that parse or execute the unparsed SQL where practical, not only string-fragment checks.
- Suggested focused command:

```bash
cargo test -p datafusion --test core_integration unparser -- --nocapture
```

Use narrower test filters during development as needed.

## Notes / open questions
- Existing helpers such as `contains_projection_before_relation`, `contains_aggregate_before_relation`, and `normalize_agg_input_columns` are likely sufficient building blocks. The refactor should mainly make their combined use harder to forget.
- Confirm whether `ORDER BY` can still hit the same derived-projection aggregate-input shape after current optimizer rewrites. If not, document that boundary instead of adding speculative logic.
- Confirm whether the `QUALIFY` branch has a stable test shape in the currently supported dialect setup. If not, still route existing code through the shared helper when behavior is unchanged.
