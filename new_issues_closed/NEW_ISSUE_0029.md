created #22668
source: pr-21801_a
# Centralize Unparser Subquery-Boundary Decisions

**Area**: SQL Unparser  
**Effort**: Medium  
**Complexity**: High  
**Priority**: Low (Refactor/Tech Debt)  

## Problem Statement

The `datafusion/sql/src/unparser/plan.rs` module contains multiple independent checks to decide whether a logical plan operator requires wrapping in a derived subquery (i.e., `SELECT ... FROM (SELECT ...) AS alias`). This decision logic is currently:

1. **Scattered** across multiple methods with similar but non-identical conditions
2. **Inconsistent** in which operators are covered by each check
3. **Difficult to maintain**, leading to risks of missed edge cases

### Current Fragmentation

- **`requires_derived_subquery()`** (line 1645): Checks for `Aggregate`, `Window`, `Sort`, `Limit`, `Union`
- **`window_input_requires_derived_subquery()`** (line 512): Checks for `Projection`, `Distinct`, `Limit`, `Sort`, `Union`
- **`SubqueryAlias` handler** (line 1062): Inline logic that calls `requires_derived_subquery()` with implicit assumptions
- **Union/Projection flattening** (throughout): Other boundary-sensitive logic scattered in various handlers

The disconnect between these checks creates subtle bugs where SQL clause evaluation order differs from logical-plan nesting. For example:

- A `Window` input containing `Projection` + `Limit` might or might not need a subquery boundary depending on which check is consulted
- `Sort` followed by `Aggregate` may produce incorrect SQL if the boundary decision changes
- `SubqueryAlias` wrapping certain operators may lose the required nesting structure

## Root Cause

Each handler was built independently to solve a specific operator's unparsing challenge. No single abstraction exists to answer: **"Can this logical plan be flattened into the parent SQL SELECT block without changing SQL evaluation semantics?"**

## Proposed Solution

Create a **single, centralized abstraction** that encodes the semantic rule:

```
A logical plan CAN be merged into its parent SELECT if and only if:
- It does NOT introduce a new SQL evaluation scope (e.g., GROUP BY, OVER, ORDER BY LIMIT)
- AND it does NOT constrain when earlier SQL clauses are evaluated
```

### Design Outline

1. **Create a unified predicate** `plan_requires_independent_select_scope(plan: &LogicalPlan, context: &UnparseContext) -> bool`
   - Single source of truth for all boundary decisions
   - Accepts optional context (e.g., "is this a Window input?" or "is this inside a SubqueryAlias?")
   - Documents why each operator type does/does not require a subquery

2. **Consolidate operator checks** into this predicate with clear comments for each case:
   - `Aggregate`: Always requires scope (GROUP BY + HAVING evaluation order)
   - `Window`: Always requires scope (OVER clause evaluation order)
   - `Sort`: Context-dependent (alone = ok, before Aggregate/Window = needs scope)
   - `Limit`: Context-dependent (alone = ok, before Sort = needs scope)
   - `Projection`: Context-dependent (simple projections ok, with `Distinct` or before Window = needs scope)
   - `Distinct`: Always requires scope (set semantics)
   - `Union`: Always requires scope (set operations have strict evaluation rules)
   - `SubqueryAlias`: Transparent (delegates to inner plan)

3. **Replace all existing checks** with calls to this unified predicate
   - Update `requires_derived_subquery()` to delegate to the new predicate
   - Update `window_input_requires_derived_subquery()` to delegate
   - Add optional `context` parameter for future extensibility (e.g., `UnparseContext::WindowInput`, `UnparseContext::SubqueryAliasChild`)

4. **Add regression tests** for boundary-sensitive combinations:
   - `Window(Sort(Aggregate(...)))` → all should wrap with derived subqueries
   - `Limit(Sort(Projection(...)))` → Sort must stay below Limit
   - `SubqueryAlias(Union(Sort(...)))` → Union requires derived scope
   - Any combination in PR #21801 that triggered the original fix

## Expected Benefits

- **Single point of maintenance**: Future boundary decisions only need one place to change
- **Reduced regression risk**: Context-dependent rules are explicit and tested
- **Clearer intent**: Developers understand *why* a plan needs a subquery by reading one method
- **Easier to extend**: Adding new operators or context-aware rules only requires one update
- **Better testability**: Each boundary case can be directly unit-tested against the predicate

## Scope & Constraints

- **In Scope**: Consolidating existing boundary logic; no behavioral changes
- **Out of Scope**: Changing which operators actually generate derived subqueries or SQL output
- **Constraint**: Must pass all existing SQLLogicTests without regression
- **Prerequisite**: PR #21801 (Window unparsing) should be merged first to minimize conflicts

## Implementation Outline

1. Create `fn plan_requires_independent_select_scope(plan: &LogicalPlan, context: Option<UnparseContext>) -> bool` with exhaustive `match` on plan types
2. Add `UnparseContext` enum: `enum UnparseContext { WindowInput, SubqueryAliasChild, GeneralChild }`
3. Document each case with a comment block explaining the SQL semantics
4. Replace calls to `requires_derived_subquery()` and `window_input_requires_derived_subquery()` where applicable
5. Add dedicated unit tests in a new `#[cfg(test)] mod boundary_tests` with combinations
6. Run full test suite: `cargo test -p datafusion-sql --test sql_integration plan_to_sql -- --nocapture`
7. Verify no SLT regressions: `cargo test --test sqllogictests` in `datafusion/sqllogictest`

## Acceptance Criteria

- [ ] New centralized predicate exists and is the single source of truth for all boundary decisions
- [ ] All existing boundary checks delegate to the new predicate (or are inlined with clear comments)
- [ ] At least 10 regression test cases covering sensitive operator combinations
- [ ] All SQLLogicTests pass (especially plan_to_sql tests)
- [ ] All unit tests pass: `cargo test -p datafusion-sql`
- [ ] PR review confirms the refactor maintains exact SQL output for all existing tests
- [ ] Code comments explain the SQL evaluation order semantics for each operator type

## Related Issues & PRs

- PR #21801: Window unparsing fix (triggers/precedes this refactor)
- `datafusion/sql/src/unparser/plan.rs`: Core file (~1850 lines)
- `datafusion/sql/src/unparser/rewrite.rs`: Projection/union flattening logic
- `datafusion/sqllogictest/test_files/plan_to_sql/`: SLT test coverage

## References

- [sqlparser-rs documentation](https://docs.rs/sqlparser/)
- DataFusion `LogicalPlan` enum: `datafusion/expr/src/logical_plan/mod.rs`
- Existing unparser module structure: `datafusion/sql/src/unparser/mod.rs`
