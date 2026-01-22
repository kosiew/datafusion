**Overview**
- Thanks for the thoughtful review. Below are targeted responses and a concrete plan for each point. The goals are: (1) make predicate scoping robust for alias-qualified columns, and (2) avoid silently broadening DELETE/UPDATE scope when non-target predicates are present.

**Predicate Scoping (Aliases + Fail-Closed)**
- Issue: Alias-qualified columns (e.g., `x.id` from `UPDATE t AS x`) won’t match the base `TableReference`, and we currently risk silently dropping join conjuncts when extracting target-only filters in [datafusion/core/src/physical_planner.rs](datafusion/core/src/physical_planner.rs#L2017).
- Plan: Make `predicate_is_on_target` alias-aware and fail closed when any conjunct is not provably target-only.
  - Alias-aware check: Accept a small set of allowed qualifiers for the target: the base `TableReference` and any alias chosen in the DML (e.g., `t` and `x` for `UPDATE t AS x`). Evaluate each `Expr::Column` qualifier against this set using `resolved_eq`.
  - Where to get alias: We already have the target reference at extraction time; we will thread through an optional alias `TableReference` captured from the DML logical plan. If the planner only exposes the alias on the qualified columns themselves, we’ll build the allowed-set from: (a) the target base ref, and (b) the qualifier seen on the `TableScan` for the DML target.
  - Fail-closed behavior: When splitting the WHERE clause into conjuncts, if any conjunct references a non-target relation, treat the entire DML fast-path extraction as unsafe and return a planning error instead of dropping that conjunct. This prevents broadening scope when calling `TableProvider::delete_from`/`update`.
  - Rationale: Dropping join predicates (even a single conjunct) can expand the set of affected rows; failing closed preserves correctness. We can add a follow-up to support a fallback non-pushdown path for such cases, but until then we prefer correctness over convenience.
  - Implementation sketch:
    - Update [datafusion/core/src/physical_planner.rs](datafusion/core/src/physical_planner.rs#L2017): change `predicate_is_on_target(expr, target)` → `predicate_is_on_target(expr, allowed_refs: &[TableReference])` and check `column.relation ∈ allowed_refs` via `resolved_eq`.
    - During DML filter extraction, collect all conjuncts; if any `!predicate_is_on_target`, return a user-facing planning error indicating that mixed-target predicates are currently unsupported for the DML fast-path.
    - Keep `strip_column_qualifiers()` as-is for provider compatibility; it runs only after we have accepted the predicate.

**Test: Update From Drops Non-Target Predicates**
- Issue: The current test in [datafusion/core/tests/custom_sources_cases/dml_planning.rs](datafusion/core/tests/custom_sources_cases/dml_planning.rs#L657) uses columns (`id`, `status`) that exist on both `t1` and `t2`. After qualifier stripping, `expr_has_table_reference(.., "t2")` can false-negative.
- Plan: Make the leakage detection unambiguous by using a `t2`-only column in the WHERE clause.
  - Concretely: extend the `t2` schema with a unique column (e.g., `src_only TEXT`) and change the predicate to `... AND t2.src_only = 'active' ...`.
  - Assertion remains the same, but now any `t2` reference is guaranteed to be detectable even if qualifiers are stripped, because `src_only` does not exist on the target schema.
  - With the fail-closed change above, this test will also validate that we do not proceed with a broadened filter set when non-target columns appear.

**Follow-ups**
- Fallback execution path: As a separate enhancement, we can add a non-pushdown DML path for mixed-target predicates (e.g., execute the full join+filter and then apply row-level updates/deletes). This is more involved and will be proposed in a dedicated PR.
- Additional SLTs: Add/update SLTs to cover `UPDATE t AS x` and `DELETE FROM t USING ...` with alias-qualified predicates to ensure both alias resolution and fail-closed behavior are exercised.

**Summary**
- Fix alias handling by allowing both base and alias `TableReference`s in predicate checks.
- Fail closed on any non-target conjunct to avoid silent scope expansion.
- Strengthen the test by using a `t2`-only column to prevent false negatives after qualifier stripping.