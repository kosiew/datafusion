# PR Review
owner: apache
repo: datafusion
pr_number: issue-22034

## Decision
- [x] Approve
- [ ] Approve with suggestions
- [ ] Request changes

## Blocking findings
No blocking findings. The implementation correctly follows the agreed design: logical recursive CTE output schema is widened to `static_nullable || recursive_nullable` (union-like), anchor/static field names are preserved, and the nullable-to-non-null physical alignment path is removed.

## Non-blocking suggestions
1. file: datafusion/expr/src/logical_plan/plan.rs
   line: 2332
   side: RIGHT
   body: Widening logical recursive CTE nullability to `static_nullable || recursive_nullable` is the correct approach per the agreed design. Confirm that field names are still taken from the static/anchor term (not the recursive term) so the original recursive-term-name leak remains fixed.

2. file: datafusion/physical-plan/src/recursive_query.rs
   line: 141
   side: RIGHT
   body: With the nullable-to-non-null alignment path removed, `try_new_with_schema` / `project_plan_to_schema` should now only need to handle non-null → nullable (safe cast) or equal-nullability cases. A small assertion or comment confirming this direction would make the invariant explicit and prevent future regressions.

3. file: datafusion/sqllogictest/test_files/cte.slt
   line: 1303
   side: RIGHT
   body: The new regression test correctly validates that a recursive CTE whose anchor is `SELECT 0 AS n` can emit `NULL` from the recursive term — this is the intended union-like behavior. Consider also adding a case that confirms the CTE output column name is still sourced from the anchor (`n`, not a renamed recursive column) to pin the name-preservation invariant alongside the nullability one.

## High-impact refactor opportunities (out of scope)
No out-of-scope refactor opportunities. The chosen approach — widen nullability at logical planning and align both physical children to the widened schema — gives a single, clear ownership point for the schema contract.

## Follow-up actions
- Verify anchor/static field names are the exposed CTE names in all paths touched by this PR.
- Confirm `project_plan_to_schema` is no longer invoked in any nullable-input → non-null-expected direction for recursive CTEs.
- Add/retain SLT coverage for `0 AS level` to verify it passes because the schema is widened correctly (not because the SQL was rewritten to make the anchor nullable).

Focused validation run:
- `cargo test -p datafusion-physical-plan recursive_query_exec` passed
- `cargo test -p datafusion-physical-plan project_plan_to_schema` passed
- `cargo test -p datafusion-sqllogictest --test sqllogictests -- cte` passed
