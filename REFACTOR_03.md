# Refactor opportunities for `d47c96268^..631040e6b`

Scope: inspected cumulative diff, ignoring commits whose messages start with `UNPICK`. Suggestions below are behavior-preserving, local, and avoid public API changes.

## Safe simplifications

1. **Factor repeated work-table scan construction in `datafusion/sql/src/cte.rs`**

   The same `create_cte_work_table` + `LogicalPlanBuilder::scan(...).build()?` sequence appears for the initial anchor schema and the widened schema path (`cte.rs:136-147`, `cte.rs:188-199`). A small private helper/closure can return `(Arc<dyn TableSource>, LogicalPlan)` for a given schema.

   Behavior preserved: same schema, name, projection, and table source construction; only removes duplicated plumbing.

2. **Use `Field::with_metadata` in recursive schema builders**

   Both recursive schema builders create a mutable field only to call `set_metadata`:

   - `datafusion/expr/src/logical_plan/plan.rs:2294-2303`
   - `datafusion/physical-plan/src/recursive_query.rs:402-408`

   These can be shortened to `Field::new(...).with_metadata(...)` and then wrapped in `Arc::new(...)`.

   Behavior preserved: `with_metadata` sets the same metadata map and returns the field.

3. **Remove unnecessary `Result` wrapping in logical `recursive_query_schema` field map**

   In `datafusion/expr/src/logical_plan/plan.rs:2287-2305`, the `.map(...)` closure never returns an error. It wraps each field in `Ok(...)` only so the collect can be `collect::<Result<Vec<_>>>()?`.

   This can be simplified to collect a plain `Vec<_>` and keep the existing `DFSchema::new_with_metadata(...)?` as the only fallible operation.

   Behavior preserved: same fields and same final schema validation/error source.

4. **Update stale comment in `RecursiveQueryExec::try_new`**

   `datafusion/physical-plan/src/recursive_query.rs:93-95` still says the static term is the declared output schema. The code now derives a widened schema from static + recursive term, then aligns both children.

   Suggested wording: "Use the static term names with nullability widened across static and recursive terms; align both children at plan construction time."

   Behavior preserved: comment-only cleanup, but prevents future confusion.

5. **Reduce duplicated `planner_context.remove_cte(cte_name)` in widened recursive path if cleanup semantics stay identical**

   `datafusion/sql/src/cte.rs:201-208` removes the CTE in both success branches. This can be made more compact by assigning the branch result to a local and removing once after the branch.

   Caveat: preserve current error-path behavior. Do not move cleanup past fallible operations if that intentionally changes whether cleanup runs after an error.

## Avoid / semantic changes

- Do not merge logical and physical recursive schema helpers across crates. They operate on different schema types (`DFSchema` vs Arrow `Schema`) and have different metadata/error behavior; sharing would be broader than a low-risk refactor.
- Do not change `align_plan_to_schema` error selection by pre-validating or returning `project_plan_to_schema` errors directly. Current behavior intentionally falls back to `SchemaAlignExec` when projection cannot express exact schema.
- Do not remove the recursive-term replanning in `cte.rs`; it prevents stale anchor-only nullability assumptions and fixes the observed SLT hang.
