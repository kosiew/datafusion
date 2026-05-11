# FIX_01: Remove Recursive CTE Logical No-op Projection Regression

## Evidence

The failing SLT diffs all have the same shape: extra `Projection` nodes appear inside the logical recursive term.

Examples:

- `select n + 1 FROM numbers ...` becomes:
  - original SQL projection: `Projection: numbers.n + Int64(1)`
  - extra schema-alignment projection: `Projection: numbers.n + Int64(1)`
- `SELECT time + 1 as time, ... FROM balances ...` gets an extra `Projection: time, balances.name, account_balance` above the real recursive projection.
- Recursive join gets several nested `Projection: l.start, r.end` nodes, showing the same alignment projection can be reintroduced during repeated logical plan reconstruction / optimizer passes.

The physical plan is mostly unchanged. Where it changes, it is the physical equivalent of the same logical no-op alignment projection.

The relevant code path is `RecursiveQuery::try_new` in `datafusion/expr/src/logical_plan/plan.rs`:

```rust
let schema = recursive_query_schema(static_term.schema(), recursive_term.schema())?;
let static_term = align_logical_plan_to_schema(static_term, Arc::clone(&schema))?;
let recursive_term = align_logical_plan_to_schema(recursive_term, schema)?;
```

`align_logical_plan_to_schema` wraps an input in a logical `Projection::try_new_with_schema(...)` whenever the child schema is not exactly equal to the desired recursive CTE schema.

That exact-equality check is too broad for the recursive term. The recursive term commonly has different expression-derived output names from the anchor/static term, even when it is already semantically compatible. Examples:

- static output name: `n`
- recursive expression name: `numbers.n + Int64(1)`

So the helper injects a no-op logical projection just to re-advertise names/nullability/metadata. That changes `EXPLAIN` output and can stack if `RecursiveQuery::try_new` is reached more than once by plan rewrites.

## Hypothesis

Root cause: recursive CTE schema reconciliation was implemented by mutating both logical children, including the recursive term, with schema-only `Projection` nodes.

This solved part of the schema contract, but at the wrong layer. Logical recursive term shape is user/optimizer-visible, so inserting schema-only projections there causes explain-plan regressions and repeated projection stacking.

The intended fix is narrower:

- logical recursive CTE output schema must be authoritative for work-table planning and physical planning;
- recursive child should not be rewritten with logical no-op projections just to match the output schema;
- physical planning should align physical children to the declared output schema.

## Smallest causal change

The causal change is the call:

```rust
let recursive_term = align_logical_plan_to_schema(recursive_term, schema)?;
```

in `RecursiveQuery::try_new`.

That call creates the extra logical projections shown in every failure.

The changes in `common.rs` / `SchemaAlignExec` are not the direct cause of these SLT diffs. The diffs are logical-plan shape changes from `align_logical_plan_to_schema` on the recursive term.

## Fix plan

1. Change `RecursiveQuery::try_new` so it does **not** align the recursive term logically.

   Use the recursive schema only to derive the desired output schema:

   ```rust
   let schema = recursive_query_schema(static_term.schema(), recursive_term.schema())?;
   let static_term = align_logical_plan_to_schema(static_term, Arc::clone(&schema))?;
   Ok(Self {
       name,
       static_term,
       recursive_term,
       is_distinct,
   })
   ```

   Rationale: `LogicalPlan::schema()` for `RecursiveQuery` currently returns `static_term.schema()`. Aligning only the static term lets the recursive query advertise widened output nullability without adding no-op projections to the recursive term.

2. Keep physical planning authoritative.

   In `datafusion/core/src/physical_planner.rs`, continue passing `node.schema().inner()` into `RecursiveQueryExec::try_new_with_schema(...)`.

   In `RecursiveQueryExec::try_new_with_schema`, keep physical `project_plan_to_schema(...)` for both physical children. Physical alignment belongs there and does not pollute logical `EXPLAIN`.

3. Guard against projection stacking.

   Add a unit test for `RecursiveQuery::try_new` where:

   - static term field name is `n`
   - recursive term expression field name differs, e.g. `n + 1`
   - assert `recursive_term` is not wrapped in an extra logical `Projection` solely for schema alignment.

4. Keep / add nullability test.

   Preserve coverage that recursive CTE output schema widens nullability when recursive term can produce nulls. With this plan, that widening should appear through the aligned static term schema.

5. Add SLT regression coverage.

   Re-run `datafusion/sqllogictest/test_files/cte.slt`. The expected logical plans should no longer contain the extra recursive-term projections shown in the failure.

## Non-goals

- Do not reintroduce broad `SchemaAlignExec`.
- Do not change public constructor signatures.
- Do not add required public fields to `RecursiveQuery`.
- Do not fix by updating snapshots to accept the extra projections; that hides the root cause.
