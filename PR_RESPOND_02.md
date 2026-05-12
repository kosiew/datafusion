# PR review responses

## neilconway

> Shouldn't we be computing the CTE's logical schema by widening the anchor and recursive schemas?

Concede.

Your reading was correct: the earlier approach treated the anchor/static schema as the recursive CTE output schema, including non-nullability. That was too strict. Recursive CTEs are union-like here: anchor/static fields should determine the exposed column names, but nullability must be widened across the anchor and recursive terms.

I revised the PR accordingly:

- logical `RecursiveQuery` schema now uses anchor/static field names and `static_nullable || recursive_nullable`;
- physical `RecursiveQueryExec::try_new_with_schema` receives that logical schema and aligns both children to it;
- the nullable-recursive-to-non-null-output path is no longer the recursive CTE contract;
- the `0 AS level` SLT case is restored;
- added the NULL-producing recursive CTE regression from this thread;
- added coverage that recursive-term aliases do not leak into the exposed CTE column names.

So the sample query should now return `0` and `NULL`, not fail with a non-nullability runtime error. The `min(...)` nullability precision issue is separate planner improvement work.

## neilconway

> This query hangs now ... Is it really that big of a loss if we mark CTE columns as nullable?

Concede.

This exposes an unsoundness in the two-pass approach. The recursive term can be planned against a self-reference schema that is still too precise, so optimizations can remove `a IS NOT NULL` before the later widening step has enough information. In your example that makes the recursion non-terminating.

I agree the safer fix is to make recursive CTE output/self-reference columns nullable conservatively, rather than trying to compute precise recursive nullability here. A fixed-point planner would be more precise, but it is larger machinery and not needed for this PR. DataFusion already treats nullability conservatively in many places, and correctness/termination matter more than preserving non-nullability for recursive CTE columns.

I will revise the PR so recursive CTE columns are nullable while still preserving anchor/static names and data types.

Action plan:

1. Change logical recursive CTE schema derivation so every output field is nullable, with field names/qualifiers and metadata still sourced from the anchor/static term.
2. Ensure the recursive self-reference/work table uses that conservative nullable schema before the recursive term is optimized, so filters such as `a IS NOT NULL` are not elided from stale non-nullability.
3. Keep data type and column-count validation between anchor and recursive terms.
4. Keep physical planning aligned to the logical recursive CTE schema; both static and recursive children should advertise the same nullable schema.
5. Add the hanging query as an SLT regression test and verify it terminates with the expected rows.
6. Adjust existing unit tests that expected `static_nullable || recursive_nullable` to now expect nullable recursive CTE output fields.
7. Run focused checks:
   - `cargo test -p datafusion-physical-plan recursive_query_exec`
   - `cargo test -p datafusion-sqllogictest --test sqllogictests -- cte`
