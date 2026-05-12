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

I will revise the PR so recursive CTE columns are nullable while still preserving anchor/static names and data types. I will also add this query as a regression test to ensure the `a IS NOT NULL` filter is not optimized away and recursion terminates with the expected rows.
