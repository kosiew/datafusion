# PR review responses

## neilconway

> Shouldn't we be computing the CTE's logical schema by widening the anchor and recursive schemas?

Concede.

Your understanding of the current behavior is accurate: this PR currently uses the anchor/static schema as the recursive CTE's declared physical output schema, so an anchor column inferred as non-nullable can force the recursive side to advertise non-nullability. If the recursive side then actually emits a NULL, the adapter can turn that into a runtime error.

That is too strong for SQL semantics. A recursive CTE is union-like: the anchor term should provide the output names / declared shape, but nullability should be compatible with both the anchor and recursive terms. Your example should not fail merely because the anchor is `0 AS n` and the recursive term can produce `NULL`.

I will revise the approach so the logical recursive CTE schema is widened for nullability across the static and recursive terms, while preserving the anchor/static field names. Then physical planning can align both children to that widened declared schema instead of narrowing nullable recursive output to a non-null anchor schema.

Planned fixes:

- add/adjust logical planning so recursive CTE output nullability is `static_nullable || recursive_nullable`, analogous to `UNION` compatibility;
- keep anchor/static names as the exposed CTE names, so the original recursive-term-name leak remains fixed;
- remove the nullable-to-non-null alignment path for recursive CTE output;
- add the NULL-producing recursive CTE test from this thread;
- restore the existing SLT coverage to `0 AS level` and verify it passes because the CTE schema is widened correctly, not because the SQL was rewritten to make the anchor nullable.
