# Recursive CTE Nullability Fix Direction

## Choice

Use conservative nullable schemas for recursive CTE output and self-reference columns.

The alternative is fixed-point replanning: repeatedly plan the recursive term until recursive CTE nullability converges. That would be more precise, but it is larger planner machinery and not needed for this fix.

## Why conservative nullable columns

Recursive CTE planning is circular: the recursive term is planned against a self-reference schema before the final recursive output schema is fully known. A two-pass approach is still unsound because the recursive term can be optimized using stale, anchor-only non-nullability before later widening occurs.

Example failure mode:

```sql
WITH RECURSIVE t(a, b) AS (
    SELECT 0 AS a, 0 AS b
    UNION ALL
    SELECT b AS a, CAST(NULL AS INT) AS b FROM t WHERE a IS NOT NULL
)
SELECT * FROM t;
```

If `t.a` is treated as non-nullable while planning the recursive term, the optimizer can remove `a IS NOT NULL`, causing non-termination. Marking recursive CTE self-reference columns nullable before recursive-term planning prevents that unsound optimization.

## Performance impact

The expected performance impact is small and acceptable.

Benefits:

- Simpler planning path.
- No repeated recursive-term planning.
- No fixed-point convergence loop.
- No max-iteration or non-convergence failure mode.
- Less planner complexity and fewer correctness risks.

Costs:

- Some nullability-based optimizations may be missed inside recursive CTEs.
- `IS NOT NULL` filters on recursive CTE columns may be retained rather than elided.
- Expressions may keep conservative null checks.
- Physical alignment may add projections/casts to advertise the nullable schema.

Scope is narrow: this affects recursive CTE output/self-reference columns only. Recursive CTEs are less common than normal scans, joins, and aggregates, and DataFusion nullability analysis is already conservative in many places.

## Why not fixed-point replanning now

Fixed-point replanning could recover more precise nullability, but it adds significant complexity:

- repeated planning of the recursive term;
- convergence detection;
- iteration limits / non-convergence handling;
- more difficult optimizer interaction reasoning;
- broader test surface.

That complexity is not justified for this PR. Correctness and termination are more important than preserving precise non-nullability for recursive CTE columns.

## Resulting contract

Recursive CTE schema derivation should be:

- field names / qualifiers: from the anchor/static term;
- data types: validated compatible across static and recursive terms;
- nullability: conservative nullable for recursive CTE output/self-reference fields;
- metadata: preserve static/logical schema metadata where applicable.

Physical planning should align static and recursive children to that declared logical recursive CTE schema.
