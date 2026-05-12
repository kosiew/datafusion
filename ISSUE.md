# Issue 22034

# Issue Summary: Recursive CTE Schema Alignment and Nullability

## Background

Issue #21912 exposed a recursive CTE regression: an existing SLT query had to change from `0 AS level` to `SUM(0) AS level` to avoid a logical-vs-physical schema mismatch.

Review discussion clarified two requirements:

1. Recursive CTE output names should come from the anchor/static term, not leak from the recursive term.
2. Recursive CTE nullability must be conservative enough to avoid invalid non-null assumptions while planning the recursive term.

A first follow-up direction was to derive recursive CTE output nullability as `static_nullable || recursive_nullable`, similar to `UNION`. That fixes nullable recursive outputs, but it is still not sufficient because the recursive term is planned against a self-reference schema before recursive output nullability reaches a fixed point.

## Problem Description

A recursive CTE can have:

- a non-null anchor expression, e.g. `0 AS level`
- a nullable recursive expression, e.g. `MIN(rs.level) + 1` or `CAST(NULL AS INT)`
- recursive predicates whose correctness depends on nullable self-reference columns, e.g. `a IS NOT NULL`

The fix must satisfy these constraints:

1. Existing valid SQL such as `0 AS level` must not require rewriting to `SUM(0) AS level`.
2. If the recursive term can produce NULL, the recursive CTE output schema must allow NULL rather than forcing a runtime non-null violation.
3. Recursive self-reference columns must not be planned with stale anchor-only non-nullability, because that can let optimizations remove filters needed for termination.

## Why `static_nullable || recursive_nullable` is not enough

A two-pass approach can still be unsound. The recursive term may be optimized before the later widened schema is known.

Example:

```sql
WITH RECURSIVE t(a, b) AS (
    SELECT 0 AS a, 0 AS b
    UNION ALL
    SELECT b AS a, CAST(NULL AS INT) AS b FROM t WHERE a IS NOT NULL
)
SELECT * FROM t;
```

If the work-table/self-reference schema treats `t.a` as non-nullable based on the anchor, the optimizer can elide `a IS NOT NULL`. That changes the recursion and can make the query non-terminating.

A fixed-point replanning approach could recover more precise nullability, but it is larger planner machinery: repeated recursive-term planning, convergence detection, iteration limits, and broader optimizer interaction risks.

## Desired Contract

Recursive CTE schema derivation should be conservative:

- field names / qualifiers: from the anchor/static term
- data types: validated compatible across static and recursive terms
- nullability: recursive CTE output and self-reference fields are nullable
- metadata: preserve static/logical schema metadata where applicable

The work-table/self-reference schema must be nullable before the recursive term is planned, so nullability-sensitive optimizations remain sound.

Physical planning should align both static and recursive children to this declared logical recursive CTE schema.

## Evidence of the Issue

The SLT workaround changed:

```sql
0 AS level
```

to:

```sql
SUM(0) AS level
```

This made the anchor nullable and hid the mismatch, but changed test semantics.

A NULL-producing recursive term should work:

```sql
WITH RECURSIVE t AS (
  SELECT 0 AS n
  UNION ALL
  SELECT CAST(NULL AS INT) AS n FROM t WHERE n IS NOT NULL
)
SELECT * FROM t;
```

Expected output includes both `0` and `NULL`.

The non-termination regression should also work and terminate:

```sql
WITH RECURSIVE t(a, b) AS (
    SELECT 0 AS a, 0 AS b
    UNION ALL
    SELECT b AS a, CAST(NULL AS INT) AS b FROM t WHERE a IS NOT NULL
)
SELECT * FROM t;
```

Expected output:

```text
0 0
0 NULL
NULL NULL
```

## Performance Impact

Marking recursive CTE columns nullable may miss some nullability-based optimizations inside recursive CTEs, such as eliding `IS NOT NULL` filters or removing null checks.

This is acceptable because:

- the scope is narrow: recursive CTE output/self-reference columns only;
- recursive CTEs are less common than normal scans, joins, and aggregates;
- DataFusion nullability analysis is already conservative in many places;
- correctness and termination are more important than precise recursive CTE nullability;
- this avoids fixed-point replanning complexity and planning overhead.

## Conclusion

The fix should derive a conservative nullable recursive CTE schema at the logical layer, use that nullable schema for the work-table/self-reference before planning the recursive term, and ensure physical execution advertises and emits the same schema.

It should preserve anchor field names, validate compatible data types, avoid SQL rewrites such as `0 AS level` -> `SUM(0) AS level`, and avoid stale non-null assumptions that can make recursive queries non-terminating.

URL: https://github.com/apache/datafusion/issues/22034
