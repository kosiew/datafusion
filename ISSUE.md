# Issue 22034

# Issue Summary: Recursive CTE Schema Alignment and Nullability

## Background

Issue #21912 exposed a recursive CTE regression: an existing SLT query had to change from `0 AS level` to `SUM(0) AS level` to avoid a logical-vs-physical schema mismatch.

A later review clarified the desired SQL semantics: recursive CTE output is union-like for nullability. The anchor/static term should provide the exposed field names / shape, but output nullability should be widened across both the static and recursive terms.

## Problem Description

A recursive CTE can have:

- a non-null anchor expression, e.g. `0 AS level`
- a nullable recursive expression, e.g. `MIN(rs.level) + 1` or `CAST(NULL AS INT)`

The fix must satisfy both constraints:

1. Existing valid SQL such as `0 AS level` must not require rewriting to `SUM(0) AS level`.
2. If the recursive term can produce NULL, the recursive CTE output schema should be nullable rather than forcing a runtime non-null violation.

## Desired Contract

Recursive CTE schema derivation should be:

- field names / qualifiers: from the anchor/static term
- data types: compatible across static and recursive terms
- nullability: `static_nullable || recursive_nullable`
- metadata: consistent between logical and physical schemas

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

A NULL-producing recursive term should also work, e.g.:

```sql
WITH RECURSIVE t AS (
  SELECT 0 AS n
  UNION ALL
  SELECT CAST(NULL AS INT) AS n FROM t WHERE n IS NOT NULL
)
SELECT * FROM t;
```

Expected output includes both `0` and `NULL`.

## Conclusion

The fix should derive a consistent union-like recursive CTE schema at the logical layer, then ensure physical execution advertises and emits the same schema. It should preserve anchor field names, widen nullability when needed, and avoid SQL rewrites such as `0 AS level` -> `SUM(0) AS level`.

URL: https://github.com/apache/datafusion/issues/22034
