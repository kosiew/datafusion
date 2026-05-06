# Issue 22034

# Issue Summary: Preserving Recursive CTE Declared Schema
## Background
**SQL Behavior Change**: A change introduced in issue #21912 requires modification of existing recursive Common Table Expressions (CTEs) from `0 AS level` to `SUM(0) AS level` to conform to new performance expectations.
**CTE Schema Declared Requirement**: Recursive CTEs must align their physical output schema with the declared static schema, including nullability.
## Problem Description
**Mismatch in Field Nullability**: A recursive CTE can have a static term that declares a non-nullable field (e.g., `0 AS level`), while the recursive term computes this field with potentially nullable expressions (e.g., `MIN(rs.level) + 1`).
**Schema Widening Issue**: The `RecursiveQueryExec` can create a physical schema that combines the nullability of both static and recursive terms, leading to a mismatch with the declared schema of the CTE.
## Evidence of the Issue
The SQL behavior change manifests in the amended SQL Logic Test (SLT), where changing:
```sql
0 AS level
```
to:
```sql
SUM(0) AS level
```
requires the handling of nullable output, inadvertently weakening the declared non-null static schema by making it nullable.
**Error Message**: If reverting to `0 AS level`, the execution results in the error:
```text
Physical input schema should be the same as the one converted from logical input schema.
Differences:
field nullability at index 2 [level]: (physical) true vs (logical) false.
```
## Conclusion
**Need for Fix**: To maintain schema integrity, particular focus is required to ensure that recursive CTE outputs preserve their declared static schema and nullability without forcing unnecessary SQL rewrites or widening schema.

URL: https://github.com/apache/datafusion/issues/22034
