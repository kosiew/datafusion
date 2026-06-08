source: pr-22988_a
# Centralize DML target-column validation and value alignment

## Problem
DataFusion's SQL planner validates target columns and input values separately for each DML path. `INSERT` has mature checks for explicit column lists, duplicate target columns, missing/default columns, source-value alignment, and placeholder/type handling. `UPDATE` has its own assignment handling. New `MERGE INTO` planning adds another path for update assignments and insert column/value handling.

This creates a boundary bug: new DML features can construct a logical DML operation with malformed target-column metadata unless they remember to duplicate every existing rule. In PR apache/datafusion#22988, `MERGE INSERT` can be planned with unknown target columns, duplicate target columns, or a mismatch between insert columns and values.

Affected area:
- `datafusion/sql/src/statement.rs`
- DML logical operations in `datafusion/expr/src/logical_plan/dml.rs`
- `TableProvider` DML hooks in `datafusion/catalog/src/table.rs`

## Why it matters
Invalid target-column handling is a correctness and maintainability risk:

- SQL-visible errors may be delayed until execution or delegated to table providers.
- Different providers may accept/reject malformed DML differently.
- Future DML features must rediscover planner invariants by copying existing code.
- Column-name normalization, duplicate detection, value arity checks, defaults, and type coercion can drift between `INSERT`, `UPDATE`, and `MERGE`.

The SQL planner is the earliest trustworthy boundary that has the target schema, SQL identifiers, and source expressions together. It should produce a validated logical DML operation, not provider-specific cleanup work.

## Invariant / desired behavior
For every SQL-planned DML operation that names target columns:

- Every referenced target column exists after identifier normalization.
- The same target column cannot be assigned or inserted more than once.
- The number of supplied values matches the target column set required by the operation.
- Omitted target columns are represented consistently according to existing `INSERT` default/missing-column behavior.
- Expressions are resolved against the correct schema and coerced/typed consistently with existing DML planning rules.
- `TableProvider` hooks receive a logical DML operation that has already passed planner-level target-column validation.

## Proposed direction
Extract a small set of private SQL-planner helpers that own DML target-column validation and value alignment. Reuse them from existing `insert_to_plan`, `update_to_plan`, and new `merge_to_plan` / `merge_clause_to_plan` code.

Suggested shape:

- A helper to normalize and resolve target column names against a `DFSchema`.
- A helper to reject duplicate target columns and return target schema indices.
- A helper to validate/align explicit insert columns and values, including the empty-column-list case.
- A helper or typed result for update assignments, so assignment extraction and validation are not reimplemented per DML statement.

Keep the helper API narrow and planner-local unless another crate needs the contract. Prefer returning validated structures over booleans or partially validated vectors.

## Scope
### In
- Audit current DML target-column validation in SQL planning.
- Extract shared validation/alignment helpers in `datafusion/sql/src/statement.rs` or a nearby planner module.
- Rewire `INSERT`, `UPDATE`, and `MERGE INTO` planning to use the shared helpers where behavior overlaps.
- Preserve current valid `INSERT` and `UPDATE` behavior.
- Add regression coverage for shared invalid cases across relevant DML statements.
- Document the planner/provider contract for DML target-column validation if the public `TableProvider` hook docs need clarification.

### Out
- Implementing storage/provider-side `MERGE INTO` semantics.
- Changing SQL parser grammar for MERGE clause/action combinations.
- Reworking logical DML node layout beyond what is needed to carry already-validated data.
- Adding broad type-coercion semantics not already supported by existing DML paths.
- Changing provider-specific execution behavior except where it depended on malformed planner input.

## Acceptance criteria
- [ ] `INSERT` still rejects unknown columns, duplicate columns, and source/value arity mismatches with actionable planner errors.
- [ ] `MERGE INSERT` rejects unknown target columns, duplicate target columns, and columns/values arity mismatches before execution planning reaches a `TableProvider`.
- [ ] `MERGE INSERT` with no explicit column list validates against the full target schema or the documented default-column behavior.
- [ ] `UPDATE` assignment target validation uses the same target-column resolution rules as the shared helper where applicable.
- [ ] Shared helper names and return types make it clear which data is validated and which remains unresolved.
- [ ] `TableProvider` DML hook docs or comments state whether target columns and value arity are planner-validated.

## Tests / verification
- Unit tests for the extracted helper behavior if helpers are exposed to test modules.
- SQL planner or SQLLogicTest coverage for:
  - `INSERT` unknown target column.
  - `INSERT` duplicate target column.
  - `INSERT` source/value count mismatch.
  - `MERGE INTO ... WHEN NOT MATCHED THEN INSERT (...) VALUES (...)` with unknown target column.
  - `MERGE INTO ... WHEN NOT MATCHED THEN INSERT (...) VALUES (...)` with duplicate target column.
  - `MERGE INTO ... WHEN NOT MATCHED THEN INSERT (...) VALUES (...)` with too few / too many values.
  - `MERGE INTO ... WHEN NOT MATCHED THEN INSERT VALUES (...)` empty-column-list behavior.
- Existing DML tests continue to pass.
- If SQL-visible errors change, update expected error text in SLTs or planner tests.

## Notes / open questions
- Existing `INSERT` logic handles default/missing target columns through `value_indices`; confirm whether `MERGE INSERT` should support the same default-fill path or require explicit values for all target columns initially.
- `sqlparser` already rejects invalid MERGE clause/action combinations. This issue should not duplicate parser grammar validation in DataFusion unless DataFusion constructs `MergeIntoOp` from non-SQL sources and needs an additional logical-plan invariant.
