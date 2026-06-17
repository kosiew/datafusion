source: pr-22902_a
# Refactor: Make join projection pushdown schema-aware via `ColumnIndex` / `JoinSide`

## Summary

`try_pushdown_through_join` currently decides whether projected join-output columns belong to the left or right child by comparing output column indices with `left.schema().fields().len()`. This encodes an implicit `left ++ right` output-schema invariant that is not true for all join types. Mark joins append a synthetic boolean `mark` column that does not originate from either child, and other non-inner join output shapes also depend on join type.

Refactor join projection pushdown to use the join's existing output column origin metadata (`ColumnIndex { index, side }`) instead of raw output-column position. This makes the helper enforce the actual join schema contract directly and can allow safe projection pushdown for mark joins rather than disabling it.

## Background

The panic fixed in PR #22902 happened because a subquery was planned as a mark join. Physical projection pushdown tried to split parent projected columns into left-child and right-child groups with `join_table_borders(left_field_count, projection_as_columns)`. That split assumes every join output field comes from either the left or right input and that the left fields occupy `0..left_len`, followed by right fields.

Mark joins break that assumption:

- `LeftMark`: output is left fields plus a synthetic `mark` boolean.
- `RightMark`: output is right fields plus a synthetic `mark` boolean.
- The synthetic `mark` field has `JoinSide::None` in `build_join_schema` and does not correspond to either child.

PR #22902 avoids the panic by skipping child projection pushdown for `LeftMark` and `RightMark`, falling back to embedding the projection into the join. That is correct as a bug fix, but leaves the underlying abstraction fragile: the helper still reasons from output position rather than the explicit output-origin contract.

Relevant code:

- `datafusion/physical-plan/src/projection.rs`
  - `try_pushdown_through_join`
  - `join_table_borders`
  - `join_allows_pushdown`
  - `new_join_children`
  - `update_join_on`
  - `update_join_filter`
- `datafusion/physical-plan/src/joins/utils.rs`
  - `build_join_schema`
  - `ColumnIndex`
- Call sites:
  - `datafusion/physical-plan/src/joins/hash_join/exec.rs`
  - `datafusion/physical-plan/src/joins/nested_loop_join.rs`

## Problem

The current helper's core invariant is wrong for some join schemas:

```rust
projection_column.index() < left_table_column_count // left side
projection_column.index() >= left_table_column_count // right side
```

This works only when join output fields are exactly `left fields ++ right fields`. It does not model:

- synthetic fields (`JoinSide::None`), such as mark join `mark`
- join types whose output is only one side (`Semi`, `Anti`)
- future join output shapes where output field order differs from child order

The result is a helper that can silently route columns to the wrong child or fail later in less obvious projection-rewrite code.

## Proposed refactor

Update projection pushdown through joins to be driven by join output origin metadata.

### Desired invariant

For each projected output column:

1. Use the join's `column_indices[output_index]` to identify its origin.
2. If `side == JoinSide::Left`, include the corresponding child column in the left child projection.
3. If `side == JoinSide::Right`, include the corresponding child column in the right child projection.
4. If `side == JoinSide::None`, do not push that column to either child. Preserve it at the join output layer, or decline full child pushdown if the current helper cannot represent the residual safely.

The helper should never infer child ownership from output index alone.

### Suggested implementation shape

1. Extend or replace `try_pushdown_through_join` so it accepts join output origin mapping:

   ```rust
   pub fn try_pushdown_through_join(
       projection: &ProjectionExec,
       join_left: &Arc<dyn ExecutionPlan>,
       join_right: &Arc<dyn ExecutionPlan>,
       join_on: JoinOnRef,
       join_schema: &SchemaRef,
       column_indices: &[ColumnIndex],
       filter: Option<&JoinFilter>,
   ) -> Result<Option<JoinData>>
   ```

2. Convert projected parent columns into a side-aware representation:

   ```rust
   enum ProjectedJoinColumn {
       Left { output_index: usize, child_index: usize, alias: String },
       Right { output_index: usize, child_index: usize, alias: String },
       Synthetic { output_index: usize, alias: String },
   }
   ```

   Exact type/name can vary; key point is to preserve output index, child index, side, and alias explicitly.

3. Replace `join_table_borders` and contiguous left/right split logic with side grouping.

4. Rebuild child projections from the side groups, preserving child field names/aliases expected by `update_join_on` and `update_join_filter`.

5. Rebuild the join with projected children and a join-level projection/residual if needed to preserve output order, aliases, and synthetic columns.

6. Remove the mark-join special-case guard from `HashJoinExec::try_swapping_with_projection` and `NestedLoopJoinExec::try_swapping_with_projection` once mark joins are safely handled.

## Design considerations

### Synthetic `mark` column

The mark column should not be pushed into either child. It is produced by the join itself. If the parent projection requests child columns plus `mark`, the refactored logic should either:

- keep the mark column in the join output and apply a residual projection above the join, or
- embed a join projection that includes projected child output columns plus the mark column.

Avoid constructing a child `ProjectionExec` for `mark`.

### Output order and aliases

Current pushdown only works for simple column projections with non-mixed left/right groups. A side-aware implementation should be explicit about which cases it supports.

At minimum, preserve existing behavior for supported plain joins. For mark joins, preserve:

- output field count and order
- field names / aliases
- data types
- nullability
- the synthetic mark column position expected by the join type

If a projection reorder or alias cannot be represented safely by child projection plus join projection, return `Ok(None)` and let `try_embed_projection` handle it.

### Join filter and join-on rewrites

`update_join_on` and `update_join_filter` need mappings from original child indices to projected child indices. Build those mappings from `ColumnIndex`, not from output-column positions.

This avoids errors when the join output order is not child order.

### API boundaries

`build_join_schema` already computes `Vec<ColumnIndex>` alongside the schema. Prefer plumbing that existing metadata into the projection helper instead of recomputing or guessing.

## Acceptance criteria

- `try_pushdown_through_join` no longer uses output index ranges to infer left/right ownership.
- `join_table_borders` is removed or no longer used for join projection pushdown.
- `HashJoinExec` and `NestedLoopJoinExec` no longer need a special-case `LeftMark | RightMark` bypass for projection pushdown correctness.
- Existing projection pushdown behavior for regular inner/outer joins is preserved.
- Mark joins with projected child columns and the synthetic `mark` column do not panic and either push down safely or deliberately fall back.
- Schema-visible behavior is unchanged: final physical plan output schema matches the parent projection contract exactly.

## Test plan

Add focused tests covering both helper-level and SQL-visible behavior.

### SQLLogicTests

Extend `datafusion/sqllogictest/test_files/subquery.slt` with cases that generate mark joins and require projection pushdown to reason about both child columns and the `mark` column:

- `EXISTS` subquery that plans to hash mark join.
- `NOT EXISTS` / non-equi correlated subquery that plans to nested-loop mark join.
- Projection uses:
  - only outer table columns plus mark-consuming filter
  - reordered columns if supported, or verify fallback plan remains valid
  - aliases above the mark join if applicable

Assert both:

- query result
- `EXPLAIN` plan shape showing no panic and valid projection placement

### Rust unit tests

Add unit coverage near `projection.rs` helpers for side-aware mapping:

- plain `Inner` join behaves as before
- `LeftMark` maps left child columns to left and `mark` to `JoinSide::None`
- `RightMark` maps right child columns to right and `mark` to `JoinSide::None`
- unsupported synthetic-only or non-column projections return `Ok(None)` rather than panic
- join `on` and `filter` column indices are rewritten correctly after child projection

## Risks

- Incorrect residual projection handling could alter output field aliases/order.
- Filter/on remapping bugs could produce wrong join results rather than a visible panic.
- Over-eager pushdown through mark joins could incorrectly treat `mark` as a child column.

Mitigation: keep fallback behavior conservative. If a projection cannot be represented with exact schema preservation, return `Ok(None)` and use existing embedding behavior.

## Estimated effort

Medium.

Expected scope:

- one helper API refactor in `projection.rs`
- call-site updates in hash join and nested-loop join
- focused helper tests
- SLT regression cases for mark joins

No public API change expected.
