# Refactor opportunities

Range: `c158ee6e3^..bcac260c0`  
Ignored `UNPICK*` commits: none in range.

## Safe, behavior-preserving simplifications

1. `datafusion/physical-plan/src/common.rs:121`
   - `align_plan_to_schema` calls `input.schema()` twice before any plan change.
   - Store it once and reuse it. Same behavior, fewer `Arc` clones / virtual calls.
   - Suggested shape:
     ```rust
     let input_schema = input.schema();
     validate_schema_alignment(&input_schema, expected_schema, "align")?;

     if input_schema.as_ref() == expected_schema.as_ref() {
         return Ok(input);
     }
     ```

2. `datafusion/physical-plan/src/common.rs:273`
   - `SchemaAlignExec::try_new` repeats the partition count extraction for `Hash` and `UnknownPartitioning`.
   - Use `Partitioning::partition_count()` for the fallback arm. Same output: keep `RoundRobinBatch`, downgrade all others to `UnknownPartitioning(n)`.
   - Suggested shape:
     ```rust
     let partitioning = match &input_properties.partitioning {
         Partitioning::RoundRobinBatch(partitions) => {
             Partitioning::RoundRobinBatch(*partitions)
         }
         partitioning => {
             Partitioning::UnknownPartitioning(partitioning.partition_count())
         }
     };
     ```

3. `datafusion/physical-plan/src/common.rs:314`
   - `DisplayFormatType::TreeRender => write!(f, "")` can be `Ok(())`.
   - Same formatting behavior: writes nothing and returns success.
   - Suggested shape:
     ```rust
     DisplayFormatType::TreeRender => Ok(()),
     ```

4. `datafusion/physical-plan/src/recursive_query.rs:546`
   - Test repeatedly calls `static_term.schema()` for equality checks.
   - Bind once for readability and fewer `Arc` clones.
   - Suggested shape:
     ```rust
     let static_schema = static_term.schema();
     assert_eq!(exec.schema(), static_schema);
     assert_eq!(exec.static_term().schema(), static_schema);
     assert_eq!(exec.recursive_term().schema(), static_schema);
     ```

## Redundancy worth considering, but only if keeping invariants explicit

1. `datafusion/physical-plan/src/common.rs:127`
   - `align_plan_to_schema` calls `project_plan_to_schema`, then checks whether the returned schema equals `expected_schema`.
   - By `project_plan_to_schema`'s contract, any `Ok(projected)` should already have the exact expected schema, so the nested equality check is redundant.
   - However, the current check is defensive against a future regression in `project_plan_to_schema` / `ProjectionExec`. If removed, keep a `debug_assert_eq!` or add a comment tying it to the helper contract.
   - Conservative option:
     ```rust
     if let Ok(projected) = project_plan_to_schema(Arc::clone(&input), expected_schema) {
         debug_assert_eq!(projected.schema().as_ref(), expected_schema.as_ref());
         return Ok(projected);
     }
     ```

## Avoid / not recommended

1. Do not merge `align_plan_to_schema` and `project_plan_to_schema`.
   - The split is useful: projection-only helper vs stronger adapter-based alignment.
   - Merging would blur the contract and risks broader behavior changes.

2. Do not remove `SchemaAlignExec::try_new` validation because `align_plan_to_schema` already validates.
   - `SchemaAlignExec::try_new` is public and must defend its own invariant.

3. Do not simplify `SchemaAlignExec::execute` by always rebuilding `RecordBatch` schemas.
   - The current exact-schema fast path avoids column vector allocation and preserves existing batches unchanged.
