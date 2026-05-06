# Refactor opportunities

Range: `739e1471b^..c0a606600`  
Ignored `UNPICK*` commits: none in range.

## Safe, behavior-preserving simplifications

1. `datafusion/physical-plan/src/common.rs:121`
   - `align_plan_to_schema` validates before calling `project_plan_to_schema` or `SchemaAlignExec::try_new`, but both paths validate their own contracts.
   - The initial validation can be removed while preserving behavior:
     - exact schema still returns unchanged
     - projection-compatible schemas still use `ProjectionExec`
     - nullability narrowing still falls through to `SchemaAlignExec`
     - count/type/metadata errors still come from `SchemaAlignExec::try_new` with `align` wording
   - Suggested shape:
     ```rust
     let input_schema = input.schema();
     if input_schema.as_ref() == expected_schema.as_ref() {
         return Ok(input);
     }

     if let Ok(projected) = project_plan_to_schema(Arc::clone(&input), expected_schema) {
         debug_assert_eq!(projected.schema().as_ref(), expected_schema.as_ref());
         return Ok(projected);
     }

     Ok(Arc::new(SchemaAlignExec::try_new(
         input,
         Arc::clone(expected_schema),
     )?))
     ```

2. `datafusion/physical-plan/src/common.rs:610` and `datafusion/physical-plan/src/common.rs:725`
   - The two “schema matches returns input” tests duplicate setup.
   - Add a small local helper for the common schema/input pair, or a helper that asserts pointer-preserving return for a passed function.
   - Behavior unchanged; test intent remains the same.
   - Conservative helper:
     ```rust
     fn single_i32_exec(name: &str, nullable: bool) -> Arc<dyn ExecutionPlan> {
         empty_exec(vec![Field::new(name, DataType::Int32, nullable)])
     }
     ```
   - This also shortens several later tests that repeatedly create a one-column `Int32` `EmptyExec`.

3. `datafusion/physical-plan/src/common.rs:775` and `datafusion/physical-plan/src/common.rs:787`
   - `align_plan_to_schema` and `project_plan_to_schema` mismatch tests repeat the same one-column inputs and expected schemas.
   - Reuse small test helpers for common expected schemas:
     ```rust
     fn single_field_schema(name: &str, data_type: DataType, nullable: bool) -> SchemaRef { ... }
     ```
   - Keep separate tests for align vs project so public helper contracts stay explicit.

4. `datafusion/physical-plan/src/common.rs:797` and `datafusion/physical-plan/src/common.rs:829`
   - Field metadata mismatch tests duplicate nearly identical schema construction for `align_plan_to_schema` and `project_plan_to_schema`.
   - Extract local helper returning `(Arc<dyn ExecutionPlan>, SchemaRef)` for the mismatched metadata case.
   - Same assertions and same errors, less repeated setup.

5. `datafusion/physical-plan/src/common.rs:813` and `datafusion/physical-plan/src/common.rs:845`
   - Schema metadata mismatch tests duplicate setup.
   - Extract a local helper similar to the field metadata case.
   - This is test-only and preserves coverage while shortening the file.

6. `datafusion/physical-plan/src/recursive_query.rs:515` and `datafusion/physical-plan/src/recursive_query.rs:539`
   - Recursive query tests repeat `RecursiveQueryExec::try_new("numbers".to_string(), ..., false)`.
   - A private test helper can shorten setup without changing production code:
     ```rust
     fn recursive_exec(
         static_term: Arc<dyn ExecutionPlan>,
         recursive_term: Arc<dyn ExecutionPlan>,
     ) -> Result<RecursiveQueryExec> {
         RecursiveQueryExec::try_new("numbers".to_string(), static_term, recursive_term, false)
     }
     ```

## Redundancy worth keeping unless clarity is improved elsewhere

1. `datafusion/physical-plan/src/common.rs:202`
   - `debug_assert_eq!(projection.schema().as_ref(), expected_schema.as_ref())` is technically redundant with the helper contract, but useful as a local invariant check.
   - Keep it unless projection construction is wrapped in a narrower helper that makes the invariant obvious.

2. `datafusion/physical-plan/src/common.rs:270`
   - `SchemaAlignExec::try_new` validation overlaps with callers like `align_plan_to_schema`.
   - Keep it because `SchemaAlignExec::try_new` is public and must defend its own invariant.

## Avoid / semantic-change risks

1. Do not collapse `align_plan_to_schema` and `project_plan_to_schema` into one public helper.
   - They encode different contracts: projection-only vs adapter-capable alignment.

2. Do not move schema rebinding back into `RecursiveQueryStream::push_batch`.
   - That would undo the plan-time alignment invariant and mask child schema bugs.

3. Do not preserve child hash partitioning in `SchemaAlignExec` as a “refactor”.
   - That may be a valid performance enhancement later, but it is semantic/optimizer-visible and not a small behavior-preserving cleanup.
