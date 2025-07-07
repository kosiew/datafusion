# Feature Request: Projection Pushdown for Recursive CTEs

## Problem Description

The `OptimizeProjections` optimizer rule in DataFusion currently skips `LogicalPlan::RecursiveQuery` nodes. This means that projection pushdown, which aims to eliminate unnecessary columns from intermediate query plans, does not occur within recursive Common Table Expressions (CTEs). As a result, DataFusion may read more data than strictly necessary from the underlying data sources, even if certain columns are not used in the final projection or intermediate computations of the recursive CTE.

This behavior can lead to:
- Increased I/O operations, especially for wide tables.
- Higher memory consumption during query execution.
- Suboptimal query performance for recursive queries involving large datasets.

**Example:**

Consider the following Rust code that creates a Parquet file and then executes a recursive CTE query:

```rust
use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::sync::Arc;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
  println!("Creating parquet file with sample data...");

  let temp_dir = TempDir::new()?;
  let parquet_path = temp_dir.path().join("sample_data.parquet");

  let ids = Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  let parent_ids = Int64Array::from(vec![
  Some(0),
  Some(1),
  Some(1),
  Some(2),
  Some(2),
  Some(3),
  Some(4),
  Some(5),
  Some(6),
  Some(7),
  ]);
  let values = Int64Array::from(vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);

  let schema = Arc::new(Schema::new(vec![
  Field::new("id", DataType::Int64, false),
  Field::new("parent_id", DataType::Int64, true),
  Field::new("value", DataType::Int64, false),
  ]));

  let record_batch = RecordBatch::try_new(
  schema.clone(),
  vec![Arc::new(ids), Arc::new(parent_ids), Arc::new(values)],
  )?;

  let file = File::create(&parquet_path)?;
  let props = WriterProperties::builder().build();
  let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
  writer.write(&record_batch)?;
  writer.close()?;

  println!("Parquet file created at: {:?}", parquet_path);

  let ctx = SessionContext::new();
  ctx.register_parquet(
  "hierarchy",
  parquet_path.to_str().unwrap(),
  ParquetReadOptions::default(),
  )
  .await?;

  println!("\nOriginal data:");
  let df = ctx.sql("SELECT * FROM hierarchy ORDER BY id").await?;
  df.show().await?;

  let recursive_query = "
  EXPLAIN ANALYZE
  WITH RECURSIVE number_series AS (
  SELECT id, 1 as level
  FROM hierarchy
  WHERE id = 1

  UNION ALL

  SELECT ns.id + 1, ns.level + 1
  FROM number_series ns
  WHERE ns.id < 10
  )
  SELECT * FROM number_series ORDER BY id
  ";

  println!("==> Explain plan for recursive query:");
  let recursive_df = ctx.sql(recursive_query).await?;
  recursive_df.show().await?;

  Ok(())
}
```

When running this code, the `EXPLAIN ANALYZE` output for the `DataSourceExec` shows that all columns (`id`, `parent_id`, `value`) are being projected, even though the `value` column is not used in the recursive CTE:

```
DataSourceExec: file_groups={1 group: [[var/folders/6z/kt4t6jkd4ss1_fj16dv_05xc0000gn/T/.tmpOjZiaN/sample_data.parquet]]}, projection=[id, parent_id, value], file_type=parquet, predicate=id@0 = 1, pruning_predicate=id_null_count@2 != row_count@3 AND id_min@0 <= 1 AND 1 <= id_max@1, required_guarantees=[id in (1)], metrics=[output_rows=10, elapsed_compute=1ns, bytes_scanned=565, file_open_errors=0, file_scan_errors=0, num_predicate_creation_errors=0, page_index_rows_matched=10, page_index_rows_pruned=0, predicate_evaluation_errors=0, pushdown_rows_matched=0, pushdown_rows_pruned=0, row_groups_matched_bloom_filter=0, row_groups_matched_statistics=1, row_groups_pruned_bloom_filter=0, row_groups_pruned_statistics=0, bloom_filter_eval_time=149.084µs, metadata_load_time=483.918µs, page_index_eval_time=124.959µs, row_pushdown_eval_time=2ns, statistics_eval_time=336.959µs, time_elapsed_opening=1.14175ms, time_elapsed_processing=1.198ms, time_elapsed_scanning_total=256.125µs, time_elapsed_scanning_until_data=232.5µs]
```

The `projection=[id, parent_id, value]` clearly shows `value` being read, despite not being referenced in the query.

## Proposed Solution

Enhance the `OptimizeProjections` rule to correctly handle `LogicalPlan::RecursiveQuery` nodes. This would involve:

1.  **Propagating Required Columns:** The optimizer needs to determine the set of columns required by both the `static_term` and `recursive_term` of the CTE.
2.  **Schema Compatibility:** Ensure that the projection applied to the `static_term` and `recursive_term` maintains schema compatibility, as the recursive term's output becomes the input for subsequent iterations. This is a key challenge, as the schema can evolve.
3.  **Handling Iterative Nature:** The optimization must account for the iterative nature of recursive CTEs, ensuring that columns necessary for the recursion (e.g., the `id` and `level` in the example) are always preserved.

This would likely involve modifying the `optimize_projections` function to:
-   Recursively call itself on the `static_term` and `recursive_term` with the appropriate `RequiredIndices`.
-   Carefully manage the schema transformations to avoid `FieldNotFound` errors or other schema mismatches during optimization.

## Benefits

-   **Improved Performance:** By reading only the necessary columns, I/O and CPU overhead will be reduced, leading to faster execution of recursive queries.
-   **Reduced Memory Usage:** Less data in memory means lower memory footprint, which is crucial for large datasets and resource-constrained environments.
-   **More Efficient Query Plans:** The generated execution plans will be more optimized and reflect the actual data requirements of the query.
-   **Consistency:** Aligning the behavior of recursive CTEs with other query types in terms of projection pushdown will lead to a more consistent and predictable optimization experience.

```