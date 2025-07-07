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

    let recursive_df = ctx.sql(recursive_query).await?;
    recursive_df.show().await?;

    Ok(())
}
