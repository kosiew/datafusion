use arrow;
use arrow::array::{ArrayRef, Decimal128Array, DictionaryArray, Int32Array, RecordBatch};
use datafusion::error::Result;
use datafusion::prelude::*;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use std::fs::File;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Prepare record batch
    let array_values = Decimal128Array::from_iter_values(vec![10, 20, 30])
        .with_precision_and_scale(4, 1)?;
    let array_keys = Int32Array::from_iter_values(vec![0, 1, 2]);
    let array = Arc::new(DictionaryArray::new(array_keys, Arc::new(array_values)));
    let batch = RecordBatch::try_from_iter(vec![("col", array as ArrayRef)])?;

    // Write batch to parquet
    let file_path = "dictionary_decimal.parquet";

    let file = File::create(file_path)?;
    let properties = WriterProperties::builder()
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_bloom_filter_enabled(true)
        .build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(properties))?;

    writer.write(&batch)?;
    writer.flush()?;
    writer.close()?;

    // Prepare context
    let config = SessionConfig::default()
        .with_parquet_bloom_filter_pruning(true)
        .with_parquet_pruning(true)
        .with_collect_statistics(true);
    let ctx = SessionContext::new_with_config(config);

    ctx.register_parquet("t", file_path, ParquetReadOptions::default())
        .await?;

    let binding = ctx.table("t").await?;
    println!("{:?}", binding.schema());

    // In case pruning predicate not created (due to cast), there is a record in resultset
    let mut sql = "select * from t where col = 1";

    print_sql_result_and_plans(&sql, &ctx).await?;
    println!();

    // In case of triggered RowGroup pruning -- the only RowGroup eliminated while pruning by statistics
    sql = "select * from t where col = cast(1 as decimal(4, 1))";

    print_sql_result_and_plans(&sql, &ctx).await?;

    Ok(())
}

async fn print_sql_result_and_plans(sql: &str, ctx: &SessionContext) -> Result<()> {
    let df = ctx.sql(sql).await?;

    let physcial_plan = df.clone().create_physical_plan().await?;
    println!("==> physical plan - {:?}", physcial_plan);
    println!("==> result of sql");
    df.show().await?;
    Ok(())
}
