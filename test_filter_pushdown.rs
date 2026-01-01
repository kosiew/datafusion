use arrow::array::{ArrayRef, ListArray, StringArray};
use arrow::buffer::{OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::physical_plan::FileScanConfig;
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::*;
use parquet::arrow::ArrowWriter;
use std::fs::File;
use std::sync::Arc;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> datafusion::common::Result<()> {
    // Create temp directory and parquet file
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("test.parquet");

    // Create schema with a list column
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "list_col",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
    ]));

    // Write some test data
    let file = File::create(&file_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();

    let mut builder = arrow::array::ListBuilder::new(arrow::array::StringBuilder::new());
    builder.values().append_value("test");
    builder.append(true);

    let id_array = Arc::new(arrow::array::Int64Array::from(vec![1])) as ArrayRef;
    let list_array = Arc::new(builder.finish()) as ArrayRef;

    let batch = RecordBatch::try_new(schema.clone(), vec![id_array, list_array]).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    // Create context with pushdown enabled
    let mut session_config = SessionConfig::new();
    session_config
        .options_mut()
        .execution
        .parquet
        .pushdown_filters = true;
    let ctx = SessionContext::new_with_config(session_config);

    // Register parquet file
    ctx.register_parquet(
        "test_table",
        file_path.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;

    // Create query with array_has filter
    let sql = "SELECT * FROM test_table WHERE array_has(list_col, 'test')";
    let df = ctx.sql(sql).await?;

    // Get physical plan
    let plan = df.create_physical_plan().await?;

    // Print the plan
    println!(
        "Physical plan:\n{}",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
    );

    // Try to find DataSourceExec and check for filter
    fn check_plan(plan: &Arc<dyn ExecutionPlan>, depth: usize) {
        let indent = "  ".repeat(depth);
        println!("{}Checking: {}", indent, plan.name());

        if let Some(source_exec) = plan.as_any().downcast_ref::<DataSourceExec>() {
            println!("{}Found DataSourceExec", indent);
            if let Some(file_scan_config) = source_exec
                .data_source()
                .as_any()
                .downcast_ref::<FileScanConfig>()
            {
                println!("{}Found FileScanConfig", indent);
                println!(
                    "{}Filter: {:?}",
                    indent,
                    file_scan_config.file_source().filter()
                );
            } else {
                println!("{}DataSource is not FileScanConfig", indent);
            }
        }

        for child in plan.children() {
            check_plan(child, depth + 1);
        }
    }

    check_plan(&plan, 0);

    Ok(())
}
