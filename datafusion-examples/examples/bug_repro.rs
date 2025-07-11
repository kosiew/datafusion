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
    println!("Creating test data for bug reproduction...");

    let temp_dir = TempDir::new()?;
    let parquet_path = temp_dir.path().join("prices.parquet");

    // Create sample prices data with prices_row_num column
    let prices_row_nums = Int64Array::from((1..=40).collect::<Vec<i64>>());
    let prices = Int64Array::from((100..=139).map(|x| x * 10).collect::<Vec<i64>>());

    let schema = Arc::new(Schema::new(vec![
        Field::new("prices_row_num", DataType::Int64, false),
        Field::new("price", DataType::Int64, false),
    ]));

    let record_batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(prices_row_nums), Arc::new(prices)],
    )?;

    let file = File::create(&parquet_path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&record_batch)?;
    writer.close()?;

    println!("Test data created at: {:?}", parquet_path);

    let ctx = SessionContext::new();
    ctx.register_parquet(
        "prices",
        parquet_path.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;

    println!("\nRunning problematic recursive CTE query...");

    // This is the problematic query from cte.slt:409 that causes the optimization error
    let problematic_query = r#"
        WITH RECURSIVE "recursive_cte" AS (
          (
            WITH "min_prices_row_num_cte" AS (
              SELECT
                MIN("prices"."prices_row_num") AS "prices_row_num"
              FROM
                "prices"
            ),
            "min_prices_row_num_cte_second" AS (
              SELECT
                MIN("prices"."prices_row_num") AS "prices_row_num_advancement"
              FROM
                "prices"
              WHERE
                "prices"."prices_row_num" > (
                  SELECT
                    "prices_row_num"
                  FROM
                    "min_prices_row_num_cte"
                )
            )
            SELECT
              0.0 AS "beg",
              (0.0 + 50) AS "end",
              (
                SELECT
                  "prices_row_num"
                FROM
                  "min_prices_row_num_cte"
              ) AS "prices_row_num",
              (
                SELECT
                  "prices_row_num_advancement"
                FROM
                  "min_prices_row_num_cte_second"
              ) AS "prices_row_num_advancement"
            FROM
              "prices"
            WHERE
              "prices"."prices_row_num" = (
                SELECT
                  DISTINCT "prices_row_num"
                FROM
                  "min_prices_row_num_cte"
              )
          )
          UNION ALL (
            WITH "min_prices_row_num_cte" AS (
              SELECT
                "prices"."prices_row_num" AS "prices_row_num",
                LEAD("prices"."prices_row_num", 1) OVER (
                  ORDER BY "prices_row_num"
                ) AS "prices_row_num_advancement"
              FROM
                (
                  SELECT
                    DISTINCT "prices_row_num"
                  FROM
                    "prices"
                ) AS "prices"
            )
            SELECT
              "recursive_cte"."end" AS "beg",
              ("recursive_cte"."end" + 50) AS "end",
              "min_prices_row_num_cte"."prices_row_num" AS "prices_row_num",
              "min_prices_row_num_cte"."prices_row_num_advancement" AS "prices_row_num_advancement"
            FROM
              "recursive_cte"
              FULL JOIN "prices" ON "prices"."prices_row_num" = "recursive_cte"."prices_row_num_advancement"
              FULL JOIN "min_prices_row_num_cte" ON "min_prices_row_num_cte"."prices_row_num" = COALESCE(
                "prices"."prices_row_num",
                "recursive_cte"."prices_row_num_advancement"
              )
            WHERE
              "recursive_cte"."prices_row_num_advancement" IS NOT NULL
          )
        )
        SELECT
          DISTINCT *
        FROM
          "recursive_cte"
        ORDER BY
          "prices_row_num" ASC
    "#;

    println!("Executing query that should trigger the optimization bug...");

    // First, let's create the logical plan and run optimization explicitly
    let logical_plan = ctx.sql(problematic_query).await?.into_optimized_plan()?;
    println!("Optimized logical plan created successfully!");
    println!("Plan: {}", logical_plan.display_indent());

    match ctx.sql(problematic_query).await {
        Ok(df) => {
            println!("Query executed successfully!");
            df.show().await?;
        }
        Err(e) => {
            println!("Error occurred (this is expected): {}", e);
            println!("Error details: {:?}", e);
        }
    }

    Ok(())
}
