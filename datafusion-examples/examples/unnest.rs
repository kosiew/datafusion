use datafusion::common::DataFusionError;
use datafusion::prelude::{NdJsonReadOptions, SessionContext};

fn main() -> Result<(), DataFusionError> {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async_main())
}

async fn async_main() -> Result<(), DataFusionError> {
    let ctx = SessionContext::new();
    ctx.register_json(
        "mock",
        "/Users/kosiew/GitHub/datafusion/datafusion-examples/examples/nested.ndjson",
        NdJsonReadOptions::default().file_extension("ndjson"),
    )
    .await?;

    let query = r#"
    WITH one AS (SELECT unnest(metadata) FROM mock),
         two as (SELECT unnest(metadata_fields.product) as product_fields FROM one)
    SELECT *
    FROM two
    WHERE product_fields.name = 'Product Name'     
    "#;

    let frame = ctx.sql(query).await?;
    frame.show().await?;

    Ok(())
}
