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
         two AS (SELECT unnest("__unnest_placeholder(mock.metadata).product") FROM one)
    SELECT * FROM two WHERE "__unnest_placeholder(one.__unnest_placeholder(mock.metadata).product).name" == 'Product Name'
    "#;

    let frame = ctx.sql(query).await?;
    frame.show().await?;

    Ok(())
}
