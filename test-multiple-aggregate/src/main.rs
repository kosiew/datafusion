use arrow::array::{Float64Array, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::functions_aggregate::min_max::{max_udaf, min_udaf};
use datafusion::{
    execution::{memory_pool::FairSpillPool, TaskContext},
    physical_expr::aggregate::AggregateExprBuilder,
    physical_plan::{
        aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
        common,
        expressions::col,
        memory::MemoryExec,
        udaf::AggregateFunctionExpr,
        ExecutionPlan,
    },
    prelude::SessionConfig,
};
use datafusion_common::Result;
use std::sync::Arc;

#[tokio::test]
async fn reproduce_spill_schema_error() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::UInt32, false),
        Field::new("b", DataType::Float64, false),
    ]));

    let batches = vec![
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![2, 3, 4, 4])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            ],
        )?,
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![2, 3, 3, 4])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            ],
        )?,
    ];
    let plan: Arc<dyn ExecutionPlan> =
        Arc::new(MemoryExec::try_new(&[batches], schema.clone(), None)?);

    let grouping_set = PhysicalGroupBy::new(
        vec![(col("a", &schema)?, "a".to_string())],
        vec![],
        vec![vec![false]],
    );

    let aggregates: Vec<Arc<AggregateFunctionExpr>> = vec![
        Arc::new(
            AggregateExprBuilder::new(min_udaf(), vec![col("b", &schema)?])
                .schema(schema.clone())
                .alias("MIN(b)")
                .build()?,
        ),
        Arc::new(
            AggregateExprBuilder::new(max_udaf(), vec![col("b", &schema)?])
                .schema(schema.clone())
                .alias("MAX(b)")
                .build()?,
        ),
    ];

    let single_aggregate = Arc::new(AggregateExec::try_new(
        AggregateMode::Single,
        grouping_set,
        aggregates,
        vec![None, None],
        plan,
        schema.clone(),
    )?);

    let batch_size = 2;
    let memory_pool = Arc::new(FairSpillPool::new(1600));
    let task_ctx = Arc::new(
        TaskContext::default()
            .with_session_config(SessionConfig::new().with_batch_size(batch_size))
            .with_runtime(Arc::new(
                RuntimeEnvBuilder::new()
                    .with_memory_pool(memory_pool)
                    .build()?,
            )),
    );

    let _result =
        common::collect(single_aggregate.execute(0, Arc::clone(&task_ctx))?).await?;
    Ok(())
}
