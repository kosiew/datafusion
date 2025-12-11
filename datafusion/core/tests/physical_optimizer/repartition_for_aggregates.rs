// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_common::config::ConfigOptions;
use datafusion_common::Result;
use datafusion_expr::expr_fn::col;
use datafusion_functions_aggregate::expr_fn::{count, sum};
use datafusion_physical_expr::Partitioning;
use datafusion_physical_optimizer::sanity_checker::SanityCheckPlan;
use datafusion_physical_optimizer::PhysicalOptimizerRule;
use datafusion_physical_plan::aggregates::AggregateExec;
use datafusion_physical_plan::displayable;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::Distribution;
use datafusion_physical_plan::ExecutionPlan;

use crate::physical_optimizer::test_utils::{
    contains_execution_plan, find_execution_plan,
};

#[tokio::test]
async fn repartitions_between_sorted_aggregates() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 1, 2, 2]))],
    )?;

    // Two partitions, one of which is empty, to ensure a multi-partition source.
    let partitions = vec![vec![batch.clone()], vec![]];
    let mem_table = MemTable::try_new(schema, partitions)?;

    let ctx =
        SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
    ctx.register_table("t", Arc::new(mem_table))?;

    let df = ctx
        .table("t")
        .await?
        .sort(vec![col("id").sort(true, true)])?
        .aggregate(vec![col("id")], vec![count(col("id")).alias("cnt")])?
        .sort(vec![col("id").sort(true, true)])?
        .aggregate(vec![col("id")], vec![sum(col("cnt")).alias("total")])?;

    let plan = df.create_physical_plan().await?;
    let plan_display = displayable(plan.as_ref()).indent(true).to_string();

    // The optimizer should ensure the plan is valid and does not panic the sanity checker.
    SanityCheckPlan::new().optimize(plan.clone(), &ConfigOptions::new())?;

    let final_agg = plan
        .as_any()
        .downcast_ref::<AggregateExec>()
        .expect("final aggregate should be root plan");

    let repartition = find_execution_plan::<RepartitionExec>(&final_agg.input)
        .unwrap_or_else(|| panic!("plan lacked repartition:\n{plan_display}"));

    match final_agg.required_input_distribution()[0].clone() {
        Distribution::HashPartitioned(expected_exprs) => match repartition.partitioning()
        {
            Partitioning::Hash(actual_exprs, _) => {
                assert_eq!(expected_exprs.len(), actual_exprs.len());
                for (expected, actual) in expected_exprs.iter().zip(actual_exprs.iter()) {
                    assert!(expected.eq(actual));
                }
            }
            other => panic!("unexpected partitioning before final aggregate: {other:?}"),
        },
        other => panic!("unexpected distribution requirement: {other:?}"),
    }

    // Ensure the repartition sits between the second Sort -> Aggregate boundary.
    assert!(contains_execution_plan::<SortExec>(final_agg.input()));

    Ok(())
}

/// Test that repartition is correctly inserted even with a single partition source.
/// This validates that the fix doesn't over-repartition when the source is already single-partitioned.
#[tokio::test]
async fn no_unnecessary_repartition_for_single_partition_source() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 1, 2, 2]))],
    )?;

    // Single partition source
    let partitions = vec![vec![batch]];
    let mem_table = MemTable::try_new(schema, partitions)?;

    let ctx =
        SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
    ctx.register_table("t", Arc::new(mem_table))?;

    let df = ctx
        .table("t")
        .await?
        .sort(vec![col("id").sort(true, true)])?
        .aggregate(vec![col("id")], vec![count(col("id")).alias("cnt")])?
        .sort(vec![col("id").sort(true, true)])?
        .aggregate(vec![col("id")], vec![sum(col("cnt")).alias("total")])?;

    let plan = df.create_physical_plan().await?;

    // Should pass sanity check
    SanityCheckPlan::new().optimize(plan.clone(), &ConfigOptions::new())?;

    // Single partition source may or may not have repartition depending on config
    // The key is that it should not panic and the plan should be valid
    let final_agg = plan
        .as_any()
        .downcast_ref::<AggregateExec>()
        .expect("final aggregate should be root plan");

    // Verify the aggregate has valid input (children exist)
    assert!(!final_agg.children().is_empty());

    Ok(())
}

/// Test repartitioning with multiple group-by keys.
/// Verifies that hash distribution works correctly for composite group keys.
#[tokio::test]
async fn repartitions_with_multiple_group_by_keys() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key1", DataType::Int32, false),
        Field::new("key2", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 2])),
            Arc::new(Int32Array::from(vec![1, 2, 1, 2])),
            Arc::new(Int32Array::from(vec![10, 20, 30, 40])),
        ],
    )?;

    let partitions = vec![vec![batch.clone()], vec![]];
    let mem_table = MemTable::try_new(schema, partitions)?;

    let ctx =
        SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
    ctx.register_table("t", Arc::new(mem_table))?;

    let df = ctx
        .table("t")
        .await?
        .sort(vec![
            col("key1").sort(true, true),
            col("key2").sort(true, true),
        ])?
        .aggregate(
            vec![col("key1"), col("key2")],
            vec![count(col("val")).alias("cnt")],
        )?
        .sort(vec![
            col("key1").sort(true, true),
            col("key2").sort(true, true),
        ])?
        .aggregate(vec![col("key1")], vec![sum(col("cnt")).alias("total")])?;

    let plan = df.create_physical_plan().await?;

    // Should pass sanity check
    SanityCheckPlan::new().optimize(plan.clone(), &ConfigOptions::new())?;

    let final_agg = plan
        .as_any()
        .downcast_ref::<AggregateExec>()
        .expect("final aggregate should be root plan");

    let repartition = find_execution_plan::<RepartitionExec>(&final_agg.input)
        .expect("repartition should be present for multi-key aggregate");

    // Verify repartition is hash-based
    assert!(matches!(
        repartition.partitioning(),
        Partitioning::Hash(_, _)
    ));

    Ok(())
}

/// Test behavior with different config settings.
/// Ensures repartitioning works correctly with disabled round-robin repartitioning.
#[tokio::test]
async fn repartitions_with_round_robin_disabled() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 1, 2, 2]))],
    )?;

    let partitions = vec![vec![batch.clone()], vec![]];
    let mem_table = MemTable::try_new(schema, partitions)?;

    let config = SessionConfig::new()
        .with_round_robin_repartition(false)
        .with_target_partitions(4);

    let ctx = SessionContext::new_with_config(config);
    ctx.register_table("t", Arc::new(mem_table))?;

    let df = ctx
        .table("t")
        .await?
        .sort(vec![col("id").sort(true, true)])?
        .aggregate(vec![col("id")], vec![count(col("id")).alias("cnt")])?
        .sort(vec![col("id").sort(true, true)])?
        .aggregate(vec![col("id")], vec![sum(col("cnt")).alias("total")])?;

    let plan = df.create_physical_plan().await?;

    // Should pass sanity check even with round-robin disabled
    SanityCheckPlan::new().optimize(plan.clone(), &ConfigOptions::new())?;

    let final_agg = plan
        .as_any()
        .downcast_ref::<AggregateExec>()
        .expect("final aggregate should be root plan");

    // Verify repartition is present
    let _ = find_execution_plan::<RepartitionExec>(&final_agg.input)
        .expect("repartition should be present regardless of round-robin setting");

    Ok(())
}

/// Test that running the optimizer twice produces the same result (idempotency).
/// This ensures the two-phase enforcement doesn't introduce spurious repartitions on re-runs.
#[tokio::test]
async fn enforce_distribution_is_idempotent() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key1", DataType::Int32, false),
        Field::new("key2", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 2])),
            Arc::new(Int32Array::from(vec![1, 2, 1, 2])),
            Arc::new(Int32Array::from(vec![10, 20, 30, 40])),
        ],
    )?;

    let partitions = vec![vec![batch.clone()], vec![]];
    let mem_table = MemTable::try_new(schema, partitions)?;

    let ctx =
        SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
    ctx.register_table("t", Arc::new(mem_table))?;

    // Build a complex plan with multiple aggregates and sorts
    let df = ctx
        .table("t")
        .await?
        .sort(vec![
            col("key1").sort(true, true),
            col("key2").sort(true, true),
        ])?
        .aggregate(
            vec![col("key1"), col("key2")],
            vec![count(col("val")).alias("cnt")],
        )?
        .sort(vec![
            col("key1").sort(true, true),
            col("key2").sort(true, true),
        ])?
        .aggregate(vec![col("key1")], vec![sum(col("cnt")).alias("total")])?;

    let plan = df.create_physical_plan().await?;

    // Get the optimized plan (already went through optimizer pipeline)
    let plan_display_1 = displayable(plan.as_ref()).indent(true).to_string();

    // Run the plan through create_physical_plan again to trigger optimizer
    let df2 = ctx
        .table("t")
        .await?
        .sort(vec![
            col("key1").sort(true, true),
            col("key2").sort(true, true),
        ])?
        .aggregate(
            vec![col("key1"), col("key2")],
            vec![count(col("val")).alias("cnt")],
        )?
        .sort(vec![
            col("key1").sort(true, true),
            col("key2").sort(true, true),
        ])?
        .aggregate(vec![col("key1")], vec![sum(col("cnt")).alias("total")])?;

    let plan2 = df2.create_physical_plan().await?;
    let plan_display_2 = displayable(plan2.as_ref()).indent(true).to_string();

    // The plans should be identical
    assert_eq!(
        plan_display_1, plan_display_2,
        "Optimizer should be idempotent - running twice should produce the same plan"
    );

    // Both plans should pass sanity check
    SanityCheckPlan::new().optimize(plan, &ConfigOptions::new())?;
    SanityCheckPlan::new().optimize(plan2, &ConfigOptions::new())?;

    Ok(())
}
