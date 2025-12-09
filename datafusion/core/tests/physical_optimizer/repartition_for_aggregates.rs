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
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::Distribution;

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

    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
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

    let repartition = find_repartition(&final_agg.input)
        .unwrap_or_else(|| panic!("plan lacked repartition:\n{plan_display}"));

    match final_agg.required_input_distribution()[0].clone() {
        Distribution::HashPartitioned(expected_exprs) => match repartition.partitioning() {
            Partitioning::Hash(actual_exprs, _) => {
                assert_eq!(expected_exprs.len(), actual_exprs.len());
                for (expected, actual) in expected_exprs.iter().zip(actual_exprs.iter()) {
                    assert!(expected.eq(actual));
                }
            }
            other => panic!("unexpected partitioning before final aggregate: {:?}", other),
        },
        other => panic!("unexpected distribution requirement: {:?}", other),
    }

    // Ensure the repartition sits between the second Sort -> Aggregate boundary.
    assert!(contains_sorted_input(final_agg.input()));

    Ok(())
}

fn find_repartition(plan: &Arc<dyn ExecutionPlan>) -> Option<&RepartitionExec> {
    if let Some(repartition) = plan.as_any().downcast_ref::<RepartitionExec>() {
        return Some(repartition);
    }

    plan.children()
        .into_iter()
        .find_map(|child| find_repartition(child))
}

fn contains_sorted_input(plan: &Arc<dyn ExecutionPlan>) -> bool {
    if plan.as_any().is::<datafusion_physical_plan::sorts::sort::SortExec>() {
        return true;
    }

    plan.children().iter().any(|child| contains_sorted_input(child))
}
