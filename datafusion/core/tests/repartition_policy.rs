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

use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::*;
use datafusion_common::Result;
use datafusion_physical_plan::displayable;

const CSV_PATH: &str = "datafusion/core/tests/data/aggregate_simple.csv";

async fn optimized_plan(sql: &str, session_config: &SessionConfig) -> Result<String> {
    let ctx = SessionContext::new_with_config(session_config.clone());
    let schema = Schema::new(vec![
        Field::new("c1", DataType::Utf8, true),
        Field::new("c2", DataType::Int64, true),
        Field::new("c3", DataType::Int64, true),
    ]);
    ctx.register_csv(
        "t",
        CSV_PATH,
        CsvReadOptions::new()
            .has_header(true)
            .schema(&schema.into()),
    )
    .await?;

    let df = ctx.sql(sql).await?;
    let plan = df.create_physical_plan().await?;
    Ok(format!("{}", displayable(plan.as_ref()).indent(false)))
}

fn datasource_line(plan: &str) -> Option<&str> {
    plan.lines().find(|line| line.contains("DataSourceExec"))
}

#[tokio::test]
async fn repartition_policy_stable_without_ordering() -> Result<()> {
    let session_config = SessionConfig::new()
        .with_target_partitions(4)
        .with_repartition_file_min_size(1);

    let base_plan = optimized_plan("SELECT c1, c2 FROM t", &session_config).await?;
    let filtered_plan =
        optimized_plan("SELECT c1, c2 FROM t WHERE c2 > 0", &session_config).await?;

    assert_eq!(
        base_plan.contains("RepartitionExec"),
        filtered_plan.contains("RepartitionExec"),
        "Filters should not change repartition wrapping when no ordering is requested"
    );

    assert_eq!(
        datasource_line(&base_plan),
        datasource_line(&filtered_plan),
        "Filter pushdown should not change how the scan is repartitioned"
    );

    Ok(())
}

#[tokio::test]
async fn repartition_policy_stable_with_ordering() -> Result<()> {
    let session_config = SessionConfig::new()
        .with_target_partitions(4)
        .with_repartition_file_min_size(1);

    let ordered_plan =
        optimized_plan("SELECT c1 FROM t ORDER BY c1", &session_config).await?;
    let ordered_filtered_plan =
        optimized_plan("SELECT c1 FROM t WHERE c2 > 0 ORDER BY c1", &session_config)
            .await?;

    assert_eq!(
        ordered_plan.contains("RepartitionExec"),
        ordered_filtered_plan.contains("RepartitionExec"),
        "Filters should not change repartition wrapping when ordering is required"
    );

    assert_eq!(
        datasource_line(&ordered_plan),
        datasource_line(&ordered_filtered_plan),
        "Filter pushdown should not change scan repartitioning when ordering is preserved"
    );

    Ok(())
}
