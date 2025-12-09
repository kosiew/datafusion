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

use datafusion::arrow::array::{Float64Array, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::functions_aggregate::count::count_udaf;
use datafusion::logical_expr::col;
use datafusion::prelude::*;

/// This example demonstrates the fix for multi-partition aggregate repartitioning.
/// Previously, a sequence of Sort -> Aggregate -> Sort -> Aggregate on a multi-partitioned
/// table would panic during the SanityCheckPlan phase due to missing repartition nodes.
/// This example verifies that the optimizer now correctly inserts RepartitionExec nodes
/// to satisfy distribution requirements.
#[tokio::main]
async fn main() {
    let ctx = SessionContext::default();

    let schema = Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Int64, false),
        Field::new("region", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
    ]));

    // create a multi-partitioned MemTable with sample data
    // partition 1: us-west region
    let partition1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1000, 1000, 2000, 2000])),
            Arc::new(StringArray::from(vec![
                "us-west", "us-west", "us-west", "us-west",
            ])),
            Arc::new(Float64Array::from(vec![10.5, 20.3, 15.2, 25.8])),
        ],
    )
    .expect("Failed to create partition 1");

    // partition 2: eu-east region
    let partition2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1000, 1000, 2000])),
            Arc::new(StringArray::from(vec!["eu-east", "eu-east", "eu-east"])),
            Arc::new(Float64Array::from(vec![30.1, 40.2, 35.5])),
        ],
    )
    .expect("Failed to create partition 2");

    let mem_table =
        MemTable::try_new(schema.clone(), vec![vec![partition1], vec![partition2]])
            .expect("Failed to create MemTable");
    ctx.register_table("metrics", Arc::new(mem_table))
        .expect("Failed to register table");

    // aggregate and sort twice - this pattern previously caused a panic
    let data_frame = ctx
        .table("metrics")
        .await
        .expect("Failed to get table")
        .aggregate(
            vec![col("region"), col("ts")],
            vec![count_udaf().call(vec![col("value")])],
        )
        .expect("Failed first aggregate")
        .sort(vec![
            col("region").sort(true, true),
            col("ts").sort(true, true),
        ])
        .expect("Failed first sort")
        .aggregate(
            vec![col("ts")],
            vec![count_udaf().call(vec![col("count(metrics.value)")])],
        )
        .expect("Failed second aggregate")
        .sort(vec![col("ts").sort(true, true)])
        .expect("Failed second sort");

    println!(
        "Logical Plan:\n{}",
        data_frame.logical_plan().display_indent()
    );

    println!("\nExecuting query (should not panic)...");
    data_frame.show().await.expect("Failed to execute query");
    println!("\n✅ Success! The query executed without panicking.");
}
