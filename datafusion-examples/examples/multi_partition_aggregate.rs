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

use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::MemTable;
use datafusion::functions_aggregate::count::count_udaf;
use datafusion::logical_expr::col;
use datafusion::prelude::*;

#[tokio::main]
async fn main() {
    let ctx = SessionContext::default();

    let schema = Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Int64, false),
        Field::new("region", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
    ]));

    // create an empty but multi-partitioned MemTable
    let mem_table = MemTable::try_new(schema.clone(), vec![vec![], vec![]]).unwrap();
    ctx.register_table("metrics", Arc::new(mem_table)).unwrap();

    // aggregate and sort twice
    let data_frame = ctx
        .table("metrics")
        .await
        .unwrap()
        .aggregate(
            vec![col("region"), col("ts")],
            vec![count_udaf().call(vec![col("value")])],
        )
        .unwrap()
        .sort(vec![
            col("region").sort(true, true),
            col("ts").sort(true, true),
        ])
        .unwrap()
        .aggregate(
            vec![col("ts")],
            vec![count_udaf().call(vec![col("count(metrics.value)")])],
        )
        .unwrap()
        .sort(vec![col("ts").sort(true, true)])
        .unwrap();

    println!(
        "Logical Plan:\n{}",
        data_frame.logical_plan().display_indent()
    );

    data_frame.show().await.unwrap();
}
