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

#[tokio::test]
async fn test_sum_null_grouping_partitions() {
    use arrow::array::{Int8Array, StringArray, UInt8Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::prelude::*;
    use datafusion_common::ScalarValue;
    use std::sync::Arc;

    // Create test data with some NULL grouping columns
    let utf8_low = StringArray::from(vec![
        Some("group1".to_string()),
        None, // NULL group
        None, // NULL group
        Some("group2".to_string()),
    ]);
    let u8_low = UInt8Array::from(vec![
        Some(1),
        None, // NULL group
        None, // NULL group
        Some(2),
    ]);
    let i8_values = Int8Array::from(vec![
        Some(100),
        Some(50), // Should be summed for NULL group
        Some(75), // Should be summed for NULL group
        Some(25),
    ]);

    let schema = Schema::new(vec![
        Field::new("utf8_low", DataType::Utf8, true),
        Field::new("u8_low", DataType::UInt8, true),
        Field::new("i8", DataType::Int8, true),
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(utf8_low), Arc::new(u8_low), Arc::new(i8_values)],
    )
    .unwrap();

    // Test with single partition (baseline)
    let mut session_config_single = SessionConfig::default();
    session_config_single = session_config_single.set(
        "datafusion.execution.target_partitions",
        &ScalarValue::UInt64(Some(1)),
    );
    let ctx_single = SessionContext::new_with_config(session_config_single);

    let table_single = datafusion::datasource::MemTable::try_new(
        batch.schema(),
        vec![vec![batch.clone()]],
    )
    .unwrap();
    ctx_single
        .register_table("test_table", Arc::new(table_single))
        .unwrap();

    // Test with multiple partitions (task)
    let mut session_config_multi = SessionConfig::default();
    session_config_multi = session_config_multi.set(
        "datafusion.execution.target_partitions",
        &ScalarValue::UInt64(Some(10)),
    );
    let ctx_multi = SessionContext::new_with_config(session_config_multi);

    let table_multi = datafusion::datasource::MemTable::try_new(
        batch.schema(),
        vec![vec![batch.clone()]],
    )
    .unwrap();
    ctx_multi
        .register_table("test_table", Arc::new(table_multi))
        .unwrap();

    let sql = "SELECT utf8_low, u8_low, SUM(i8) as sum_i8 FROM test_table GROUP BY utf8_low, u8_low";

    // Execute on single partition
    let result_single = ctx_single.sql(sql).await.unwrap().collect().await.unwrap();
    println!("Single partition result:");
    println!(
        "{}",
        arrow::util::pretty::pretty_format_batches(&result_single).unwrap()
    );

    // Execute on multiple partitions
    let result_multi = ctx_multi.sql(sql).await.unwrap().collect().await.unwrap();
    println!("Multiple partition result:");
    println!(
        "{}",
        arrow::util::pretty::pretty_format_batches(&result_multi).unwrap()
    );

    // Check if results are equal
    use crate::fuzz_cases::aggregation_fuzzer::check_equality_of_batches;
    let comparison_result = check_equality_of_batches(&result_single, &result_multi);

    match comparison_result {
        Ok(()) => println!("Results match!"),
        Err(e) => {
            println!(
                "Results differ at row {}: {} vs {}",
                e.row_idx, e.lhs_row, e.rhs_row
            );
            panic!("SUM aggregation results differ between single and multi-partition execution");
        }
    }
}
