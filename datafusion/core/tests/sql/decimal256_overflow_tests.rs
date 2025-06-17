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

//! Unit tests for the Decimal256 overflow issue discovered in fuzz testing

use std::sync::Arc;

use arrow::array::{Array, Decimal256Array, RecordBatch, StringArray, UInt8Array};
use arrow::datatypes::{i256, DataType, Field, Schema};
use datafusion::prelude::*;
use datafusion_common::Result;
use datafusion_expr::Volatility;

/// Test for Decimal256 overflow in sum aggregation
/// This reproduces the fuzz test failure where very large Decimal256 values
/// cause arithmetic overflow during summation.
#[tokio::test]
async fn test_decimal256_sum_overflow() -> Result<()> {
    let ctx = SessionContext::new();

    // Create very large Decimal256 values that will overflow when summed
    // These values are similar to those that caused the fuzz test failure
    let large_value1 = i256::from_string(
        "35065962874285364933282905221533377301606977751949986046567835428737101608988",
    )
    .unwrap();
    let large_value2 = i256::from_string(
        "25929409697036843916674808623678737097170354588534750692949503362074250386052",
    )
    .unwrap();

    // Create a Decimal256Array with precision 76, scale 0 (max precision)
    let decimal_array = Decimal256Array::from(vec![large_value1, large_value2])
        .with_precision_and_scale(76, 0)
        .unwrap();

    let schema = Arc::new(Schema::new(vec![Field::new(
        "decimal256",
        DataType::Decimal256(76, 0),
        false,
    )]));

    let batch = RecordBatch::try_new(schema, vec![Arc::new(decimal_array)])?;

    // Register the table
    let table =
        datafusion::datasource::MemTable::try_new(batch.schema(), vec![vec![batch]])?;
    ctx.register_table("test_table", Arc::new(table))?;

    // Execute the sum query that should trigger overflow
    let result = ctx
        .sql("SELECT sum(decimal256) as total FROM test_table")
        .await;

    // This should result in an arithmetic overflow error
    match result {
        Ok(df) => {
            let execution_result = df.collect().await;
            match execution_result {
                Err(e) => {
                    // Verify we get an ArithmeticOverflow error
                    assert!(
                        e.to_string().contains("ArithmeticOverflow")
                            || e.to_string().contains("Overflow")
                    );
                }
                Ok(_) => {
                    panic!("Expected an overflow error, but query succeeded");
                }
            }
        }
        Err(e) => {
            // Verify we get an ArithmeticOverflow error
            assert!(
                e.to_string().contains("ArithmeticOverflow")
                    || e.to_string().contains("Overflow")
            );
        }
    }

    Ok(())
}

/// Test for Decimal256 overflow in grouped sum aggregation
/// This reproduces the specific case from the fuzz test with grouping
#[tokio::test]
async fn test_decimal256_grouped_sum_overflow() -> Result<()> {
    let ctx = SessionContext::new();

    // Create the same large values that caused overflow
    let large_value1 = i256::from_string(
        "35065962874285364933282905221533377301606977751949986046567835428737101608988",
    )
    .unwrap();
    let large_value2 = i256::from_string(
        "25929409697036843916674808623678737097170354588534750692949503362074250386052",
    )
    .unwrap();

    // Create arrays for a grouped aggregation scenario
    let decimal_array =
        Decimal256Array::from(vec![large_value1, large_value2, large_value1])
            .with_precision_and_scale(76, 0)
            .unwrap();

    let group_array = StringArray::from(vec!["group1", "group1", "group1"]);

    let schema = Arc::new(Schema::new(vec![
        Field::new("group_col", DataType::Utf8, false),
        Field::new("decimal256", DataType::Decimal256(76, 0), false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(group_array), Arc::new(decimal_array)],
    )?;

    // Register the table
    let table =
        datafusion::datasource::MemTable::try_new(batch.schema(), vec![vec![batch]])?;
    ctx.register_table("test_table", Arc::new(table))?;

    // Execute the grouped sum query similar to the fuzz test
    let result = ctx
        .sql("SELECT group_col, sum(decimal256) as total FROM test_table GROUP BY group_col")
        .await;

    // This should result in an arithmetic overflow error
    match result {
        Ok(df) => {
            let execution_result = df.collect().await;
            match execution_result {
                Err(e) => {
                    // Verify we get an ArithmeticOverflow error
                    assert!(
                        e.to_string().contains("ArithmeticOverflow")
                            || e.to_string().contains("Overflow")
                    );
                }
                Ok(_) => {
                    panic!("Expected an overflow error, but query succeeded");
                }
            }
        }
        Err(e) => {
            // Verify we get an ArithmeticOverflow error
            assert!(
                e.to_string().contains("ArithmeticOverflow")
                    || e.to_string().contains("Overflow")
            );
        }
    }

    Ok(())
}

/// Test for Decimal256 sum with DISTINCT that causes overflow
/// This reproduces another pattern from the fuzz test
#[tokio::test]
async fn test_decimal256_sum_distinct_overflow() -> Result<()> {
    let ctx = SessionContext::new();

    // Create large values that will overflow when summed
    let large_value1 = i256::from_string(
        "35065962874285364933282905221533377301606977751949986046567835428737101608988",
    )
    .unwrap();
    let large_value2 = i256::from_string(
        "25929409697036843916674808623678737097170354588534750692949503362074250386052",
    )
    .unwrap();

    // Include duplicates to test DISTINCT behavior
    let decimal_array = Decimal256Array::from(vec![
        large_value1,
        large_value2,
        large_value1,
        large_value2,
    ])
    .with_precision_and_scale(76, 0)
    .unwrap();

    let u8_array = UInt8Array::from(vec![1, 1, 1, 1]);

    let schema = Arc::new(Schema::new(vec![
        Field::new("u8", DataType::UInt8, false),
        Field::new("decimal256", DataType::Decimal256(76, 0), false),
    ]));

    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(u8_array), Arc::new(decimal_array)])?;

    // Register the table
    let table =
        datafusion::datasource::MemTable::try_new(batch.schema(), vec![vec![batch]])?;
    ctx.register_table("test_table", Arc::new(table))?;

    // Execute the sum distinct query similar to the fuzz test
    let result = ctx
        .sql("SELECT u8, sum(DISTINCT decimal256) as total FROM test_table GROUP BY u8")
        .await;

    // This should result in an arithmetic overflow error
    match result {
        Ok(df) => {
            let execution_result = df.collect().await;
            match execution_result {
                Err(e) => {
                    // Verify we get an ArithmeticOverflow error
                    assert!(
                        e.to_string().contains("ArithmeticOverflow")
                            || e.to_string().contains("Overflow")
                    );
                }
                Ok(_) => {
                    panic!("Expected an overflow error, but query succeeded");
                }
            }
        }
        Err(e) => {
            // Verify we get an ArithmeticOverflow error
            assert!(
                e.to_string().contains("ArithmeticOverflow")
                    || e.to_string().contains("Overflow")
            );
        }
    }

    Ok(())
}

/// Test a successful case with smaller Decimal256 values to ensure the aggregation works correctly
#[tokio::test]
async fn test_decimal256_sum_success() -> Result<()> {
    let ctx = SessionContext::new();

    // Create smaller values that won't overflow
    let small_value1 = i256::from(12345);
    let small_value2 = i256::from(67890);
    let expected_sum = i256::from(80235);

    let decimal_array = Decimal256Array::from(vec![small_value1, small_value2])
        .with_precision_and_scale(10, 0)
        .unwrap();

    let schema = Arc::new(Schema::new(vec![Field::new(
        "decimal256",
        DataType::Decimal256(10, 0),
        false,
    )]));

    let batch = RecordBatch::try_new(schema, vec![Arc::new(decimal_array)])?;

    // Register the table
    let table =
        datafusion::datasource::MemTable::try_new(batch.schema(), vec![vec![batch]])?;
    ctx.register_table("test_table", Arc::new(table))?;

    // Execute the sum query that should succeed
    let df = ctx
        .sql("SELECT sum(decimal256) as total FROM test_table")
        .await?;

    let results = df.collect().await?;
    assert_eq!(results.len(), 1);

    let result_batch = &results[0];
    assert_eq!(result_batch.num_rows(), 1);
    assert_eq!(result_batch.num_columns(), 1);

    let sum_array = result_batch
        .column(0)
        .as_any()
        .downcast_ref::<Decimal256Array>()
        .unwrap();

    assert_eq!(sum_array.len(), 1);
    assert_eq!(sum_array.value(0), expected_sum);

    Ok(())
}

/// Test with maximum safe values that shouldn't overflow
#[tokio::test]
async fn test_decimal256_sum_near_max_safe() -> Result<()> {
    let ctx = SessionContext::new();

    // Use values that are large but still safe to sum
    let safe_value1 =
        i256::from_string("100000000000000000000000000000000000000").unwrap();
    let safe_value2 =
        i256::from_string("200000000000000000000000000000000000000").unwrap();
    let expected_sum =
        i256::from_string("300000000000000000000000000000000000000").unwrap();

    let decimal_array = Decimal256Array::from(vec![safe_value1, safe_value2])
        .with_precision_and_scale(50, 0)
        .unwrap();

    let schema = Arc::new(Schema::new(vec![Field::new(
        "decimal256",
        DataType::Decimal256(50, 0),
        false,
    )]));

    let batch = RecordBatch::try_new(schema, vec![Arc::new(decimal_array)])?;

    // Register the table
    let table =
        datafusion::datasource::MemTable::try_new(batch.schema(), vec![vec![batch]])?;
    ctx.register_table("test_table", Arc::new(table))?;

    // Execute the sum query that should succeed
    let df = ctx
        .sql("SELECT sum(decimal256) as total FROM test_table")
        .await?;

    let results = df.collect().await?;
    assert_eq!(results.len(), 1);

    let result_batch = &results[0];
    assert_eq!(result_batch.num_rows(), 1);

    let sum_array = result_batch
        .column(0)
        .as_any()
        .downcast_ref::<Decimal256Array>()
        .unwrap();

    assert_eq!(sum_array.len(), 1);
    assert_eq!(sum_array.value(0), expected_sum);

    Ok(())
}
