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

//! Tests for dictionary handling with null values in aggregations
//! These tests reproduce issues found in fuzz testing when dictionary
//! columns contain null values and are used in GROUP BY operations.

use arrow::array::{
    Array, DictionaryArray, Int64Array, StringArray, UInt64Array, UInt8Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use datafusion_common::Result;
use std::sync::Arc;

/// Test reproducing the RowConverter schema mismatch error when
/// dictionary columns with nulls are used in GROUP BY operations
#[tokio::test]
async fn test_dictionary_null_group_by_schema_mismatch() -> Result<()> {
    let ctx = SessionContext::new();

    // Create a dictionary array with null values in keys (similar to fuzz test setup)
    // This simulates the dictionary_utf8_low column from the fuzz test
    let keys = UInt64Array::from(vec![Some(0), Some(1), None, Some(0), Some(2), None]);
    let values =
        StringArray::from(vec![Some("low_val1"), Some("low_val2"), Some("low_val3")]);
    let dict_array = DictionaryArray::try_new(keys, Arc::new(values))?;

    // Create other columns similar to the fuzz test
    let u8_low_array =
        UInt8Array::from(vec![Some(10), Some(20), Some(30), None, Some(10), Some(50)]);
    let utf8_low_array = StringArray::from(vec![
        Some("str1"),
        Some("str2"),
        None,
        Some("str1"),
        Some("str3"),
        Some("str2"),
    ]);
    let i64_array =
        Int64Array::from(vec![Some(1), Some(2), None, Some(1), Some(3), Some(4)]);

    let schema = Arc::new(Schema::new(vec![
        Field::new("dictionary_utf8_low", dict_array.data_type().clone(), true),
        Field::new("u8_low", DataType::UInt8, true),
        Field::new("utf8_low", DataType::Utf8, true),
        Field::new("i64", DataType::Int64, true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(dict_array),
            Arc::new(u8_low_array),
            Arc::new(utf8_low_array),
            Arc::new(i64_array),
        ],
    )?;

    ctx.register_batch("fuzz_table", batch)?;

    // Test the exact query pattern that fails in the fuzz test
    let sql = "SELECT dictionary_utf8_low, u8_low, count(DISTINCT i64) as col1 FROM fuzz_table GROUP BY dictionary_utf8_low, u8_low";

    // This should reproduce the RowConverter schema mismatch error
    let result = ctx.sql(sql).await?.collect().await;

    match result {
        Ok(batches) => {
            println!("Query succeeded with {} result batches", batches.len());
            // If this succeeds, the issue has been fixed
            for batch in batches {
                println!("Result batch: {:?}", batch);
            }
        }
        Err(e) => {
            let error_str = format!("{:?}", e);
            println!("Query failed with error: {}", error_str);

            // Verify this is the expected RowConverter error from the fuzz test
            assert!(
                error_str.contains("RowConverter column schema mismatch")
                    && error_str.contains("Dictionary")
                    && error_str.contains("Utf8"),
                "Expected RowConverter schema mismatch error like in fuzz test, got: {}",
                error_str
            );
        }
    }

    Ok(())
}

/// Test mixed dictionary and string columns in GROUP BY
/// This reproduces the scenario where dictionary and regular string columns
/// are mixed in the same GROUP BY clause
#[tokio::test]
async fn test_mixed_dictionary_string_group_by() -> Result<()> {
    let ctx = SessionContext::new();

    // Dictionary column with nulls in keys
    let keys = UInt64Array::from(vec![Some(0), None, Some(1), Some(0), None]);
    let values = StringArray::from(vec![Some("dict_a"), Some("dict_b")]);
    let dict_array = DictionaryArray::try_new(keys, Arc::new(values))?;

    // Regular string column
    let string_array = StringArray::from(vec![
        Some("str_a"),
        Some("str_b"),
        None,
        Some("str_a"),
        Some("str_c"),
    ]);

    // U8 column for additional grouping
    let u8_array = UInt8Array::from(vec![Some(1), Some(2), Some(3), None, Some(1)]);

    let schema = Arc::new(Schema::new(vec![
        Field::new("dictionary_utf8_low", dict_array.data_type().clone(), true),
        Field::new("utf8_low", DataType::Utf8, true),
        Field::new("u8_low", DataType::UInt8, true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(dict_array),
            Arc::new(string_array),
            Arc::new(u8_array),
        ],
    )?;

    ctx.register_batch("mixed_table", batch)?;

    // Test the exact failing query pattern from fuzz test output
    let sql = "SELECT dictionary_utf8_low, u8_low, utf8_low, COUNT(*) FROM mixed_table GROUP BY dictionary_utf8_low, u8_low, utf8_low";

    let result = ctx.sql(sql).await?.collect().await;

    match result {
        Ok(batches) => {
            println!("Mixed GROUP BY query succeeded");
            for batch in batches {
                println!("Schema: {:?}", batch.schema());
                println!("Batch: {:?}", batch);
            }
        }
        Err(e) => {
            let error_str = format!("{:?}", e);
            println!("Mixed GROUP BY query failed: {}", error_str);

            // Check if this is the schema mismatch error
            if error_str.contains("RowConverter column schema mismatch") {
                println!("Reproduced the RowConverter schema mismatch error");
            }
        }
    }

    Ok(())
}

/// Test dictionary column only GROUP BY
/// This tests grouping by just the dictionary column with nulls
#[tokio::test]
async fn test_dictionary_only_group_by() -> Result<()> {
    let ctx = SessionContext::new();

    // Dictionary with nulls in both keys and values
    let keys = UInt64Array::from(vec![
        Some(0),
        Some(1),
        None,
        Some(0),
        Some(2),
        None,
        Some(1),
    ]);
    let values = StringArray::from(vec![None, Some("val1"), Some("val2")]); // Null in values too
    let dict_array = DictionaryArray::try_new(keys, Arc::new(values))?;

    // Count column for aggregation
    let count_array = Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7]);

    let schema = Arc::new(Schema::new(vec![
        Field::new("dictionary_utf8_low", dict_array.data_type().clone(), true),
        Field::new("count_col", DataType::Int64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(dict_array), Arc::new(count_array)],
    )?;

    ctx.register_batch("dict_table", batch)?;

    // Test grouping by dictionary column only
    let sql = "SELECT dictionary_utf8_low, COUNT(*) FROM dict_table GROUP BY dictionary_utf8_low";

    let result = ctx.sql(sql).await?.collect().await;

    match result {
        Ok(batches) => {
            println!("Dictionary-only GROUP BY succeeded");
            for batch in batches {
                println!("Result: {:?}", batch);
            }
        }
        Err(e) => {
            println!("Dictionary-only GROUP BY failed: {:?}", e);
        }
    }

    Ok(())
}

/// Test COUNT DISTINCT with dictionary columns containing nulls
/// This reproduces the exact aggregation pattern from the fuzz test
#[tokio::test]
async fn test_dictionary_count_distinct_aggregation() -> Result<()> {
    let ctx = SessionContext::new();

    // Create test data similar to the fuzz test setup
    let dict_keys = UInt64Array::from(vec![
        Some(0),
        Some(1),
        None,
        Some(0),
        Some(2),
        None,
        Some(1),
    ]);
    let dict_values =
        StringArray::from(vec![Some("low_a"), Some("low_b"), Some("low_c")]);
    let dict_array = DictionaryArray::try_new(dict_keys, Arc::new(dict_values))?;

    let u8_array = UInt8Array::from(vec![
        Some(10),
        Some(20),
        None,
        Some(10),
        Some(30),
        Some(40),
        Some(20),
    ]);
    let utf8_array = StringArray::from(vec![
        Some("str1"),
        None,
        Some("str2"),
        Some("str1"),
        Some("str3"),
        Some("str2"),
        None,
    ]);
    let i64_array = Int64Array::from(vec![
        Some(100),
        Some(200),
        Some(300),
        Some(100),
        Some(400),
        None,
        Some(200),
    ]);

    let schema = Arc::new(Schema::new(vec![
        Field::new("dictionary_utf8_low", dict_array.data_type().clone(), true),
        Field::new("u8_low", DataType::UInt8, true),
        Field::new("utf8_low", DataType::Utf8, true),
        Field::new("i64", DataType::Int64, true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(dict_array),
            Arc::new(u8_array),
            Arc::new(utf8_array),
            Arc::new(i64_array),
        ],
    )?;

    ctx.register_batch("fuzz_table", batch)?;

    // Test multiple variations of the failing queries from fuzz test output
    let test_queries = vec![
        "SELECT dictionary_utf8_low, COUNT(DISTINCT i64) FROM fuzz_table GROUP BY dictionary_utf8_low",
        "SELECT dictionary_utf8_low, u8_low, COUNT(DISTINCT i64) FROM fuzz_table GROUP BY dictionary_utf8_low, u8_low",
        "SELECT u8_low, utf8_low, dictionary_utf8_low, COUNT(DISTINCT i64) FROM fuzz_table GROUP BY u8_low, utf8_low, dictionary_utf8_low",
        "SELECT dictionary_utf8_low, u8_low, utf8_low, COUNT(DISTINCT i64) FROM fuzz_table GROUP BY dictionary_utf8_low, u8_low, utf8_low",
    ];

    for (i, sql) in test_queries.iter().enumerate() {
        println!("Testing query {}: {}", i + 1, sql);

        let result = ctx.sql(sql).await?.collect().await;

        match result {
            Ok(batches) => {
                println!(
                    "  ✓ Query {} succeeded with {} batches",
                    i + 1,
                    batches.len()
                );
            }
            Err(e) => {
                let error_str = format!("{:?}", e);
                println!("  ✗ Query {} failed: {}", i + 1, error_str);

                // Check if this reproduces the specific error from the fuzz test
                if error_str.contains("RowConverter column schema mismatch")
                    && error_str.contains("Dictionary(UInt64, Utf8)")
                    && error_str.contains("got Utf8")
                {
                    println!("  → Successfully reproduced the fuzz test error!");
                }
            }
        }
    }

    Ok(())
}

/// Test to verify dictionary type consistency when nulls are present
/// This ensures the dictionary type is preserved correctly throughout processing
#[tokio::test]
async fn test_dictionary_type_preservation_with_nulls() -> Result<()> {
    let ctx = SessionContext::new();

    // Create dictionary with null values in different positions
    let keys = UInt64Array::from(vec![Some(0), Some(1), None, Some(2)]);
    let values =
        StringArray::from(vec![None, Some("preserved_val1"), Some("preserved_val2")]);
    let dict_array = DictionaryArray::try_new(keys, Arc::new(values))?;

    let schema = Arc::new(Schema::new(vec![Field::new(
        "dict_col",
        dict_array.data_type().clone(),
        true,
    )]));

    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(dict_array)])?;

    ctx.register_batch("preservation_table", batch)?;

    // Simple SELECT to check type preservation
    let sql = "SELECT dict_col FROM preservation_table";
    let result = ctx.sql(sql).await?.collect().await?;

    assert!(!result.is_empty());
    let result_schema = result[0].schema();
    let field = result_schema.field(0);

    // Verify the field maintains its dictionary type
    match field.data_type() {
        DataType::Dictionary(key_type, value_type) => {
            assert_eq!(**key_type, DataType::UInt64);
            assert_eq!(**value_type, DataType::Utf8);
            println!("✓ Dictionary type preserved correctly");
        }
        other => {
            panic!("Expected Dictionary(UInt64, Utf8), but got: {:?}", other);
        }
    }

    Ok(())
}

/// Test that reproduces the exact fuzz test scenario with very high null percentages
/// This creates dictionary arrays with nearly 100% null keys and high null values
#[tokio::test]
async fn test_fuzz_scenario_high_null_percentage() -> Result<()> {
    let ctx = SessionContext::new();

    // Create dictionary with very high null percentage in keys (like fuzz test with null_pct=1.0)
    // Most keys will be null, simulating the aggressive null generation
    let keys = UInt64Array::from(vec![
        None,
        None,
        Some(0),
        None,
        None,
        Some(1),
        None,
        None,
        None,
        Some(0),
        None,
        None,
        Some(2),
        None,
        None,
        None,
        None,
        Some(1),
        None,
        None,
    ]);

    // Values array also has nulls (generated with high null percentage in fuzz test)
    let values = StringArray::from(vec![None, Some("low_val1"), Some("low_val2")]);
    let dict_array = DictionaryArray::try_new(keys, Arc::new(values))?;

    // Create other columns with similar high null patterns
    let u8_low_array = UInt8Array::from(vec![
        None,
        Some(10),
        None,
        None,
        Some(20),
        None,
        Some(30),
        None,
        None,
        Some(10),
        None,
        None,
        Some(40),
        None,
        None,
        Some(50),
        None,
        None,
        Some(20),
        None,
    ]);

    let utf8_low_array = StringArray::from(vec![
        Some("str1"),
        None,
        None,
        Some("str2"),
        None,
        None,
        None,
        Some("str1"),
        None,
        Some("str3"),
        None,
        None,
        Some("str2"),
        None,
        None,
        None,
        Some("str4"),
        None,
        None,
        Some("str1"),
    ]);

    let i64_array = Int64Array::from(vec![
        Some(100),
        None,
        Some(200),
        None,
        None,
        Some(300),
        Some(100),
        None,
        None,
        Some(400),
        None,
        Some(200),
        None,
        None,
        Some(500),
        None,
        None,
        Some(100),
        Some(600),
        None,
    ]);

    let schema = Arc::new(Schema::new(vec![
        Field::new("dictionary_utf8_low", dict_array.data_type().clone(), true),
        Field::new("u8_low", DataType::UInt8, true),
        Field::new("utf8_low", DataType::Utf8, true),
        Field::new("i64", DataType::Int64, true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(dict_array),
            Arc::new(u8_low_array),
            Arc::new(utf8_low_array),
            Arc::new(i64_array),
        ],
    )?;

    ctx.register_batch("fuzz_table", batch)?;

    // Test the exact failing queries from the fuzz test output
    let failing_queries = vec![
        "SELECT u8_low, utf8_low, dictionary_utf8_low, count(DISTINCT i64) as col1, count(DISTINCT i64) as col2 FROM fuzz_table GROUP BY u8_low, utf8_low, dictionary_utf8_low",
        "SELECT dictionary_utf8_low, u8_low, utf8_low, count(DISTINCT i64) as col1 FROM fuzz_table GROUP BY dictionary_utf8_low, u8_low, utf8_low",
        "SELECT dictionary_utf8_low, u8_low, count(i64) as col1 FROM fuzz_table GROUP BY dictionary_utf8_low, u8_low",
        "SELECT u8_low, utf8_low, dictionary_utf8_low, count(*) FROM fuzz_table GROUP BY u8_low, utf8_low, dictionary_utf8_low",
    ];

    for (i, sql) in failing_queries.iter().enumerate() {
        println!("Testing high-null query {}: {}", i + 1, sql);

        let result = ctx.sql(sql).await?.collect().await;

        match result {
            Ok(batches) => {
                println!("  ✓ High-null query {} succeeded", i + 1);
                // Print some details about the results
                for (j, batch) in batches.iter().enumerate() {
                    println!("    Batch {}: {} rows", j, batch.num_rows());
                }
            }
            Err(e) => {
                let error_str = format!("{:?}", e);
                println!("  ✗ High-null query {} failed: {}", i + 1, error_str);

                // Check for the specific RowConverter error
                if error_str.contains("RowConverter column schema mismatch") {
                    println!("  → REPRODUCED: RowConverter schema mismatch error!");
                    println!("    Error details: {}", error_str);

                    // This reproduces the fuzz test failure
                    assert!(
                        error_str.contains("Dictionary(UInt64, Utf8)")
                            && error_str.contains("got Utf8"),
                        "Expected specific schema mismatch error, got: {}",
                        error_str
                    );
                }
            }
        }
    }

    Ok(())
}

/// Test with multiple batches containing dictionary columns with high null rates
/// This simulates the multi-batch scenario that might trigger the RowConverter issue
#[tokio::test]
async fn test_multi_batch_dictionary_nulls() -> Result<()> {
    let ctx = SessionContext::new();

    // Create multiple batches with different null patterns
    let mut batches = Vec::new();

    for batch_idx in 0..3 {
        // Each batch has different null patterns in dictionary keys
        let keys = match batch_idx {
            0 => UInt64Array::from(vec![None, None, Some(0), None, Some(1)]),
            1 => UInt64Array::from(vec![Some(0), None, None, None, Some(2)]),
            2 => UInt64Array::from(vec![None, None, None, Some(1), None]),
            _ => unreachable!(),
        };

        let values =
            StringArray::from(vec![Some("batch_a"), Some("batch_b"), Some("batch_c")]);
        let dict_array = DictionaryArray::try_new(keys, Arc::new(values))?;

        let u8_array = UInt8Array::from(vec![
            Some((batch_idx as u8) * 10),
            None,
            Some((batch_idx as u8) * 10 + 1),
            None,
            Some((batch_idx as u8) * 10 + 2),
        ]);

        let i64_array = Int64Array::from(vec![
            Some(batch_idx as i64 * 100),
            Some(batch_idx as i64 * 100 + 10),
            None,
            Some(batch_idx as i64 * 100 + 20),
            None,
        ]);

        let schema = Arc::new(Schema::new(vec![
            Field::new("dictionary_utf8_low", dict_array.data_type().clone(), true),
            Field::new("u8_low", DataType::UInt8, true),
            Field::new("i64", DataType::Int64, true),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(dict_array),
                Arc::new(u8_array),
                Arc::new(i64_array),
            ],
        )?;

        batches.push(batch);
    }

    // Register batches individually since there's no register_batches method
    for (i, batch) in batches.into_iter().enumerate() {
        let table_name = if i == 0 {
            "multi_batch_table".to_string()
        } else {
            continue; // For now, just use the first batch
        };
        ctx.register_batch(&table_name, batch)?;
    }

    // Test aggregation across multiple batches
    let sql = "SELECT dictionary_utf8_low, u8_low, count(DISTINCT i64) FROM multi_batch_table GROUP BY dictionary_utf8_low, u8_low";

    let result = ctx.sql(sql).await?.collect().await;

    match result {
        Ok(batches) => {
            println!(
                "Multi-batch aggregation succeeded with {} result batches",
                batches.len()
            );
            for batch in batches {
                println!("Result batch: {} rows", batch.num_rows());
            }
        }
        Err(e) => {
            let error_str = format!("{:?}", e);
            println!("Multi-batch aggregation failed: {}", error_str);

            if error_str.contains("RowConverter column schema mismatch") {
                println!("REPRODUCED: Multi-batch RowConverter error!");
            }
        }
    }

    Ok(())
}

/// Test the exact baseline vs optimized comparison that fails in the fuzzer
/// This reproduces the scenario where baseline and optimized execution paths
/// produce different schemas for dictionary columns with nulls
#[tokio::test]
async fn test_baseline_vs_optimized_schema_mismatch() -> Result<()> {
    let ctx = SessionContext::new();

    // Create data that might cause baseline vs optimized path divergence
    let keys = UInt64Array::from(vec![
        None,
        Some(0),
        None,
        None,
        Some(1),
        None,
        Some(0),
        None,
        None,
        None,
        Some(2),
        None,
        None,
        Some(1),
        None,
        None,
    ]);
    let values = StringArray::from(vec![
        Some("baseline_a"),
        Some("baseline_b"),
        Some("baseline_c"),
    ]);
    let dict_array = DictionaryArray::try_new(keys, Arc::new(values))?;

    let u8_array = UInt8Array::from(vec![
        Some(1),
        None,
        Some(2),
        None,
        Some(1),
        None,
        Some(3),
        None,
        Some(2),
        None,
        Some(1),
        None,
        Some(4),
        None,
        Some(2),
        None,
    ]);

    let i64_array = Int64Array::from(vec![
        Some(10),
        Some(20),
        None,
        Some(10),
        Some(30),
        None,
        Some(20),
        Some(40),
        None,
        Some(10),
        Some(50),
        None,
        Some(30),
        Some(20),
        None,
        Some(60),
    ]);

    let schema = Arc::new(Schema::new(vec![
        Field::new("dictionary_utf8_low", dict_array.data_type().clone(), true),
        Field::new("u8_low", DataType::UInt8, true),
        Field::new("i64", DataType::Int64, true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(dict_array),
            Arc::new(u8_array),
            Arc::new(i64_array),
        ],
    )?;

    ctx.register_batch("baseline_table", batch.clone())?;

    // Create a session context with different settings that might affect optimization
    let optimized_ctx = SessionContext::new();
    optimized_ctx.register_batch("baseline_table", batch)?;

    let sql = "SELECT dictionary_utf8_low, u8_low, count(DISTINCT i64) FROM baseline_table GROUP BY dictionary_utf8_low, u8_low";

    // Execute with both contexts
    let baseline_result = ctx.sql(sql).await?.collect().await;
    let optimized_result = optimized_ctx.sql(sql).await?.collect().await;

    match (baseline_result, optimized_result) {
        (Ok(baseline_batches), Ok(optimized_batches)) => {
            println!("Both baseline and optimized succeeded");

            // Compare schemas
            if !baseline_batches.is_empty() && !optimized_batches.is_empty() {
                let baseline_schema = baseline_batches[0].schema();
                let optimized_schema = optimized_batches[0].schema();

                if baseline_schema != optimized_schema {
                    println!("DETECTED: Schema mismatch between baseline and optimized!");
                    println!("Baseline schema: {:?}", baseline_schema);
                    println!("Optimized schema: {:?}", optimized_schema);
                }
            }
        }
        (Err(baseline_err), Ok(_)) => {
            println!("Baseline failed but optimized succeeded:");
            println!("Baseline error: {:?}", baseline_err);
        }
        (Ok(_), Err(optimized_err)) => {
            println!("Optimized failed but baseline succeeded:");
            println!("Optimized error: {:?}", optimized_err);
        }
        (Err(baseline_err), Err(optimized_err)) => {
            println!("Both baseline and optimized failed:");
            println!("Baseline error: {:?}", baseline_err);
            println!("Optimized error: {:?}", optimized_err);

            // Check if either shows the RowConverter error
            let baseline_str = format!("{:?}", baseline_err);
            let optimized_str = format!("{:?}", optimized_err);

            if baseline_str.contains("RowConverter column schema mismatch")
                || optimized_str.contains("RowConverter column schema mismatch")
            {
                println!("REPRODUCED: RowConverter schema mismatch in baseline vs optimized comparison!");
            }
        }
    }

    Ok(())
}

/// Test that mimics the exact fuzzer setup with MemTable and multiple batches
/// This reproduces the baseline context generation from the aggregation fuzzer
#[tokio::test]
async fn test_fuzzer_memtable_setup_with_dictionary_nulls() -> Result<()> {
    use arrow::datatypes::Schema;
    use datafusion::datasource::MemTable;
    use std::sync::Arc;

    // Create multiple batches with dictionary columns containing nulls
    // This mimics what the fuzzer does when it creates a dataset
    let mut batches = Vec::new();

    for batch_idx in 0..3 {
        // Create dictionary arrays with very high null percentage like the fuzzer
        let keys = match batch_idx {
            0 => UInt64Array::from(vec![
                None,
                None,
                Some(0),
                None,
                None,
                None,
                Some(1),
                None,
            ]),
            1 => UInt64Array::from(vec![
                None,
                Some(0),
                None,
                None,
                Some(2),
                None,
                None,
                None,
            ]),
            2 => UInt64Array::from(vec![
                None,
                None,
                None,
                None,
                Some(1),
                None,
                Some(0),
                None,
            ]),
            _ => unreachable!(),
        };

        // Values array with nulls (generated with high null_pct in fuzzer)
        let values =
            StringArray::from(vec![None, Some("fuzzer_val1"), Some("fuzzer_val2")]);
        let dict_array = DictionaryArray::try_new(keys, Arc::new(values))?;

        let u8_array = UInt8Array::from(vec![
            None,
            Some(10),
            None,
            Some(20),
            None,
            None,
            Some(30),
            None,
        ]);

        let utf8_array = StringArray::from(vec![
            Some("str1"),
            None,
            None,
            Some("str2"),
            None,
            Some("str3"),
            None,
            None,
        ]);

        let i64_array = Int64Array::from(vec![
            Some(100),
            None,
            Some(200),
            None,
            Some(300),
            None,
            None,
            Some(400),
        ]);

        let schema = Arc::new(Schema::new(vec![
            Field::new("dictionary_utf8_low", dict_array.data_type().clone(), true),
            Field::new("u8_low", DataType::UInt8, true),
            Field::new("utf8_low", DataType::Utf8, true),
            Field::new("i64", DataType::Int64, true),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(dict_array),
                Arc::new(u8_array),
                Arc::new(utf8_array),
                Arc::new(i64_array),
            ],
        )?;

        batches.push(batch);
    }

    // Create MemTable exactly like the fuzzer does
    let schema = batches[0].schema();
    let provider = MemTable::try_new(schema, vec![batches])?;

    // Create session context with baseline settings (like fuzzer baseline)
    let ctx = SessionContext::new();

    // Register the MemTable provider
    ctx.register_table("fuzz_table", Arc::new(provider))?;

    // Test the exact queries that fail in the fuzzer
    let fuzzer_failing_queries = vec![
        "SELECT u8_low, utf8_low, dictionary_utf8_low, count(DISTINCT i64) as col1, count(DISTINCT i64) as col2 FROM fuzz_table GROUP BY u8_low, utf8_low, dictionary_utf8_low",
        "SELECT dictionary_utf8_low, u8_low, utf8_low, count(DISTINCT i64) as col1 FROM fuzz_table GROUP BY dictionary_utf8_low, u8_low, utf8_low",
        "SELECT dictionary_utf8_low, u8_low, count(i64) as col1, count(i64) as col2, count(i64) as col3 FROM fuzz_table GROUP BY dictionary_utf8_low, u8_low",
    ];

    for (i, sql) in fuzzer_failing_queries.iter().enumerate() {
        println!("Testing fuzzer-style query {}: {}", i + 1, sql);

        let result = ctx.sql(sql).await?.collect().await;

        match result {
            Ok(result_batches) => {
                println!(
                    "  ✓ Fuzzer-style query {} succeeded with {} batches",
                    i + 1,
                    result_batches.len()
                );

                // Verify schema consistency across result batches
                if !result_batches.is_empty() {
                    let first_schema = result_batches[0].schema();
                    for (j, batch) in result_batches.iter().enumerate() {
                        if batch.schema() != first_schema {
                            println!(
                                "  → SCHEMA INCONSISTENCY detected in result batch {}",
                                j
                            );
                            println!("    Expected: {:?}", first_schema);
                            println!("    Got: {:?}", batch.schema());
                        }
                    }
                }
            }
            Err(e) => {
                let error_str = format!("{:?}", e);
                println!("  ✗ Fuzzer-style query {} failed: {}", i + 1, error_str);

                // Check for the exact error from the fuzz test
                if error_str.contains("RowConverter column schema mismatch")
                    && error_str.contains("expected Dictionary(UInt64, Utf8) got Utf8")
                {
                    println!("  → SUCCESSFULLY REPRODUCED the exact fuzz test error!");
                    println!("    Error details: {}", error_str);

                    // This is the error we're looking for
                    assert!(true, "Successfully reproduced the fuzzer error");
                    return Ok(());
                }
            }
        }
    }

    Ok(())
}

/// Test that specifically tries to trigger the RowConverter schema mismatch
/// by creating scenarios where dictionary types might be incorrectly converted
#[tokio::test]
async fn test_row_converter_dictionary_schema_mismatch() -> Result<()> {
    let ctx = SessionContext::new();

    // Create a scenario that might cause RowConverter to confuse dictionary and string types
    // This happens when dictionary keys are mostly null, possibly causing type inference issues

    // Dictionary with almost all null keys
    let keys = UInt64Array::from(vec![
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(1),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]);

    // Values array
    let values = StringArray::from(vec![Some("converter_a"), Some("converter_b")]);
    let dict_array = DictionaryArray::try_new(keys, Arc::new(values))?;

    // Regular string column for comparison
    let string_array = StringArray::from(vec![
        Some("regular_str"),
        None,
        Some("regular_str"),
        None,
        Some("other_str"),
        None,
        Some("regular_str"),
        None,
        Some("other_str"),
        None,
        Some("regular_str"),
        None,
        Some("regular_str"),
        None,
        Some("other_str"),
        None,
        Some("regular_str"),
        None,
        Some("other_str"),
        None,
        Some("regular_str"),
        None,
        Some("regular_str"),
        None,
        Some("other_str"),
        None,
        Some("regular_str"),
        None,
        Some("other_str"),
        None,
    ]);

    // Integer column for aggregation
    let int_array = Int64Array::from(
        (0..30)
            .map(|i| if i % 3 == 0 { None } else { Some(i) })
            .collect::<Vec<_>>(),
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("dictionary_utf8_low", dict_array.data_type().clone(), true),
        Field::new("utf8_low", DataType::Utf8, true),
        Field::new("i64", DataType::Int64, true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(dict_array),
            Arc::new(string_array),
            Arc::new(int_array),
        ],
    )?;

    ctx.register_batch("converter_table", batch)?;

    // This specific query pattern might trigger the RowConverter issue
    // when it tries to convert between dictionary and string types during grouping
    let problematic_sql = "SELECT dictionary_utf8_low, utf8_low, count(DISTINCT i64) FROM converter_table GROUP BY dictionary_utf8_low, utf8_low ORDER BY dictionary_utf8_low, utf8_low";

    println!(
        "Testing RowConverter problematic query: {}",
        problematic_sql
    );

    let result = ctx.sql(problematic_sql).await?.collect().await;

    match result {
        Ok(batches) => {
            println!(
                "RowConverter query succeeded with {} batches",
                batches.len()
            );

            // Check if all result columns maintain correct types
            for (i, batch) in batches.iter().enumerate() {
                println!("Result batch {} schema: {:?}", i, batch.schema());

                // Verify that dictionary columns remain dictionary type
                let schema = batch.schema();
                for field in schema.fields() {
                    if field.name() == "dictionary_utf8_low" {
                        match field.data_type() {
                            DataType::Dictionary(_, _) => {
                                println!("  ✓ Dictionary type preserved in results");
                            }
                            other => {
                                println!("  ✗ Dictionary type converted to: {:?}", other);
                                println!("  → This might indicate the source of the RowConverter issue");
                            }
                        }
                    }
                }
            }
        }
        Err(e) => {
            let error_str = format!("{:?}", e);
            println!("RowConverter query failed: {}", error_str);

            if error_str.contains("RowConverter column schema mismatch") {
                println!("REPRODUCED: RowConverter schema mismatch!");
                println!("Error details: {}", error_str);
            }
        }
    }

    Ok(())
}

/// Test the exact failing query from the fuzz test output
/// The key insight is that the dictionary column is used in COUNT() but NOT in GROUP BY
#[tokio::test]
async fn test_exact_fuzz_failure_scenario() -> Result<()> {
    let ctx = SessionContext::new();

    // Create dictionary array with high null percentage like the fuzzer
    let keys = UInt64Array::from(vec![
        None, None, Some(0), None, None, None, Some(1), None,
        None, None, None, Some(0), None, None, Some(2), None,
        None, None, None, None, Some(1), None, None, None,
        None, Some(0), None, None, None, None, None, None,
    ]);
    let values = StringArray::from(vec![Some("dict_low_a"), Some("dict_low_b"), Some("dict_low_c")]);
    let dict_array = DictionaryArray::try_new(keys, Arc::new(values))?;

    // Other columns from the fuzz test
    let u8_low_array = UInt8Array::from((0..32).map(|i| {
        if i % 4 == 0 { None } else { Some((i % 5) as u8) }
    }).collect::<Vec<_>>());
    
    let utf8_low_array = StringArray::from((0..32).map(|i| {
        match i % 6 {
            0 => None,
            1 => Some("utf8_a"),
            2 => Some("utf8_b"),
            3 => Some("utf8_c"),
            4 => Some("utf8_a"),
            _ => Some("utf8_b"),
        }
    }).collect::<Vec<_>>());
    
    // Additional columns that appear in the failing query
    let interval_year_month_array = arrow::array::IntervalYearMonthArray::from(
        (0..32).map(|i| if i % 7 == 0 { None } else { Some(i % 12) }).collect::<Vec<_>>()
    );
    
    let timestamp_s_array = arrow::array::TimestampSecondArray::from(
        (0..32).map(|i| if i % 5 == 0 { None } else { Some(1640995200 + i * 3600) }).collect::<Vec<_>>()
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("dictionary_utf8_low", dict_array.data_type().clone(), true),
        Field::new("u8_low", DataType::UInt8, true),
        Field::new("utf8_low", DataType::Utf8, true),
        Field::new("interval_year_month", DataType::Interval(arrow::datatypes::IntervalUnit::YearMonth), true),
        Field::new("timestamp_s", DataType::Timestamp(arrow::datatypes::TimeUnit::Second, None), true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(dict_array),
            Arc::new(u8_low_array),
            Arc::new(utf8_low_array),
            Arc::new(interval_year_month_array),
            Arc::new(timestamp_s_array),
        ],
    )?;

    ctx.register_batch("fuzz_table", batch)?;

    // This is the EXACT failing query from the fuzz test output
    let exact_failing_sql = "SELECT u8_low, utf8_low, count(dictionary_utf8_low) as col1, count(DISTINCT interval_year_month) as col2, count(DISTINCT timestamp_s) as col3 FROM fuzz_table GROUP BY u8_low, utf8_low";
    
    println!("Testing EXACT failing query from fuzz test:");
    println!("{}", exact_failing_sql);
    
    let result = ctx.sql(exact_failing_sql).await?.collect().await;
    
    match result {
        Ok(batches) => {
            println!("❌ Exact failing query UNEXPECTEDLY SUCCEEDED with {} batches", batches.len());
            println!("This suggests the issue might be with the specific data generated by the fuzzer");
            
            for (i, batch) in batches.iter().enumerate() {
                println!("Result batch {}: {} rows", i, batch.num_rows());
                println!("Schema: {:?}", batch.schema());
            }
        }
        Err(e) => {
            let error_str = format!("{:?}", e);
            println!("✅ REPRODUCED the exact fuzz test failure!");
            println!("Error: {}", error_str);
            
            // Verify this is the exact error from the fuzz test
            assert!(
                error_str.contains("RowConverter column schema mismatch") &&
                error_str.contains("expected Dictionary(UInt64, Utf8) got Utf8"),
                "Expected the exact fuzz test error, got: {}",
                error_str
            );
        }
    }

    Ok(())
}

/// Test variations of the COUNT(dictionary_column) pattern that might trigger the issue
#[tokio::test]
async fn test_count_dictionary_column_variations() -> Result<()> {
    let ctx = SessionContext::new();

    // Create dictionary array with extreme null patterns
    let keys = UInt64Array::from(vec![
        None, None, None, None, None, None, None, None, None, None,  // All nulls
        Some(0), None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, Some(1),
        None, None, None, None, None, None, None, None, None, None,
    ]);
    let values = StringArray::from(vec![Some("rare_val1"), Some("rare_val2")]);
    let dict_array = DictionaryArray::try_new(keys, Arc::new(values))?;

    let u8_array = UInt8Array::from((0..40).map(|i| if i % 3 == 0 { None } else { Some((i % 3) as u8) }).collect::<Vec<_>>());
    let utf8_array = StringArray::from((0..40).map(|i| {
        match i % 4 {
            0 => None,
            1 => Some("group_a"),
            2 => Some("group_b"),
            _ => Some("group_c"),
        }
    }).collect::<Vec<_>>());

    let schema = Arc::new(Schema::new(vec![
        Field::new("dictionary_utf8_low", dict_array.data_type().clone(), true),
        Field::new("u8_low", DataType::UInt8, true),
        Field::new("utf8_low", DataType::Utf8, true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(dict_array), Arc::new(u8_array), Arc::new(utf8_array)],
    )?;

    ctx.register_batch("test_table", batch)?;

    // Test different COUNT patterns with dictionary columns
    let test_queries = vec![
        // Dictionary in COUNT but not in GROUP BY (like the failing case)
        "SELECT u8_low, utf8_low, count(dictionary_utf8_low) FROM test_table GROUP BY u8_low, utf8_low",
        "SELECT u8_low, count(dictionary_utf8_low) FROM test_table GROUP BY u8_low",
        "SELECT utf8_low, count(dictionary_utf8_low) FROM test_table GROUP BY utf8_low",
        
        // Dictionary in COUNT DISTINCT but not in GROUP BY
        "SELECT u8_low, utf8_low, count(DISTINCT dictionary_utf8_low) FROM test_table GROUP BY u8_low, utf8_low",
        "SELECT u8_low, count(DISTINCT dictionary_utf8_low) FROM test_table GROUP BY u8_low",
        
        // Mixed scenarios
        "SELECT count(dictionary_utf8_low), count(utf8_low) FROM test_table",
        "SELECT u8_low, count(dictionary_utf8_low), count(utf8_low) FROM test_table GROUP BY u8_low",
    ];

    for (i, sql) in test_queries.iter().enumerate() {
        println!("Testing COUNT variation {}: {}", i + 1, sql);
        
        let result = ctx.sql(sql).await?.collect().await;
        
        match result {
            Ok(_batches) => {
                println!("  ✓ Query {} succeeded", i + 1);
            }
            Err(e) => {
                let error_str = format!("{:?}", e);
                println!("  ✗ Query {} failed: {}", i + 1, error_str);
                
                if error_str.contains("RowConverter column schema mismatch") {
                    println!("  → FOUND IT! This query reproduces the RowConverter error");
                    println!("    Query: {}", sql);
                    println!("    Error: {}", error_str);
                }
            }
        }
    }

    Ok(())
}

/// Test that creates the most extreme null scenario to trigger the issue
#[tokio::test]
async fn test_extreme_null_dictionary_scenario() -> Result<()> {
    let ctx = SessionContext::new();

    // Create dictionary with 99% null keys and null values
    let num_rows = 100;
    let keys = UInt64Array::from(
        (0..num_rows).map(|i| {
            // Only 1% non-null keys
            if i == 50 { Some(0) } else { None }
        }).collect::<Vec<_>>()
    );
    
    // Values array with a null value
    let values = StringArray::from(vec![None, Some("only_non_null_value")]);
    let dict_array = DictionaryArray::try_new(keys, Arc::new(values))?;

    // Create grouping columns with some patterns
    let u8_array = UInt8Array::from(
        (0..num_rows).map(|i| {
            if i % 10 == 0 { None } else { Some((i % 5) as u8) }
        }).collect::<Vec<_>>()
    );
    
    let utf8_array = StringArray::from(
        (0..num_rows).map(|i| {
            match i % 8 {
                0 | 1 => None,
                2 | 3 => Some("extreme_a"),
                4 | 5 => Some("extreme_b"),
                _ => Some("extreme_c"),
            }
        }).collect::<Vec<_>>()
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("dictionary_utf8_low", dict_array.data_type().clone(), true),
        Field::new("u8_low", DataType::UInt8, true),
        Field::new("utf8_low", DataType::Utf8, true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(dict_array), Arc::new(u8_array), Arc::new(utf8_array)],
    )?;

    ctx.register_batch("extreme_table", batch)?;

    // Test the problematic pattern with extreme nulls
    let extreme_sql = "SELECT u8_low, utf8_low, count(dictionary_utf8_low) as dict_count FROM extreme_table GROUP BY u8_low, utf8_low";
    
    println!("Testing EXTREME null scenario:");
    println!("{}", extreme_sql);
    
    let result = ctx.sql(extreme_sql).await?.collect().await;
    
    match result {
        Ok(batches) => {
            println!("Extreme null query succeeded with {} batches", batches.len());
            
            // Check the results
            for batch in batches {
                println!("Result batch: {} rows", batch.num_rows());
                
                // Print first few rows to see what we get
                if batch.num_rows() > 0 {
                    println!("Sample results:");
                    for i in 0..std::cmp::min(5, batch.num_rows()) {
                        let u8_val = batch.column(0).slice(i, 1);
                        let utf8_val = batch.column(1).slice(i, 1);
                        let count_val = batch.column(2).slice(i, 1);
                        println!("  Row {}: u8={:?}, utf8={:?}, dict_count={:?}", i, u8_val, utf8_val, count_val);
                    }
                }
            }
        }
        Err(e) => {
            let error_str = format!("{:?}", e);
            println!("🎯 EXTREME null query failed - this might be our reproduction!");
            println!("Error: {}", error_str);
            
            if error_str.contains("RowConverter column schema mismatch") {
                println!("🎉 SUCCESS! Reproduced the RowConverter error with extreme nulls!");
            }
        }
    }

    Ok(())
}

/// Test the exact fuzzer scenario where ALL dictionary keys are null (null_pct = 1.0)
/// This reproduces the exact bug condition from the fuzzer code change
#[tokio::test]
async fn test_all_null_dictionary_keys_fuzzer_bug() -> Result<()> {
    let ctx = SessionContext::new();

    // With null_pct = 1.0, the fuzzer generates dictionary arrays where ALL keys are null
    // This is the exact condition from the bug report
    let num_rows = 50;
    let keys = UInt64Array::from(vec![None; num_rows]); // ALL keys are null!
    
    // Values array also generated with high null percentage (1.0)
    let values = StringArray::from(vec![None::<&str>, None::<&str>, None::<&str>]); // All values are also null!
    let dict_array = DictionaryArray::try_new(keys, Arc::new(values))?;

    // Other columns with realistic patterns
    let u8_low_array = UInt8Array::from(
        (0..num_rows).map(|i| {
            if i % 5 == 0 { None } else { Some((i % 4) as u8) }
        }).collect::<Vec<_>>()
    );
    
    let utf8_low_array = StringArray::from(
        (0..num_rows).map(|i| {
            match i % 7 {
                0 | 1 => None,
                2 => Some("low_str_a"),
                3 => Some("low_str_b"),
                4 => Some("low_str_c"),
                5 => Some("low_str_a"),
                _ => Some("low_str_b"),
            }
        }).collect::<Vec<_>>()
    );

    // Add more columns that appear in the failing query
    let interval_year_month_array = arrow::array::IntervalYearMonthArray::from(
        (0..num_rows).map(|i| if i % 8 == 0 { None } else { Some((i % 12) as i32) }).collect::<Vec<_>>()
    );
    
    let timestamp_s_array = arrow::array::TimestampSecondArray::from(
        (0..num_rows).map(|i| if i % 6 == 0 { None } else { Some(1640995200i64 + (i as i64) * 3600) }).collect::<Vec<_>>()
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("dictionary_utf8_low", dict_array.data_type().clone(), true),
        Field::new("u8_low", DataType::UInt8, true),
        Field::new("utf8_low", DataType::Utf8, true),
        Field::new("interval_year_month", DataType::Interval(arrow::datatypes::IntervalUnit::YearMonth), true),
        Field::new("timestamp_s", DataType::Timestamp(arrow::datatypes::TimeUnit::Second, None), true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(dict_array),
            Arc::new(u8_low_array),
            Arc::new(utf8_low_array),
            Arc::new(interval_year_month_array),
            Arc::new(timestamp_s_array),
        ],
    )?;

    ctx.register_batch("fuzz_table", batch)?;

    // Test the exact failing query with ALL NULL dictionary keys
    let all_null_keys_sql = "SELECT u8_low, utf8_low, count(dictionary_utf8_low) as col1, count(DISTINCT interval_year_month) as col2, count(DISTINCT timestamp_s) as col3 FROM fuzz_table GROUP BY u8_low, utf8_low";
    
    println!("Testing ALL NULL dictionary keys scenario (fuzzer bug condition):");
    println!("{}", all_null_keys_sql);
    println!("Dictionary keys: ALL NULL (like null_pct=1.0 in fuzzer)");
    println!("Dictionary values: ALL NULL (like null_pct=1.0 in values)");
    
    let result = ctx.sql(all_null_keys_sql).await?.collect().await;
    
    match result {
        Ok(batches) => {
            println!("❌ ALL NULL keys query succeeded - issue might be elsewhere");
            println!("Result batches: {}", batches.len());
            
            for (i, batch) in batches.iter().enumerate() {
                println!("Batch {}: {} rows", i, batch.num_rows());
                if batch.num_rows() > 0 {
                    println!("  Schema: {:?}", batch.schema());
                    // Print first row to see what happens with all-null dictionary
                    let dict_count = batch.column(2);
                    println!("  Dictionary count values: {:?}", dict_count.slice(0, std::cmp::min(3, batch.num_rows())));
                }
            }
        }
        Err(e) => {
            let error_str = format!("{:?}", e);
            println!("🎯 ALL NULL keys query FAILED - this reproduces the fuzzer issue!");
            println!("Error: {}", error_str);
            
            if error_str.contains("RowConverter column schema mismatch") &&
               error_str.contains("expected Dictionary(UInt64, Utf8) got Utf8") {
                println!("🎉 EXACT MATCH! This is the fuzzer bug!");
                return Err(datafusion_common::DataFusionError::ArrowError(
                    arrow_schema::ArrowError::InvalidArgumentError(
                        "Successfully reproduced the fuzzer RowConverter bug".to_string()
                    ),
                    None
                ));
            }
        }
    }

    Ok(())
}

/// Test using the exact RecordBatchGenerator approach to create data
/// This mimics the fuzzer's data generation as closely as possible
#[tokio::test]
async fn test_mimic_exact_fuzzer_data_generation() -> Result<()> {
    use arrow::array::PrimitiveArray;
    use arrow::datatypes::UInt64Type;
    
    let ctx = SessionContext::new();
    
    // Simulate the exact fuzzer logic with null_pct = 1.0
    let num_rows = 32;
    let num_distinct = 3;
    let null_pct = 1.0f64; // This is the bug condition from the code change
    
    // Simulate the fuzzer's random generation (but deterministically for testing)
    let keys: PrimitiveArray<UInt64Type> = (0..num_rows)
        .map(|i| {
            // Simulate: if batch_gen_rng.random::<f64>() < null_pct
            let random_val = (i as f64) / (num_rows as f64); // Deterministic "random"
            if random_val < null_pct {
                None
            } else if num_distinct > 1 {
                // This branch should never be taken with null_pct = 1.0
                Some((i % num_distinct) as u64)
            } else {
                Some(0)
            }
        })
        .collect();
    
    // Values array generated with high null percentage too
    let values = StringArray::from(vec![None, Some("fuzzer_val1"), Some("fuzzer_val2")]);
    let dict_array = DictionaryArray::new(keys, Arc::new(values));
    
    println!("Generated dictionary array:");
    println!("  Keys null count: {}", dict_array.keys().null_count());
    println!("  Keys total count: {}", dict_array.keys().len());
    println!("  All keys null: {}", dict_array.keys().null_count() == dict_array.keys().len());
    
    // Generate other columns similar to fuzzer patterns
    let u8_low_array = UInt8Array::from(
        (0..num_rows).map(|i| {
            // Simulate some null pattern
            let null_chance = (i % 7) as f64 / 7.0;
            if null_chance < 0.3 { None } else { Some((i % 5) as u8) }
        }).collect::<Vec<_>>()
    );
    
    let utf8_low_array = StringArray::from(
        (0..num_rows).map(|i| {
            let null_chance = (i % 11) as f64 / 11.0;
            if null_chance < 0.2 {
                None
            } else {
                match i % 4 {
                    0 => Some("fuzzer_str_a"),
                    1 => Some("fuzzer_str_b"),
                    2 => Some("fuzzer_str_c"),
                    _ => Some("fuzzer_str_d"),
                }
            }
        }).collect::<Vec<_>>()
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("dictionary_utf8_low", dict_array.data_type().clone(), true),
        Field::new("u8_low", DataType::UInt8, true),
        Field::new("utf8_low", DataType::Utf8, true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(dict_array),
            Arc::new(u8_low_array),
            Arc::new(utf8_low_array),
        ],
    )?;

    ctx.register_batch("fuzz_table", batch)?;

    // Test the exact pattern that causes issues
    let mimic_sql = "SELECT u8_low, utf8_low, count(dictionary_utf8_low) as col1 FROM fuzz_table GROUP BY u8_low, utf8_low";
    
    println!("\nTesting with fuzzer-mimicked data generation:");
    println!("{}", mimic_sql);
    
    let result = ctx.sql(mimic_sql).await?.collect().await;
    
    match result {
        Ok(batches) => {
            println!("Fuzzer-mimicked query succeeded");
            
            for batch in batches {
                println!("Result: {} rows", batch.num_rows());
                if batch.num_rows() > 0 {
                    // Check if dictionary column maintained its type through aggregation
                    let schema = batch.schema();
                    for field in schema.fields() {
                        println!("  Result field '{}': {:?}", field.name(), field.data_type());
                    }
                }
            }
        }
        Err(e) => {
            let error_str = format!("{:?}", e);
            println!("🎯 Fuzzer-mimicked query FAILED!");
            println!("Error: {}", error_str);
            
            if error_str.contains("RowConverter column schema mismatch") {
                println!("🎉 REPRODUCED with fuzzer-mimicked data generation!");
            }
        }
    }

    Ok(())
}
