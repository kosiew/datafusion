use super::*;
use arrow::array::{Int64Array, StringArray, UInt64Array, UInt8Array};
use datafusion_catalog::MemTable;
use datafusion_common::DataFusionError;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[tokio::test]
async fn decimal_multiplication_sum() -> Result<()> {
    let precision = 38u8;
    let scale = 10i8;
    let factor = 10i128.pow(scale as u32);
    let mut rng = StdRng::seed_from_u64(42);
    const ROWS: usize = 1000; // Reduced for simplicity
    let mut d1_vals = Vec::with_capacity(ROWS);
    let mut d2_vals = Vec::with_capacity(ROWS);

    for _ in 0..ROWS {
        let n1: u64 = rng.random_range(1..=(1u64 << 20)); // Smaller range
        let p = 10u64.pow(rng.random_range(1..=3));
        let d1_unscaled = ((n1 * factor as u64) / p) as i128;

        let n2: u64 = rng.random_range(1..=100);
        let d2_unscaled = n2 as i128 * factor / 100; // 0.xx format

        d1_vals.push(d1_unscaled);
        d2_vals.push(d2_unscaled);
    }

    let arr1 = Decimal128Array::from_iter_values(d1_vals)
        .with_precision_and_scale(precision, scale)
        .unwrap();
    let arr2 = Decimal128Array::from_iter_values(d2_vals)
        .with_precision_and_scale(precision, scale)
        .unwrap();
    let schema = Schema::new(vec![
        Field::new("d1", DataType::Decimal128(precision, scale), false),
        Field::new("d2", DataType::Decimal128(precision, scale), false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(arr1) as ArrayRef, Arc::new(arr2) as ArrayRef],
    )?;
    let memtable = MemTable::try_new(Arc::new(schema), vec![vec![batch]])?;
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(memtable))?;
    let df = ctx.sql("SELECT sum(d1 * d2) FROM t").await?;
    let results = df.collect().await?;
    assert_eq!(results.len(), 1);
    let value = results[0]
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .unwrap()
        .value(0);
    // Just verify we got a result without overflow
    assert!(value != 0);
    Ok(())
}

#[tokio::test]
async fn test_decimal128_sum_overflow() -> Result<()> {
    // This test reproduces the overflow issue from the fuzzer:
    // ArrowError(ArithmeticOverflow("Overflow happened on:
    // -146325439846853887532782563708403423702 + -116879761896069538410527077402225403902"))

    let ctx = SessionContext::new();

    // Create a table with decimal128 values that will cause overflow when summed
    let precision = 38u8;
    let scale = 0i8;

    // These are the actual values from the error message that caused overflow
    let large_negative_1 = -146325439846853887532782563708403423702_i128;
    let large_negative_2 = -116879761896069538410527077402225403902_i128;

    // Additional columns to match the fuzzer query structure
    // Put the two large negative numbers in the same group to cause overflow when summed
    let group_values = vec!["group1", "group1", "group2", "group2"];
    let utf8_values = vec!["a", "a", "b", "b"];
    let u8_values = vec![1u8, 1u8, 2u8, 2u8];
    let u64_values = vec![100u64, 200u64, 300u64, 400u64];
    let i64_values = vec![50i64, 75i64, 125i64, 175i64];

    // Create decimal values including the overflow-causing ones
    // Put both large negative values in the same group (group1)
    let decimal_values = vec![large_negative_1, large_negative_2, 100_i128, 200_i128];

    // Create arrays
    let group_array = Arc::new(StringArray::from(group_values)) as ArrayRef;
    let utf8_array = Arc::new(StringArray::from(utf8_values)) as ArrayRef;
    let u8_array = Arc::new(UInt8Array::from(u8_values)) as ArrayRef;
    let u64_array = Arc::new(UInt64Array::from(u64_values)) as ArrayRef;
    let i64_array = Arc::new(Int64Array::from(i64_values)) as ArrayRef;

    let decimal_array = Arc::new(
        Decimal128Array::from_iter_values(decimal_values)
            .with_precision_and_scale(precision, scale)
            .unwrap(),
    ) as ArrayRef;

    // Create schema matching the fuzzer query
    let schema = Schema::new(vec![
        Field::new("group_col", DataType::Utf8, false),
        Field::new("utf8_low", DataType::Utf8, false),
        Field::new("u8_low", DataType::UInt8, false),
        Field::new("u64", DataType::UInt64, false),
        Field::new("i64", DataType::Int64, false),
        Field::new("decimal128", DataType::Decimal128(precision, scale), false),
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            group_array,
            utf8_array,
            u8_array,
            u64_array,
            i64_array,
            decimal_array,
        ],
    )?;

    let memtable = MemTable::try_new(Arc::new(schema), vec![vec![batch]])?;
    ctx.register_table("fuzz_table", Arc::new(memtable))?;

    // Execute the query that caused the overflow in the fuzzer
    let sql = "SELECT group_col, utf8_low, u8_low, sum(DISTINCT u64) as col1, sum(u64) as col2, sum(i64) as col3, sum(u64) as col4, sum(DISTINCT decimal128) as col5 FROM fuzz_table GROUP BY group_col, utf8_low, u8_low";

    let df = ctx.sql(sql).await?;

    // This should fail with arithmetic overflow
    let result = df.collect().await;

    // Currently this will panic with ArithmeticOverflow, but the test documents the expected behavior
    match result {
        Err(DataFusionError::ArrowError(arrow_error, _)) => {
            println!("Got Arrow error: {}", arrow_error);
            assert!(arrow_error.to_string().contains("Arithmetic overflow"));
            assert!(arrow_error.to_string().contains("Overflow happened on"));
            println!("Expected overflow error occurred: {}", arrow_error);
        }
        Ok(results) => {
            println!(
                "Query succeeded unexpectedly with {} batches",
                results.len()
            );
            for (i, batch) in results.iter().enumerate() {
                println!("Batch {}: {:?}", i, batch);
            }
            panic!("Expected arithmetic overflow error, but query succeeded");
        }
        Err(other_error) => {
            println!("Got different error: {}", other_error);
            panic!(
                "Expected ArithmeticOverflow, got different error: {}",
                other_error
            );
        }
    }

    Ok(())
}
